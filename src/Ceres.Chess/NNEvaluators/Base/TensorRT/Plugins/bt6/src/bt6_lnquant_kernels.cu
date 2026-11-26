// k3: BT6LnQuant v1 -- residual add + LayerNorm + static INT8 quantization of Kovax's BT6 flagship (1024 wide), one kernel
// for the chain TensorRT ran as one Myelin kernel in k2 (per-token GEMM output * r + bias, * alpha, + residual, FP32
// LayerNorm with the SmoothQuant prescale folded into gamma / beta, QuantizeLinear(al)):
//   t = fp16(yn * r + bias), s = fp16(alpha * t + residual)   (FP32 ops, TensorRT's rounding points)
//   s = residual + t                           FP16 -> output 0 (the updated residual stream)
//   y = (s - mean) / sqrt(var + eps) * gamma + beta    FP32, two-pass mean / variance over the 1024 channels
//   codes = saturate_int8(rint(y / al))        -> output 1 (read by DequantizeLinear(al) -> MatMul)
// tab FP32 [3 * 1024] = bias | gamma | beta (bias holds FP16 values).
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdlib>
#include "bt6_kernels.h"

namespace {

// v2 (k3 step 2b): 128 threads per row (8 channels each, one 16-byte load per operand), kRowsPerCta rows in flight per CTA,
// a grid-stride loop over row groups so that bias / gamma / beta stay in registers; mean and variance reduced by warp
// shuffles + a 4-warp shared-memory step (two-pass, FP32). The v1 one-warp-per-row kernel was 1.1-2x slower than TensorRT's.
constexpr int kN = 1024;
constexpr int kTpr = 128;          // threads per row
constexpr int kRowsPerCta = 2;     // rows in flight per CTA
constexpr int kWpr = kTpr / 32;    // warps per row

__device__ __forceinline__ void ld8f(const float* p, float (&v)[8]) {
  const float4 a = __ldg(reinterpret_cast<const float4*>(p)), b = __ldg(reinterpret_cast<const float4*>(p) + 1);
  v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
}

// kR rows per row slot per iteration: their loads are issued together and their reductions share the barriers.
template <int kR>
__global__ void __launch_bounds__(kTpr * kRowsPerCta) bt6_ln_quant(__half* __restrict__ res_out, int8_t* __restrict__ codes,
                                                                  const __half* __restrict__ yn, const __half* __restrict__ r,
                                                                  const __half* __restrict__ res, const float* __restrict__ tab,
                                                                  __half alpha, float eps, float scale, int rows) {
  __shared__ float red[2][kRowsPerCta][kR][kWpr];
  const float inv_scale = 1.f / scale;
  const int t = threadIdx.x % kTpr, slot = threadIdx.x / kTpr, lane = threadIdx.x & 31, w = t >> 5;
  const int col = 8 * t;
  float b[8], g[8], be[8];
  ld8f(tab + col, b);
  ld8f(tab + kN + col, g);
  ld8f(tab + 2 * kN + col, be);
#ifdef LNQ_CONTROL_NO_BIAS  // must-fail control: the GEMM bias left out
#pragma unroll
  for (int j = 0; j < 8; ++j) b[j] = 0.f;
#endif
  const float af = __half2float(alpha);
  constexpr int kStep = kRowsPerCta * kR;
  for (int row0 = blockIdx.x * kStep; row0 < rows; row0 += gridDim.x * kStep) {
    uint4 y16[kR], s16[kR];
    float rf[kR];
    bool live[kR];
    size_t base[kR];
#pragma unroll
    for (int k = 0; k < kR; ++k) {
      const int row = row0 + k * kRowsPerCta + slot;
      live[k] = row < rows;   // dead rows still take the barriers
      base[k] = (size_t)(live[k] ? row : 0) * kN + col;
      y16[k] = __ldg(reinterpret_cast<const uint4*>(yn + base[k]));
      s16[k] = __ldg(reinterpret_cast<const uint4*>(res + base[k]));
      rf[k] = r ? __half2float(r[live[k] ? row : 0]) : 1.f;
    }
    float x[kR][8], sum[kR];
#pragma unroll
    for (int k = 0; k < kR; ++k) {
      const __half2* yp = reinterpret_cast<const __half2*>(&y16[k]);
      const __half2* sp = reinterpret_cast<const __half2*>(&s16[k]);
      uint4 o16;
      __half2* op = reinterpret_cast<__half2*>(&o16);
      sum[k] = 0.f;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        // TensorRT's rounding (k3 probe tiny_trt.py, 100 % of 4M values): t = fp16(fp32(yn r) + b), then
        // s = fp16(fp32(alpha t) + res); separate FP32 ops, no FMA contraction
        const float2 y = __half22float2(yp[j]);
        const __half2 tt = __floats2half2_rn(__fadd_rn(__fmul_rn(y.x, rf[k]), b[2 * j]), __fadd_rn(__fmul_rn(y.y, rf[k]), b[2 * j + 1]));
        const float2 tf = __half22float2(tt), rs = __half22float2(sp[j]);
        const __half2 sv = __floats2half2_rn(__fadd_rn(__fmul_rn(af, tf.x), rs.x), __fadd_rn(__fmul_rn(af, tf.y), rs.y));
        op[j] = sv;
        const float2 f = __half22float2(sv);
        x[k][2 * j] = f.x;
        x[k][2 * j + 1] = f.y;
        sum[k] += f.x + f.y;
      }
      if (live[k]) *reinterpret_cast<uint4*>(res_out + base[k]) = o16;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
#pragma unroll
      for (int k = 0; k < kR; ++k) sum[k] += __shfl_xor_sync(0xffffffffu, sum[k], o);
    if (lane == 0)
#pragma unroll
      for (int k = 0; k < kR; ++k) red[0][slot][k][w] = sum[k];
    __syncthreads();
    float ss[kR];
#pragma unroll
    for (int k = 0; k < kR; ++k) {
      const float mean = (red[0][slot][k][0] + red[0][slot][k][1] + red[0][slot][k][2] + red[0][slot][k][3]) * (1.f / kN);
      ss[k] = 0.f;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        x[k][j] -= mean;
        ss[k] += x[k][j] * x[k][j];
      }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
#pragma unroll
      for (int k = 0; k < kR; ++k) ss[k] += __shfl_xor_sync(0xffffffffu, ss[k], o);
    if (lane == 0)
#pragma unroll
      for (int k = 0; k < kR; ++k) red[1][slot][k][w] = ss[k];
    __syncthreads();
    // (red[0] of the next iteration is written after this one's second barrier, red[1] after the next first: no race)
#pragma unroll
    for (int k = 0; k < kR; ++k) {
      const float sd = sqrtf((red[1][slot][k][0] + red[1][slot][k][1] + red[1][slot][k][2] + red[1][slot][k][3]) * (1.f / kN) + eps);
      uint2 out;
      int8_t* q = reinterpret_cast<int8_t*>(&out);
#ifdef LNQ_EXACT_DIV  // probe: IEEE divisions as the ONNX ops are written (2 per element; ~25 % slower)
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float y = __fdiv_rn(x[k][j], sd) * g[j] + be[j];
        q[j] = static_cast<int8_t>(fminf(fmaxf(rintf(__fdiv_rn(y, scale)), -128.f), 127.f));
      }
#else  // x * (1 / sd) (1 ulp of y at most) and y * inv_scale (exact: the scale is a power of two, checked on the host)
      const float isd = __frcp_rn(sd);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float y = x[k][j] * isd * g[j] + be[j];
        q[j] = static_cast<int8_t>(fminf(fmaxf(rintf(y * inv_scale), -128.f), 127.f));
      }
#endif
      if (live[k]) *reinterpret_cast<uint2*>(codes + base[k]) = out;
    }
  }
}

}  // namespace

namespace bt6 {

int launch_ln_quant(__half* res_out, int8_t* codes, const __half* yn, const __half* r, const __half* res, const float* tab,
                    float alpha, float eps, float scale, int rows, cudaStream_t stream) {
  if (rows <= 0) return 0;
  static int sms = 0;
  if (sms == 0) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    if (sms <= 0) sms = 128;
  }
  static const int force_r = [] { const char* e = std::getenv("BT6_LNQ_R"); return e ? std::atoi(e) : 0; }();
  // rows per slot per iteration: enough CTAs to fill the GPU first, then batching (BT6_LNQ_R=1/2/4 forces)
  const int kr = force_r ? force_r : rows >= sms * 64 ? 2 : 1;  // nsys: R2 only helps at b256 (R4 never)
  const int step = kRowsPerCta * kr, groups = (rows + step - 1) / step, cap = sms * (2048 / (kTpr * kRowsPerCta));
  const int grid = groups < cap ? groups : cap;
  const __half a16 = __float2half_rn(alpha);
  if (kr >= 4) bt6_ln_quant<4><<<grid, kTpr * kRowsPerCta, 0, stream>>>(res_out, codes, yn, r, res, tab, a16, eps, scale, rows);
  else if (kr == 2) bt6_ln_quant<2><<<grid, kTpr * kRowsPerCta, 0, stream>>>(res_out, codes, yn, r, res, tab, a16, eps, scale, rows);
  else bt6_ln_quant<1><<<grid, kTpr * kRowsPerCta, 0, stream>>>(res_out, codes, yn, r, res, tab, a16, eps, scale, rows);
  return static_cast<int>(cudaGetLastError());
}

}  // namespace bt6

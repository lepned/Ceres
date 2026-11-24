// Round 3: BT6GluQuant kernel. SwiGLU of the INT8 FFN gate / up GEMM outputs plus the dynamic per-token INT8
// quantization of the FFN down input, in one pass over each row:
//   g = qg * gate_scale + bias,  u = qu * up_scale,  m = g * sigmoid(g) * u                     (FP32)
//   r = max(max_j |m_j|, 1e-3),  codes_j = round(127 * m_j / r) clamped to +-127,  rmax = r (FP16)
// The graph then dequantizes the codes with the constant scale 1/127, runs the INT8 FFN down GEMM and multiplies the
// output rows by rmax (the same arithmetic as the graph-only per-token variant, which TensorRT/Myelin ran as a two-pass
// kernel at ~2x this kernel's cost).
// One warp per row of N = 1536 channels: 6 chunks of 8 consecutive channels per lane (8-byte INT8 loads / stores).
#include <cuda_fp16.h>
#include <cstdint>
#include "bt6_kernels.h"

namespace {

constexpr int kN = bt6::kGluChannels;  // 1536
constexpr int kChunks = kN / (32 * 8);
static_assert(kChunks * 32 * 8 == kN, "N must be a multiple of 256");
constexpr int kRowsPerCta = 8;

#ifdef GLU_CONTROL_NO_SIGMOID  // the fidelity gate's CONTROL (must fail): g * u instead of SiLU(g) * u
__device__ __forceinline__ float silu_f(float x) { return x; }
#else
__device__ __forceinline__ float silu_f(float x) { return x / (1.f + __expf(-x)); }
#endif

__global__ void __launch_bounds__(32 * kRowsPerCta) bt6_glu_quant(int8_t* __restrict__ codes, __half* __restrict__ rmax,
                                                                 const int8_t* __restrict__ qg, const int8_t* __restrict__ qu,
                                                                 const __half* __restrict__ bias, float gs, float us,
                                                                 int rows) {
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * kRowsPerCta + (threadIdx.x >> 5);
  if (row >= rows) return;
  const size_t base = (size_t)row * kN;
  float m[kChunks][8];
  float mx = 0.f;
#pragma unroll
  for (int c = 0; c < kChunks; ++c) {
    const int col = (c * 32 + lane) * 8;
    const uint2 g8 = __ldg(reinterpret_cast<const uint2*>(qg + base + col));
    const uint2 u8 = __ldg(reinterpret_cast<const uint2*>(qu + base + col));
    const uint4 b16 = __ldg(reinterpret_cast<const uint4*>(bias + col));
    const int8_t* gp = reinterpret_cast<const int8_t*>(&g8);
    const int8_t* up = reinterpret_cast<const int8_t*>(&u8);
    const __half* bp = reinterpret_cast<const __half*>(&b16);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float g = (float)gp[j] * gs + __half2float(bp[j]);
      const float v = silu_f(g) * ((float)up[j] * us);
      m[c][j] = v;
      mx = fmaxf(mx, fabsf(v));
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
  const float r = fmaxf(mx, 1e-3f);
  const __half r16 = __float2half_rn(r);
  // quantize with the FP16-rounded r, which is what the graph multiplies the GEMM output by
  const float inv = 127.f / __half2float(r16);
#pragma unroll
  for (int c = 0; c < kChunks; ++c) {
    const int col = (c * 32 + lane) * 8;
    uint2 out;
    int8_t* op = reinterpret_cast<int8_t*>(&out);
#pragma unroll
    for (int j = 0; j < 8; ++j) op[j] = (int8_t)fminf(fmaxf(rintf(m[c][j] * inv), -127.f), 127.f);
    *reinterpret_cast<uint2*>(codes + base + col) = out;
  }
  if (lane == 0) rmax[row] = r16;
}

}  // namespace

namespace bt6 {

int launch_glu_quant(int8_t* codes, __half* rmax, const int8_t* qg, const int8_t* qu, const __half* bias, float gate_scale,
                     float up_scale, int rows, cudaStream_t stream) {
  if (rows <= 0) return 0;
  const int grid = (rows + kRowsPerCta - 1) / kRowsPerCta;
  bt6_glu_quant<<<grid, 32 * kRowsPerCta, 0, stream>>>(codes, rmax, qg, qu, bias, gate_scale, up_scale, rows);
  return static_cast<int>(cudaGetLastError());
}

}  // namespace bt6

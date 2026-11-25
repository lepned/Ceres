// k2: dynamic per-token INT8 quantization kernels for the INT8 Kovax BT6 flagship (1024 wide), one warp per row:
//   BT6RowQuant v1     x FP16 [rows, 1024]                     -> codes = round(127 x / r), r = max(max_j |x_j|, eps)
//   BT6SigGluQuant v1  p FP16 [rows, 2048] = [gate | up + bu]  -> m = sigmoid(gate) * up, codes / r of m as above
// codes INT8 [rows, 1024], r FP16 [rows] (the graph dequantizes the codes with 1/127, runs the INT8 GEMM and multiplies
// the GEMM output rows by r). The codes are made with the FP16-rounded r the graph multiplies by.
#include <cuda_fp16.h>
#include <cstdint>
#include "bt6_kernels.h"

namespace {

constexpr int kN = 1024;
constexpr int kChunks = kN / (32 * 8);  // 4 chunks of 8 channels per lane
constexpr int kRows = 8;                // rows (warps) per CTA

#ifdef ROWQ_CONTROL_NO_SIGMOID  // must-fail control: m = gate * up
__device__ __forceinline__ float sig_f(float x) { return x; }
#else
__device__ __forceinline__ float sig_f(float x) { return 1.f / (1.f + __expf(-x)); }
#endif

template <bool kGlu>
__global__ void __launch_bounds__(32 * kRows) bt6_row_quant(int8_t* __restrict__ codes, __half* __restrict__ rmax,
                                                           const __half* __restrict__ x, float eps, int rows) {
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * kRows + (threadIdx.x >> 5);
  if (row >= rows) return;
  const size_t in_base = (size_t)row * (kGlu ? 2 * kN : kN);
  float m[kChunks][8];
  float mx = 0.f;
#pragma unroll
  for (int c = 0; c < kChunks; ++c) {
    const int col = (c * 32 + lane) * 8;
    const uint4 a16 = __ldg(reinterpret_cast<const uint4*>(x + in_base + col));
    const __half2* ap = reinterpret_cast<const __half2*>(&a16);
    if (kGlu) {
      const uint4 u16 = __ldg(reinterpret_cast<const uint4*>(x + in_base + kN + col));
      const __half2* up = reinterpret_cast<const __half2*>(&u16);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 g = __half22float2(ap[j]);
        const float2 u = __half22float2(up[j]);
        m[c][2 * j] = sig_f(g.x) * u.x;
        m[c][2 * j + 1] = sig_f(g.y) * u.y;
      }
    } else {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 v = __half22float2(ap[j]);
        m[c][2 * j] = v.x;
        m[c][2 * j + 1] = v.y;
      }
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) mx = fmaxf(mx, fabsf(m[c][j]));
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
  const __half r16 = __float2half_rn(fmaxf(mx, eps));
  const float inv = 127.f / __half2float(r16);
  const size_t out_base = (size_t)row * kN;
#pragma unroll
  for (int c = 0; c < kChunks; ++c) {
    const int col = (c * 32 + lane) * 8;
    uint2 out;
    int8_t* op = reinterpret_cast<int8_t*>(&out);
#pragma unroll
    for (int j = 0; j < 8; ++j) op[j] = (int8_t)fminf(fmaxf(rintf(m[c][j] * inv), -127.f), 127.f);
    *reinterpret_cast<uint2*>(codes + out_base + col) = out;
  }
  if (lane == 0) rmax[row] = r16;
}

}  // namespace

namespace bt6 {

int launch_row_quant(int8_t* codes, __half* rmax, const __half* x, int glu, float eps, int rows, cudaStream_t stream) {
  if (rows <= 0) return 0;
  const int grid = (rows + kRows - 1) / kRows;
  if (glu)
    bt6_row_quant<true><<<grid, 32 * kRows, 0, stream>>>(codes, rmax, x, eps, rows);
  else
    bt6_row_quant<false><<<grid, 32 * kRows, 0, stream>>>(codes, rmax, x, eps, rows);
  return static_cast<int>(cudaGetLastError());
}

}  // namespace bt6

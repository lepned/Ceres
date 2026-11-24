// BT6EdgeSite (round 2): one edge-state update site of the flat 768x16 trunk in ONE kernel
// (bt6_net.BT6EdgeSite.forward without the triplet branch, channels-last, as exported):
//
//   e_hat = e + R . Hraw              R = readback [16, heads]; Hraw = the layer's raw logits (q.k / sqrt(hd) + bias)
//   n     = rms(e_hat)                over the 16 channels, eps 1e-6, no gain
//   h     = relu(W1 n + Wr n^T + b1)  n^T = n of the transposed cell (j, i); hidden 64
//   e'    = rms(e_hat + W2 h)
// FP32 arithmetic (the export runs the maps in FP16 with FP32 islands for the rms), FP16 in / out.
//
// Inputs: state FP16 [B, 64, 64, 16] cell-major, hraw FP16 [B, heads, 64, 64], tab FP32 [3520]:
//   R [16][24] | W1 [64][16] | Wr [64][16] | b1 [64] | W2 [16][64]
// Output: state' FP16 [B, 64, 64, 16] cell-major.
// A CTA owns 8 source rows of one position (512 cells, one per thread) and also builds n for the 64 x 8 mirrored
// cells its rows' n^T reads (the column strip), so the transpose never leaves shared memory.
#include <cuda_fp16.h>
#include <cstdint>
#include "bt6_kernels.h"

namespace {

constexpr int kC = 16, kHid = 64, kHeads = 24, kRowsPerCta = 8, kThreads = 64 * kRowsPerCta;
constexpr int kOffR = 0, kOffW1 = kOffR + kC * kHeads, kOffWr = kOffW1 + kHid * kC, kOffB1 = kOffWr + kHid * kC;
constexpr int kOffW2 = kOffB1 + kHid, kTabN = kOffW2 + kC * kHid;

__device__ __forceinline__ void load_state(const __half* p, float (&x)[kC]) {
  const uint4 a = __ldg(reinterpret_cast<const uint4*>(p)), b = __ldg(reinterpret_cast<const uint4*>(p) + 1);
  const __half2* ha = reinterpret_cast<const __half2*>(&a);
  const __half2* hb = reinterpret_cast<const __half2*>(&b);
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const float2 fa = __half22float2(ha[k]), fb = __half22float2(hb[k]);
    x[2 * k] = fa.x; x[2 * k + 1] = fa.y; x[8 + 2 * k] = fb.x; x[9 + 2 * k] = fb.y;
  }
}

// e_hat of cell (x, y) of position b
__device__ __forceinline__ void e_hat_of(const __half* __restrict__ state, const __half* __restrict__ hraw,
                                         const float* sR, int b, int x, int y, float (&e)[kC]) {
  load_state(state + (((size_t)b * 64 + x) * 64 + y) * kC, e);
  const __half* hp = hraw + ((size_t)b * kHeads * 64 + x) * 64 + y;
#pragma unroll
  for (int h = 0; h < kHeads; ++h) {
    const float v = __half2float(__ldg(hp + (size_t)h * 4096));
#pragma unroll
    for (int c = 0; c < kC; ++c) e[c] = fmaf(sR[c * kHeads + h], v, e[c]);
  }
}

__device__ __forceinline__ float rms_scale(const float (&x)[kC], float eps) {
  float s = 0.f;
#pragma unroll
  for (int c = 0; c < kC; ++c) s = fmaf(x[c], x[c], s);
  return rsqrtf(s * (1.f / kC) + eps);
}

__global__ void __launch_bounds__(kThreads) bt6_edge_site(__half* __restrict__ out, const __half* __restrict__ state,
                                                          const __half* __restrict__ hraw, const float* __restrict__ tab,
                                                          float eps) {
  __shared__ __align__(16) float sW[kTabN];
  __shared__ __align__(16) float sNT[kRowsPerCta][kC][64];  // n of cell (j, i0 + r) at [r][c][j]
  const int b = blockIdx.x, i0 = blockIdx.y * kRowsPerCta, t = threadIdx.x;
  for (int k = t; k < kTabN; k += kThreads) sW[k] = __ldg(tab + k);
  __syncthreads();
  const float* sR = sW + kOffR;
  {  // the column strip: cell (jj, i0 + r)
    const int jj = t >> 3, r = t & 7;
    float e[kC];
    e_hat_of(state, hraw, sR, b, jj, i0 + r, e);
    const float s = rms_scale(e, eps);
#pragma unroll
    for (int c = 0; c < kC; ++c) sNT[r][c][jj] = e[c] * s;
  }
  // this thread's own cell (i, j)
  const int r = t >> 6, i = i0 + r, j = t & 63;
  float e[kC];
  e_hat_of(state, hraw, sR, b, i, j, e);
  float n[kC];
  {
    const float s = rms_scale(e, eps);
#pragma unroll
    for (int c = 0; c < kC; ++c) n[c] = e[c] * s;
  }
  __syncthreads();
  float nt[kC];
#pragma unroll
  for (int c = 0; c < kC; ++c) nt[c] = sNT[r][c][j];
#ifdef SITE_CONTROL_NO_REV  // the fidelity gate's CONTROL (must fail): the reverse-edge term dropped
#pragma unroll
  for (int c = 0; c < kC; ++c) nt[c] = 0.f;
#endif
  const float* W1 = sW + kOffW1;
  const float* Wr = sW + kOffWr;
  const float* b1 = sW + kOffB1;
  const float* W2 = sW + kOffW2;
#pragma unroll 4
  for (int k = 0; k < kHid; ++k) {
    float hk = b1[k];
#pragma unroll
    for (int c = 0; c < kC; ++c) hk = fmaf(W1[k * kC + c], n[c], fmaf(Wr[k * kC + c], nt[c], hk));
    hk = fmaxf(hk, 0.f);
#pragma unroll
    for (int c = 0; c < kC; ++c) e[c] = fmaf(W2[c * kHid + k], hk, e[c]);
  }
  const float s = rms_scale(e, eps);
  uint4 w[2];
  __half2* hw = reinterpret_cast<__half2*>(w);
#pragma unroll
  for (int k = 0; k < 8; ++k) hw[k] = __floats2half2_rn(e[2 * k] * s, e[2 * k + 1] * s);
  uint4* dst = reinterpret_cast<uint4*>(out + (((size_t)b * 64 + i) * 64 + j) * kC);
  dst[0] = w[0];
  dst[1] = w[1];
}

}  // namespace

namespace bt6 {

int launch_site(__half* out, const __half* state, const __half* hraw, const float* tab, int batch, cudaStream_t stream) {
  if (batch <= 0) return 0;
  dim3 grid(batch, 64 / kRowsPerCta);
  bt6_edge_site<<<grid, kThreads, 0, stream>>>(out, state, hraw, tab, 1e-6f);
  return static_cast<int>(cudaGetLastError());
}

}  // namespace bt6
static_assert(kTabN == bt6::kSiteTab32, "site table size");

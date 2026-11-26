// BT6TripletSite: one edge-state update site of Kovax's BT6 flagship (1024x24x32) WITH the triplet branch, three kernels
// (bt6_net.served_site + _triplet, channels-last as the r4 BT6EdgeSite):
//
//   e_hat  = e + R . H                    R = readback [16][heads]; H = the layer's attention logits (door-scaled, pre-softmax)
//   n      = rms(e_hat)                   over the 16 channels, eps 1e-6, no gain
//   v      = Wv n  [32]   eg = Wg n + bg [16] -> e_in, g_in, e_out, g_out (4 triplet heads each)
//   a_in [h,i,k] = softmax_k(e_in[h,i,k]) * sigmoid(g_in[h,i,k])
//   a_out[h,k,i] = softmax_k(e_out[h,k,i]) * sigmoid(g_out[h,k,i])          (softmax over the ROW axis k)
//   path: va_in[h,d,i,j] = sum_k a_in[h,i,k] v_in[h,d,k,j];  va_out[h,d,i,j] = sum_k a_out[h,k,i] v_out[h,d,j,k]
//   ag:   va_in reads v_in[h,d,j,k], va_out reads v_out[h,d,k,j]
//   e_hat2 = e_hat + Wo [va_in | va_out]   (va channel = direction * 16 + head * 4 + dot)
//   n2     = rms(e_hat2);  h = relu(W1 n2 + Wr n2^T + b1) [64];  e' = rms(e_hat2 + W2 h)
//
// Kernels (all on the plugin's stream; workspace = e_hat FP32 | e_hat FP16 | va FP16, 160 bytes per cell):
//   1 site_readback   one thread per cell: e_hat FP32 and its FP16 copy, both cell-major [B, 4096, 16]
//   2 site_triplet    one CTA (512 threads) per (sample, direction, triplet head): the head's 6 maps (logit, door, 4 values)
//                     on the tensor cores from the FP16 e_hat copy (weights split hi + lo FP16), softmax x sigmoid in FP32,
//                     va = A V on the tensor cores (FP16 operands, FP32 accumulation), va stored FP16 planar [B, 32, 4096].
//                     Structure adapted from lc0ex's triplet_mma.cu (the served fast form "fused", state_f16, dot fp16).
//   3 site_out_ffn    one CTA (256 threads, 2 cells each) per (sample, 8 rows): e_hat2 = e_hat + Wo va, rms, edge FFN with the reverse edge (the
//                     mirrored column strip's n2 built in shared memory, as r4's BT6EdgeSite), rms -> e' FP16 cell-major.
// FP32 arithmetic except the triplet's two MMA stages (class of the served lc0ex program).
//
// Controls (builds that MUST fail the fidelity gate): TRIPLET_CONTROL_SWAP_SOFTMAX runs both softmaxes over the wrong axis.
//
// k6 (2026-11-26, NOTES_k6.md): kernel 3 on the tensor cores (site_out_ffn_mma) with the layouts that feed it -- e_hat in
// MMA-fragment order (site_readback kPerm), va in 8 planes of [cell][4 dots] (site_triplet kCM). Kernels 1 and 2 compute
// the same values as before (bit for bit); kernel 3 differs from the FP32 kernel by the FP16 rounding of n2 and h.
// Env BT6_SITE_K5=1 runs k5's three kernels (bit-identical to k5); BT6_SITE_LO = mask of the weights' lo terms (default 7).
// Control: SITE_CONTROL_NO_MIRROR. Pricing / probe switches (off): SITE_K2_RET, SITE_K2_INFLIGHT.
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdlib>
#include "bt6_triplet_site.h"

#ifdef SITE_PHASE  // k6 debug build only (NOTES_k6): phase clocks of the k6 kernels
#include <cstdio>
#include <vector>
#include <algorithm>
__device__ unsigned long long g_site_ph[3 * 8];           // [kernel][phase]: cycles summed over the warps (lane 0)
__device__ unsigned long long g_site_n[3];                // warps counted
__device__ unsigned long long g_site_t[3 * 2 * 2048];     // [kernel][CTA]: start, end (globaltimer, ns)
#define PH_DECL                                                                          \
  long long ph_prev = clock64(), ph_acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};                   \
  unsigned long long ph_t0;                                                              \
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ph_t0));
#define PH(slot) do { if ((threadIdx.x & 31) == 0) { const long long n_ = clock64(); ph_acc[slot] += n_ - ph_prev; ph_prev = n_; } } while (0)
#define PH_USE(x) do { if ((x) == 0x12345679u) ph_acc[7] += 1; } while (0)  /* a branch that waits for the loaded value */
#define PH_END(kern, cta) do {                                                                                        \
    if ((threadIdx.x & 31) == 0) {                                                                                    \
      for (int i_ = 0; i_ < 8; ++i_) atomicAdd(&g_site_ph[(kern) * 8 + i_], (unsigned long long)ph_acc[i_]);          \
      atomicAdd(&g_site_n[kern], 1ull);                                                                               \
    }                                                                                                                 \
    if (threadIdx.x == 0 && (cta) < 2048) {                                                                           \
      unsigned long long t1_;                                                                                         \
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1_));                                                         \
      g_site_t[((kern) * 2048 + (cta)) * 2] = ph_t0;                                                                  \
      g_site_t[((kern) * 2048 + (cta)) * 2 + 1] = t1_;                                                                \
    }                                                                                                                 \
  } while (0)
#else
#define PH_DECL
#define PH(slot) do { } while (0)
#define PH_USE(x) do { } while (0)
#define PH_END(kern, cta) do { } while (0)
#endif

namespace {

constexpr int kC = 16, kHid = 64, kVa = 32;
constexpr float kEps = 1e-6f;
// tail of the table after R [16][heads]: Wv [32][16] | Wg [16][16] | bg [16] | Wo [16][32] | W1 [64][16] | Wr [64][16] | b1 [64] | W2 [16][64]
constexpr int kOffWv = 0, kOffWg = kOffWv + 32 * 16, kOffBg = kOffWg + 16 * 16, kOffWo = kOffBg + 16;
constexpr int kOffW1 = kOffWo + 16 * 32, kOffWr = kOffW1 + kHid * kC, kOffB1 = kOffWr + kHid * kC, kOffW2 = kOffB1 + kHid;
constexpr int kTail = kOffW2 + kC * kHid;
static_assert(kTail == bt6t::kTripletTabFixed, "table tail size");
// what kernel 3 reads, from Wo on
constexpr int kFfnTab = kTail - kOffWo;

// ------------------------------------------------------------------------------------------------------- kernel 1
// kPerm (k6): both e_hat copies are written in MMA-fragment order -- chunk t of a cell = channels 2t, 2t + 1, 8 + 2t, 9 + 2t
// (FP32: 16 bytes, FP16: 8 bytes) -- so a lane of kernels 2 / 3 gets its fragment words of a cell with ONE load (k5: two
// 4- or 8-byte loads per cell, each a separate L2 request for the same line). Values unchanged.
template <int HEADS, bool kPerm>
__global__ void __launch_bounds__(256, 4) site_readback(float* __restrict__ ehat32, __half* __restrict__ ehat16,
                                                     const __half* __restrict__ state, const __half* __restrict__ H,
                                                     const float* __restrict__ tab) {
  __shared__ float sR[kC * HEADS];
  PH_DECL
  for (int k = threadIdx.x; k < kC * HEADS; k += 256) sR[k] = __ldg(tab + k);
  __syncthreads();
  PH(0);
  const size_t idx = (size_t)blockIdx.x * 256 + threadIdx.x;  // b * 4096 + cell
  const size_t b = idx >> 12, cell = idx & 4095;
  float e[kC];
  {
    const uint4 a = __ldg(reinterpret_cast<const uint4*>(state + idx * kC));
    const uint4 c = __ldg(reinterpret_cast<const uint4*>(state + idx * kC) + 1);
    const __half2* ha = reinterpret_cast<const __half2*>(&a);
    const __half2* hc = reinterpret_cast<const __half2*>(&c);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const float2 fa = __half22float2(ha[k]), fc = __half22float2(hc[k]);
      e[2 * k] = fa.x; e[2 * k + 1] = fa.y; e[8 + 2 * k] = fc.x; e[9 + 2 * k] = fc.y;
    }
  }
  PH_USE(static_cast<unsigned>(e[0] + e[15]));
  PH(1);
  const __half* hp = H + b * HEADS * 4096 + cell;
#pragma unroll 8
  for (int h = 0; h < HEADS; ++h) {
    const float v = __half2float(__ldg(hp + (size_t)h * 4096));
#pragma unroll
    for (int c = 0; c < kC; ++c) e[c] = fmaf(sR[c * HEADS + h], v, e[c]);
  }
  PH_USE(static_cast<unsigned>(e[0]));
  PH(2);
  float4* d32 = reinterpret_cast<float4*>(ehat32 + idx * kC);
#pragma unroll
  for (int k = 0; k < 4; ++k)
    d32[k] = kPerm ? make_float4(e[2 * k], e[2 * k + 1], e[8 + 2 * k], e[9 + 2 * k])
                   : make_float4(e[4 * k], e[4 * k + 1], e[4 * k + 2], e[4 * k + 3]);
  uint4 w[2];
  __half2* hw = reinterpret_cast<__half2*>(w);
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const int pr = kPerm ? (k >> 1) + 4 * (k & 1) : k;  // the channel pair stored at slot k
    hw[k] = __floats2half2_rn(e[2 * pr], e[2 * pr + 1]);
  }
  uint4* d16 = reinterpret_cast<uint4*>(ehat16 + idx * kC);
  d16[0] = w[0];
  d16[1] = w[1];
  PH(3);
  PH_END(0, blockIdx.x);
}

// ------------------------------------------------------------------------------------------------------- kernel 2
#ifndef SITE_K2_INFLIGHT
#define SITE_K2_INFLIGHT 4
#endif
constexpr int kTThreads = 512;
constexpr int kTHeads = 4, kTDots = 4;
constexpr int kLPitch = 65;   // FP32 rows of 64 (+1 word: conflict-free transposed stores)
constexpr int kVPitch = 72;   // FP16 rows of 64 (+8 halves: conflict-free B-fragment loads)
constexpr int kAPitch = 72;
constexpr int kLBytes = 64 * kLPitch * 4;
constexpr int kVBytes = kTDots * 64 * kVPitch * 2;
constexpr int kABytes = 64 * kAPitch * 2;
constexpr int kTSmem = 2 * kLBytes + kVBytes + kABytes;  // 79,360 bytes: logit, sigmoid(door), 4 value maps, A

__device__ __forceinline__ uint32_t ld32(const __half* p) { return __ldg(reinterpret_cast<const unsigned int*>(p)); }
__device__ __forceinline__ uint32_t pack(float lo, float hi) {
  const __half2 v = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&v);
}
__device__ __forceinline__ float2 unpack(uint32_t x) { return __half22float2(*reinterpret_cast<const __half2*>(&x)); }
__device__ __forceinline__ void mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ float squares(uint32_t x) {
  const float2 v = unpack(x);
  return v.x * v.x + v.y * v.y;
}

// kCM (k6): va is stored in 8 planes [B][direction * 4 + head][4096 cells][4 dots] (one 16-byte store per 2 cells) for the
// tensor-core kernel 3 (site_out_ffn_mma); the values are the planar form's, bit for bit. (Fully cell-major [cell][32] made
// this kernel 30 % slower: every cell's 64-byte line is then written 8 bytes at a time by 8 CTAs.)
template <bool kCM>
__global__ void __launch_bounds__(kTThreads, 1) site_triplet(__half* __restrict__ va, const __half* __restrict__ ehat16,
                                                             const float* __restrict__ tail, int contraction) {
  extern __shared__ __align__(16) unsigned char smem[];
  float* sL = reinterpret_cast<float*>(smem);                            // logit as A's rows: [i][k], softmax over k
  float* sG = reinterpret_cast<float*>(smem + kLBytes);                  // sigmoid(door), [i][k]
  __half* sV = reinterpret_cast<__half*>(smem + 2 * kLBytes);            // values as the B operand: [d][j][k]
  __half* sA = reinterpret_cast<__half*>(smem + 2 * kLBytes + kVBytes);  // A = softmax x sigmoid, FP16 [i][k]
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int unit = blockIdx.x, head = unit & 3, direction = (unit >> 2) & 1, sample = unit >> 3;
  const bool swap = ((direction + contraction) & 1) != 0;  // path swaps the outward branch's V, ag the inward one's
#ifdef TRIPLET_CONTROL_SWAP_SOFTMAX
  const bool transposed = direction == 0;
#else
  const bool transposed = direction == 1;  // the outward logits are read transposed: its row softmax becomes a last-axis one
#endif
  const float* Wv = tail + kOffWv;
  const float* Wg = tail + kOffWg;
  const float* bg = tail + kOffBg;

  PH_DECL
  // the head's weights as the six maps' B fragments (map g: logit, door, v0..v3, two zero columns), hi + lo FP16
  const int gate_row = direction * 2 * kTHeads + head, door_row = gate_row + kTHeads;
  const int value_row = (direction * kTHeads + head) * kTDots;
  const float* wrow = g == 0 ? Wg + gate_row * kC : g == 1 ? Wg + door_row * kC : g < 6 ? Wv + (value_row + g - 2) * kC : nullptr;
  uint32_t bh[2], bl[2];
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const float w0 = wrow ? __ldg(wrow + 2 * t + 8 * r) : 0.f, w1 = wrow ? __ldg(wrow + 2 * t + 8 * r + 1) : 0.f;
    bh[r] = pack(w0, w1);
    const float2 h = unpack(bh[r]);
    bl[r] = pack(w0 - h.x, w1 - h.y);
  }
  const float logit_bias = __ldg(bg + gate_row), door_bias = __ldg(bg + door_row);

  // phase 1: 256 tiles of 16 cells (one grid row segment each); warp w takes tiles w, w + 16, ... four in flight
  const __half* src = ehat16 + (size_t)sample * 4096 * kC;
  constexpr int kU = kCM ? SITE_K2_INFLIGHT : 4;  // tiles in flight per warp (k6 probe -DSITE_K2_INFLIGHT=8 / 16: no gain, 73 / 77 us vs 73)
#pragma unroll 1
  for (int base = warp; base < 256; base += 16 * kU) {
    uint32_t a[kU][4];
#pragma unroll
    for (int u = 0; u < kU; ++u) {
      const int c0 = (base + 16 * u) * 16 + g;
      if (kCM) {  // fragment-ordered e_hat (site_readback kPerm): one 8-byte load per cell
        const uint2 v0 = __ldg(reinterpret_cast<const uint2*>(src + c0 * kC) + t);
        const uint2 v1 = __ldg(reinterpret_cast<const uint2*>(src + (c0 + 8) * kC) + t);
        a[u][0] = v0.x; a[u][2] = v0.y; a[u][1] = v1.x; a[u][3] = v1.y;
        if (u == kU - 1) { PH(0); PH_USE(a[0][0] ^ a[kU - 1][3] ^ bh[0] ^ bl[1]); PH(1); }
      } else {
        a[u][0] = ld32(src + c0 * kC + 2 * t);
        a[u][1] = ld32(src + (c0 + 8) * kC + 2 * t);
        a[u][2] = ld32(src + c0 * kC + 2 * t + 8);
        a[u][3] = ld32(src + (c0 + 8) * kC + 2 * t + 8);
      }
    }
#pragma unroll
    for (int u = 0; u < kU; ++u) {
      float s0 = squares(a[u][0]) + squares(a[u][2]), s1 = squares(a[u][1]) + squares(a[u][3]);
      s0 += __shfl_xor_sync(0xffffffffu, s0, 1);
      s1 += __shfl_xor_sync(0xffffffffu, s1, 1);
      s0 += __shfl_xor_sync(0xffffffffu, s0, 2);
      s1 += __shfl_xor_sync(0xffffffffu, s1, 2);
      const float inv0 = rsqrtf(fmaf(s0, 1.f / kC, kEps)), inv1 = rsqrtf(fmaf(s1, 1.f / kC, kEps));
      float c[4] = {0.f, 0.f, 0.f, 0.f};
      mma(c, a[u], bh[0], bh[1]);
      mma(c, a[u], bl[0], bl[1]);
      // c[0], c[1]: cell c0, maps 2t, 2t + 1; c[2], c[3]: cell c0 + 8 (W rms(x) = inv * (W x): the maps are linear)
      const int tile = base + 16 * u, row = tile >> 2;
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int col = ((tile & 3) << 4) + g + 8 * half;
        const float inv = half ? inv1 : inv0;
        const float x0 = c[2 * half] * inv, x1 = c[2 * half + 1] * inv;
        if (t == 0) {
          // A[i][k]: inward A = softmax_k(e_in[i, k]) at cell (i, k); outward A[i][k] = a_out[k, i] at cell (k, i)
          const int at = transposed ? col * kLPitch + row : row * kLPitch + col;
          sL[at] = x0 + logit_bias;
          sG[at] = 1.f / (1.f + __expf(-(x1 + door_bias)));
        } else if (t < 3) {
          // B = [j][k] of V_eff[k][j]: no swap V_eff[k][j] = v[k, j] (cell (row, col) = (k, j)) -> [col][row];
          // swap V_eff[k][j] = v[j, k] (cell (row, col) = (j, k)) -> [row][col]
          const int at = swap ? row * kVPitch + col : col * kVPitch + row;
          const int d = 2 * (t - 1);
          sV[d * 64 * kVPitch + at] = __float2half_rn(x0);
          sV[(d + 1) * 64 * kVPitch + at] = __float2half_rn(x1);
        }
      }
    }
    PH(2);
  }
  __syncthreads();
  PH(3);

#if defined(SITE_K2_RET) && SITE_K2_RET == 1  // k6 pricing switch (results wrong): stop after phase 1
  if (kCM) return;
#endif
  // phase 2a: A = softmax_k(L) x sigmoid(door) per row (FP32): warp w takes rows 4w .. 4w + 3, lane l keys 2l, 2l + 1
#pragma unroll
  for (int rr = 0; rr < 4; ++rr) {
    const int row = 4 * warp + rr;
    const float2 x = make_float2(sL[row * kLPitch + 2 * lane], sL[row * kLPitch + 2 * lane + 1]);
    float mx = fmaxf(x.x, x.y);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    const float p0 = __expf(x.x - mx), p1 = __expf(x.y - mx);
    float z = p0 + p1;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) z += __shfl_xor_sync(0xffffffffu, z, o);
    const float rz = 1.f / z;
    *reinterpret_cast<uint32_t*>(sA + row * kAPitch + 2 * lane) =
        pack(p0 * rz * sG[row * kLPitch + 2 * lane], p1 * rz * sG[row * kLPitch + 2 * lane + 1]);
  }
  PH(4);
  __syncthreads();
  PH(5);

#if defined(SITE_K2_RET) && SITE_K2_RET == 2  // stop after phase 2a
  if (kCM) return;
#endif
  if (kCM) {
    // phase 2b, cell-major: warp = (row tile mt, n-tile pair nq), all 4 dot channels: 2 n-tiles x 4 dots x 4 k-steps
    const int mt = warp & 3, nq = warp >> 2;
    const int r0 = 16 * mt + g, r1 = r0 + 8;
    uint32_t af[4][4];
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      af[ks][0] = *reinterpret_cast<const uint32_t*>(sA + r0 * kAPitch + 16 * ks + 2 * t);
      af[ks][1] = *reinterpret_cast<const uint32_t*>(sA + r1 * kAPitch + 16 * ks + 2 * t);
      af[ks][2] = *reinterpret_cast<const uint32_t*>(sA + r0 * kAPitch + 16 * ks + 2 * t + 8);
      af[ks][3] = *reinterpret_cast<const uint32_t*>(sA + r1 * kAPitch + 16 * ks + 2 * t + 8);
    }
    __half* outp = va + ((size_t)sample * 8 + direction * kTHeads + head) * 4096 * kTDots;
#pragma unroll
    for (int nn = 0; nn < 2; ++nn) {
      const int nt = 2 * nq + nn;
      float c[kTDots][4];
#pragma unroll
      for (int d = 0; d < kTDots; ++d) {
        c[d][0] = c[d][1] = c[d][2] = c[d][3] = 0.f;
        const __half* brow = sV + d * 64 * kVPitch + (8 * nt + g) * kVPitch + 2 * t;
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
          const uint32_t b0 = *reinterpret_cast<const uint32_t*>(brow + 16 * ks);
          const uint32_t b1 = *reinterpret_cast<const uint32_t*>(brow + 16 * ks + 8);
          mma(c[d], af[ks], b0, b1);
        }
      }
#pragma unroll
      for (int r = 0; r < 2; ++r)  // cells (row, 8 nt + 2 t) and (row, 8 nt + 2 t + 1), 4 dots each
        *reinterpret_cast<uint4*>(outp + ((r ? r1 : r0) * 64 + 8 * nt + 2 * t) * kTDots) =
            make_uint4(pack(c[0][2 * r], c[1][2 * r]), pack(c[2][2 * r], c[3][2 * r]),
                       pack(c[0][2 * r + 1], c[1][2 * r + 1]), pack(c[2][2 * r + 1], c[3][2 * r + 1]));
    }
    PH(6);
    PH_END(1, blockIdx.x);
    return;
  }
  // phase 2b: warp = (row tile mt, dot channel d): va[d] = A V[d], 4 k-steps x 8 n-tiles, stored FP16
  const int mt = warp & 3, d = warp >> 2;
  const int r0 = 16 * mt + g, r1 = r0 + 8;
  uint32_t af[4][4];
#pragma unroll
  for (int ks = 0; ks < 4; ++ks) {
    af[ks][0] = *reinterpret_cast<const uint32_t*>(sA + r0 * kAPitch + 16 * ks + 2 * t);
    af[ks][1] = *reinterpret_cast<const uint32_t*>(sA + r1 * kAPitch + 16 * ks + 2 * t);
    af[ks][2] = *reinterpret_cast<const uint32_t*>(sA + r0 * kAPitch + 16 * ks + 2 * t + 8);
    af[ks][3] = *reinterpret_cast<const uint32_t*>(sA + r1 * kAPitch + 16 * ks + 2 * t + 8);
  }
  const __half* vd = sV + d * 64 * kVPitch;
  __half* out = va + ((size_t)sample * kVa + direction * kC + head * kTDots + d) * 4096;
#pragma unroll
  for (int q = 0; q < 2; ++q) {
    uint32_t p[2][4];  // [row r0 / r1][n-tile 4q + n]
#pragma unroll
    for (int n = 0; n < 4; ++n) {
      float c[4] = {0.f, 0.f, 0.f, 0.f};
      const __half* brow = vd + (8 * (4 * q + n) + g) * kVPitch + 2 * t;
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        const uint32_t b0 = *reinterpret_cast<const uint32_t*>(brow + 16 * ks);
        const uint32_t b1 = *reinterpret_cast<const uint32_t*>(brow + 16 * ks + 8);
        mma(c, af[ks], b0, b1);
      }
      p[0][n] = pack(c[0], c[1]);
      p[1][n] = pack(c[2], c[3]);
    }
    // 4 x 4 transpose across the quad: lane t gets n-tile 4q + t's 8 keys -> one 16-byte store per row
    const bool hi1 = t & 2, hi0 = t & 1;
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      uint32_t x0 = __shfl_xor_sync(0xffffffffu, hi1 ? p[r][0] : p[r][2], 2);
      uint32_t x1 = __shfl_xor_sync(0xffffffffu, hi1 ? p[r][1] : p[r][3], 2);
      if (hi1) { p[r][0] = x0; p[r][1] = x1; } else { p[r][2] = x0; p[r][3] = x1; }
      x0 = __shfl_xor_sync(0xffffffffu, hi0 ? p[r][0] : p[r][1], 1);
      x1 = __shfl_xor_sync(0xffffffffu, hi0 ? p[r][2] : p[r][3], 1);
      if (hi0) { p[r][0] = x0; p[r][2] = x1; } else { p[r][1] = x0; p[r][3] = x1; }
      *reinterpret_cast<uint4*>(out + (r ? r1 : r0) * 64 + 32 * q + 8 * t) = make_uint4(p[r][0], p[r][1], p[r][2], p[r][3]);
    }
  }
}

// ------------------------------------------------------------------------------------------------------- kernel 3
constexpr int kRowsPerCta = 8, kFThreads = 64 * kRowsPerCta;
constexpr int kNTPitchR = kC * 65 + 1;  // [r][c][jj] with pitches (16 * 65 + 1, 65, 1): fewer bank conflicts on the strip stores

// e_hat2 = e_hat + Wo va of two cells of sample b (each weight read from shared memory feeds both cells)
__device__ __forceinline__ void e_hat2_of2(const float* __restrict__ ehat32, const __half* __restrict__ va, const float* sWoT,
                                           size_t b, int cell0, int cell1, float (&e0)[kC], float (&e1)[kC]) {
  const float4* p0 = reinterpret_cast<const float4*>(ehat32 + (b * 4096 + cell0) * kC);
  const float4* p1 = reinterpret_cast<const float4*>(ehat32 + (b * 4096 + cell1) * kC);
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const float4 v = __ldg(p0 + k), w = __ldg(p1 + k);
    e0[4 * k] = v.x; e0[4 * k + 1] = v.y; e0[4 * k + 2] = v.z; e0[4 * k + 3] = v.w;
    e1[4 * k] = w.x; e1[4 * k + 1] = w.y; e1[4 * k + 2] = w.z; e1[4 * k + 3] = w.w;
  }
  const __half* vp = va + b * kVa * 4096;
#pragma unroll 4
  for (int ch = 0; ch < kVa; ++ch) {
    const float v0 = __half2float(__ldg(vp + (size_t)ch * 4096 + cell0));
    const float v1 = __half2float(__ldg(vp + (size_t)ch * 4096 + cell1));
    const float4* w = reinterpret_cast<const float4*>(sWoT + ch * kC);  // Wo^T [32][16]
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const float4 x = w[q];
      e0[4 * q] = fmaf(x.x, v0, e0[4 * q]); e0[4 * q + 1] = fmaf(x.y, v0, e0[4 * q + 1]);
      e0[4 * q + 2] = fmaf(x.z, v0, e0[4 * q + 2]); e0[4 * q + 3] = fmaf(x.w, v0, e0[4 * q + 3]);
      e1[4 * q] = fmaf(x.x, v1, e1[4 * q]); e1[4 * q + 1] = fmaf(x.y, v1, e1[4 * q + 1]);
      e1[4 * q + 2] = fmaf(x.z, v1, e1[4 * q + 2]); e1[4 * q + 3] = fmaf(x.w, v1, e1[4 * q + 3]);
    }
  }
}

__device__ __forceinline__ float rms_scale(const float (&x)[kC]) {
  float s = 0.f;
#pragma unroll
  for (int c = 0; c < kC; ++c) s = fmaf(x[c], x[c], s);
  return rsqrtf(s * (1.f / kC) + kEps);
}

__device__ __forceinline__ void store_cell(__half* dst, const float (&e)[kC], float s) {
  uint4 w[2];
  __half2* hw = reinterpret_cast<__half2*>(w);
#pragma unroll
  for (int k = 0; k < 8; ++k) hw[k] = __floats2half2_rn(e[2 * k] * s, e[2 * k + 1] * s);
  uint4* d = reinterpret_cast<uint4*>(dst);
  d[0] = w[0];
  d[1] = w[1];
}

// shared-memory table of kernel 3, transposed so a cell's 16 output channels are 4 float4 loads:
//   WoT [32][16] | W1 [64][16] | Wr [64][16] | b1 [64] | W2T [64][16]
constexpr int kSWoT = 0, kSW1 = kSWoT + 512, kSWr = kSW1 + 1024, kSB1 = kSWr + 1024, kSW2T = kSB1 + 64;

// 256 threads, 2 cells per thread: own cells (i0 + r, j) and (i0 + r + 4, j); strip cells idx = t and t + 256
__global__ void __launch_bounds__(kFThreads / 2) site_out_ffn(__half* __restrict__ out, const float* __restrict__ ehat32,
                                                              const __half* __restrict__ va, const float* __restrict__ ffn_tab) {
  __shared__ __align__(16) float sW[kFfnTab];
  __shared__ float sNT[kRowsPerCta * kNTPitchR];             // n2 of cell (jj, i0 + r) at [r][c][jj]
  const size_t b = blockIdx.x;
  const int i0 = blockIdx.y * kRowsPerCta, t = threadIdx.x;
  constexpr int kT = kFThreads / 2;
  for (int k = t; k < kFfnTab; k += kT) {
    float x = __ldg(ffn_tab + k);
    int at = k;
    if (k < 512) at = kSWoT + (k & 31) * kC + (k >> 5);                          // Wo [c][ch] -> [ch][c]
    else if (k >= kOffW2 - kOffWo) { const int q = k - (kOffW2 - kOffWo); at = kSW2T + (q & 63) * kC + (q >> 6); }  // W2 [c][k] -> [k][c]
    sW[at] = x;
  }
  __syncthreads();
  {  // the mirrored column strip: cell (jj, i0 + r), idx = t, t + 256
    const int jj0 = t >> 3, r0 = t & 7, jj1 = (t + kT) >> 3;
    float e0[kC], e1[kC];
    e_hat2_of2(ehat32, va, sW + kSWoT, b, jj0 * 64 + i0 + r0, jj1 * 64 + i0 + r0, e0, e1);
    const float s0 = rms_scale(e0), s1 = rms_scale(e1);
#pragma unroll
    for (int c = 0; c < kC; ++c) {
      sNT[r0 * kNTPitchR + c * 65 + jj0] = e0[c] * s0;
      sNT[r0 * kNTPitchR + c * 65 + jj1] = e1[c] * s1;
    }
  }
  const int r = t >> 6, j = t & 63, ia = i0 + r, ib = ia + 4;
  float ea[kC], eb[kC];
  e_hat2_of2(ehat32, va, sW + kSWoT, b, ia * 64 + j, ib * 64 + j, ea, eb);
  float na[kC], nb[kC];
  {
    const float sa = rms_scale(ea), sb = rms_scale(eb);
#pragma unroll
    for (int c = 0; c < kC; ++c) { na[c] = ea[c] * sa; nb[c] = eb[c] * sb; }
  }
  __syncthreads();
  float ta[kC], tb[kC];
#pragma unroll
  for (int c = 0; c < kC; ++c) {
    ta[c] = sNT[r * kNTPitchR + c * 65 + j];
    tb[c] = sNT[(r + 4) * kNTPitchR + c * 65 + j];
  }
#pragma unroll 2
  for (int k = 0; k < kHid; ++k) {
    float ha = sW[kSB1 + k], hb = ha;
    const float4* w1 = reinterpret_cast<const float4*>(sW + kSW1 + k * kC);
    const float4* wr = reinterpret_cast<const float4*>(sW + kSWr + k * kC);
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const float4 x = w1[q], y = wr[q];
      ha = fmaf(x.x, na[4 * q], ha); ha = fmaf(x.y, na[4 * q + 1], ha); ha = fmaf(x.z, na[4 * q + 2], ha); ha = fmaf(x.w, na[4 * q + 3], ha);
      hb = fmaf(x.x, nb[4 * q], hb); hb = fmaf(x.y, nb[4 * q + 1], hb); hb = fmaf(x.z, nb[4 * q + 2], hb); hb = fmaf(x.w, nb[4 * q + 3], hb);
      ha = fmaf(y.x, ta[4 * q], ha); ha = fmaf(y.y, ta[4 * q + 1], ha); ha = fmaf(y.z, ta[4 * q + 2], ha); ha = fmaf(y.w, ta[4 * q + 3], ha);
      hb = fmaf(y.x, tb[4 * q], hb); hb = fmaf(y.y, tb[4 * q + 1], hb); hb = fmaf(y.z, tb[4 * q + 2], hb); hb = fmaf(y.w, tb[4 * q + 3], hb);
    }
    ha = fmaxf(ha, 0.f);
    hb = fmaxf(hb, 0.f);
    const float4* w2 = reinterpret_cast<const float4*>(sW + kSW2T + k * kC);
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const float4 x = w2[q];
      ea[4 * q] = fmaf(x.x, ha, ea[4 * q]); ea[4 * q + 1] = fmaf(x.y, ha, ea[4 * q + 1]);
      ea[4 * q + 2] = fmaf(x.z, ha, ea[4 * q + 2]); ea[4 * q + 3] = fmaf(x.w, ha, ea[4 * q + 3]);
      eb[4 * q] = fmaf(x.x, hb, eb[4 * q]); eb[4 * q + 1] = fmaf(x.y, hb, eb[4 * q + 1]);
      eb[4 * q + 2] = fmaf(x.z, hb, eb[4 * q + 2]); eb[4 * q + 3] = fmaf(x.w, hb, eb[4 * q + 3]);
    }
  }
  store_cell(out + ((b * 64 + ia) * 64 + j) * kC, ea, rms_scale(ea));
  store_cell(out + ((b * 64 + ib) * 64 + j) * kC, eb, rms_scale(eb));
}

// ------------------------------------------------------------------------------------------- kernel 3, tensor cores (k6)
// site_out_ffn_mma: the same math as site_out_ffn with the three maps as m16n8k16 MMAs (FP16 operands, FP32 accumulation;
// the residual e_hat stays FP32 as the accumulators' start value).
//   stage 1  e_hat2 = e_hat + Wo va          A = va [16 cells][32] from the 8-plane va, 2 n-tiles x 2 k-steps
//   stage 2  h = relu(W1 n2 + Wr n2T + b1)   A = n2 / n2T (the stage-1 accumulators scaled and packed), 8 n-tiles x 2
//   stage 3  e' = rms(e_hat2 + W2 h)         A = h (the stage-2 accumulators packed), 2 n-tiles x 4 k-steps
// Work split: the 64 x 64 cells are 4 x 4 blocks of 16 x 16; a CTA (512 threads) owns a block and its TRANSPOSE block
// ((I, J) + (J, I), 6 CTAs) or two diagonal blocks (2 CTAs): 32 tiles of 16 cells (a row segment), 2 per warp. Every cell's
// stage 1 runs once: its n2 goes to shared memory (FP16, the A-fragment words), and after one CTA barrier a tile reads its
// mirror cells' n2 (cells (j0 + g, i): a column of the transpose block) from there. So each input line is requested once
// per kernel (the first k6 form recomputed the mirror tile per warp: 3 x the L2 requests, which is what bounds this
// kernel: ~3.5 cycles per requested line and SM).
// Weights: B fragments prepared per CTA in shared memory, [fragment][lane], FP16 hi + lo; argument lo = mask of the stages
// that also run the lo fragments (1 | 2 | 4: the weights are then exact to 2^-22 and the only FP16 roundings are n2 and h).
constexpr int kMWo = 0, kMW1 = 8, kMWr = 24, kMW2 = 40, kMFrag = 56;
constexpr int kMOffW1 = 512, kMOffWr = kMOffW1 + kHid * kC, kMOffB1 = kMOffWr + kHid * kC, kMOffW2 = kMOffB1 + kHid;
static_assert(kMOffW2 == kOffW2 - kOffWo, "ffn table offsets");
constexpr int kNPitch = 16 * 8 + 4;  // words per block row of n2 (+4: the mirror reads of a warp hit 32 different banks)

__device__ __forceinline__ void mma1(float (&c)[4], const uint32_t (&a)[4], const uint32_t* sB, int frag, int lane) {
  mma(c, a, sB[frag * 32 + lane], sB[(frag + 1) * 32 + lane]);
}

__global__ void __launch_bounds__(512, 2) site_out_ffn_mma(__half* __restrict__ out, const float* __restrict__ ehat32,
                                                           const __half* __restrict__ va, const float* __restrict__ ffn_tab,
                                                           int lo) {
  __shared__ uint32_t sBh[kMFrag * 32];
  __shared__ uint32_t sBl[kMFrag * 32];
  __shared__ float sb1[kHid];
  __shared__ uint32_t sN[2 * 16 * kNPitch];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  PH_DECL
  for (int k = tid; k < kMFrag * 32; k += 512) {
    const int idx = k >> 5, l = k & 31, gg = l >> 2, tt = l & 3;
    int off;
    if (idx < kMW1) off = (8 * (idx >> 2) + gg) * kVa + 16 * ((idx >> 1) & 1) + 2 * tt + 8 * (idx & 1);
    else if (idx < kMWr) { const int q = idx - kMW1; off = kMOffW1 + (8 * (q >> 1) + gg) * kC + 2 * tt + 8 * (q & 1); }
    else if (idx < kMW2) { const int q = idx - kMWr; off = kMOffWr + (8 * (q >> 1) + gg) * kC + 2 * tt + 8 * (q & 1); }
    else { const int q = idx - kMW2; off = kMOffW2 + (8 * (q >> 3) + gg) * kHid + 16 * ((q >> 1) & 3) + 2 * tt + 8 * (q & 1); }
    const float2 w = __ldg(reinterpret_cast<const float2*>(ffn_tab + off));
    const uint32_t h = pack(w.x, w.y);
    sBh[k] = h;
    const float2 hf = unpack(h);
    sBl[k] = pack(w.x - hf.x, w.y - hf.y);
  }
  if (tid < kHid) sb1[tid] = __ldg(ffn_tab + kMOffB1 + tid);
  // the CTA's two blocks: y < 6: (I, J) and (J, I), I < J; y = 6, 7: the diagonal blocks (0, 0) + (1, 1), (2, 2) + (3, 3)
  const int y = blockIdx.y;
  const bool diag = y >= 6;
  const int bI = diag ? 2 * (y - 6) : y < 3 ? 0 : y < 5 ? 1 : 2;
  const int bJ = diag ? bI + 1 : y < 3 ? y + 1 : y < 5 ? y - 1 : 3;
  const size_t b = blockIdx.x;
  const float* eh = ehat32 + b * 4096 * kC;
  const __half* vb = va + b * 4096 * kVa;  // the sample's 8 planes
  __half* ob = out + b * 4096 * kC;
  __syncthreads();
  PH(0);
  float eo[2][2][4];  // [block][n-tile]: e_hat2 of the warp's two tiles (block row `warp`), kept across the barrier
#pragma unroll
  for (int blk = 0; blk < 2; ++blk) {
    const int bi = blk ? bJ : bI, bj = diag ? bi : (blk ? bI : bJ);
    const int c0 = (16 * bi + warp) * 64 + 16 * bj + g, c1 = c0 + 8;
    uint32_t a[2][4];
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      // channel 16 ks + 2 t (+ 8) = plane 4 ks + (t >> 1) (+ 2), dots 2 (t & 1), + 1
      const __half* pl = vb + (size_t)(4 * ks + (t >> 1)) * 4096 * 4 + 2 * (t & 1);
      a[ks][0] = ld32(pl + c0 * 4);
      a[ks][1] = ld32(pl + c1 * 4);
      a[ks][2] = ld32(pl + 2 * 4096 * 4 + c0 * 4);
      a[ks][3] = ld32(pl + 2 * 4096 * 4 + c1 * 4);
    }
    {  // fragment-ordered e_hat FP32 (site_readback kPerm): chunk t = channels 2t, 2t + 1, 8 + 2t, 9 + 2t
      const float4 x = __ldg(reinterpret_cast<const float4*>(eh + (size_t)c0 * kC) + t);
      const float4 z = __ldg(reinterpret_cast<const float4*>(eh + (size_t)c1 * kC) + t);
      eo[blk][0][0] = x.x; eo[blk][0][1] = x.y; eo[blk][0][2] = z.x; eo[blk][0][3] = z.y;
      eo[blk][1][0] = x.z; eo[blk][1][1] = x.w; eo[blk][1][2] = z.z; eo[blk][1][3] = z.w;
    }
    PH_USE(a[0][0] ^ a[0][1] ^ a[0][2] ^ a[0][3] ^ a[1][0] ^ a[1][1] ^ a[1][2] ^ a[1][3] ^ static_cast<unsigned>(eo[blk][0][0] + eo[blk][1][3]));
    PH(1);
#pragma unroll
    for (int nt = 0; nt < 2; ++nt)
#pragma unroll
      for (int ks = 0; ks < 2; ++ks) {
        mma1(eo[blk][nt], a[ks], sBh, kMWo + nt * 4 + ks * 2, lane);
        if (lo & 1) mma1(eo[blk][nt], a[ks], sBl, kMWo + nt * 4 + ks * 2, lane);
      }
    const float (&e)[2][4] = eo[blk];
    float s0 = e[0][0] * e[0][0] + e[0][1] * e[0][1] + e[1][0] * e[1][0] + e[1][1] * e[1][1];
    float s1 = e[0][2] * e[0][2] + e[0][3] * e[0][3] + e[1][2] * e[1][2] + e[1][3] * e[1][3];
    s0 += __shfl_xor_sync(0xffffffffu, s0, 1);
    s1 += __shfl_xor_sync(0xffffffffu, s1, 1);
    s0 += __shfl_xor_sync(0xffffffffu, s0, 2);
    s1 += __shfl_xor_sync(0xffffffffu, s1, 2);
    const float inv0 = rsqrtf(fmaf(s0, 1.f / kC, kEps)), inv1 = rsqrtf(fmaf(s1, 1.f / kC, kEps));
    uint32_t* nrow = sN + (blk * 16 + warp) * kNPitch;  // n2 of block row `warp`: [column][channel pair]
    nrow[g * 8 + t] = pack(e[0][0] * inv0, e[0][1] * inv0);
    nrow[(g + 8) * 8 + t] = pack(e[0][2] * inv1, e[0][3] * inv1);
    nrow[g * 8 + 4 + t] = pack(e[1][0] * inv0, e[1][1] * inv0);
    nrow[(g + 8) * 8 + 4 + t] = pack(e[1][2] * inv1, e[1][3] * inv1);
    PH(2);
  }
  __syncthreads();
  PH(3);
#pragma unroll 1
  for (int blk = 0; blk < 2; ++blk) {
    const int bi = blk ? bJ : bI, bj = diag ? bi : (blk ? bI : bJ);
    const int c0 = (16 * bi + warp) * 64 + 16 * bj + g, c1 = c0 + 8;
    uint32_t no[4], nm[4];
    {
      const uint32_t* nrow = sN + (blk * 16 + warp) * kNPitch;
      no[0] = nrow[g * 8 + t];
      no[1] = nrow[(g + 8) * 8 + t];
      no[2] = nrow[g * 8 + 4 + t];
      no[3] = nrow[(g + 8) * 8 + 4 + t];
      // the mirror cells (16 bj + g (+ 8), 16 bi + warp): rows g, g + 8 of the transpose block, column `warp`
#ifdef SITE_CONTROL_NO_MIRROR  // must-fail control: the reverse-edge map reads the tile's OWN n2 instead of the mirror cells
      const uint32_t* mcol = sN + (blk * 16 + warp) * kNPitch + g * 8 + t;
      (void)diag;
#else
      const uint32_t* mcol = sN + ((diag ? blk : 1 - blk) * 16 + g) * kNPitch + warp * 8 + t;
#endif
#ifdef SITE_CONTROL_NO_MIRROR
      nm[0] = mcol[0]; nm[1] = mcol[64]; nm[2] = mcol[4]; nm[3] = mcol[68];
#else
      nm[0] = mcol[0];
      nm[1] = mcol[8 * kNPitch];
      nm[2] = mcol[4];
      nm[3] = mcol[8 * kNPitch + 4];
#endif
    }
    uint32_t ha[4][4];
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      const float2 bb = *reinterpret_cast<const float2*>(sb1 + 8 * nt + 2 * t);
      float c[4] = {bb.x, bb.y, bb.x, bb.y};
      mma1(c, no, sBh, kMW1 + 2 * nt, lane);
      mma1(c, nm, sBh, kMWr + 2 * nt, lane);
      if (lo & 2) {
        mma1(c, no, sBl, kMW1 + 2 * nt, lane);
        mma1(c, nm, sBl, kMWr + 2 * nt, lane);
      }
      ha[nt >> 1][2 * (nt & 1)] = pack(fmaxf(c[0], 0.f), fmaxf(c[1], 0.f));
      ha[nt >> 1][2 * (nt & 1) + 1] = pack(fmaxf(c[2], 0.f), fmaxf(c[3], 0.f));
    }
    PH_USE(ha[0][0] ^ ha[3][3]);
    PH(4);
    float e[2][4];
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      e[nt][0] = eo[blk][nt][0]; e[nt][1] = eo[blk][nt][1]; e[nt][2] = eo[blk][nt][2]; e[nt][3] = eo[blk][nt][3];
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        mma1(e[nt], ha[ks], sBh, kMW2 + nt * 8 + ks * 2, lane);
        if (lo & 4) mma1(e[nt], ha[ks], sBl, kMW2 + nt * 8 + ks * 2, lane);
      }
    }
    float s0 = e[0][0] * e[0][0] + e[0][1] * e[0][1] + e[1][0] * e[1][0] + e[1][1] * e[1][1];
    float s1 = e[0][2] * e[0][2] + e[0][3] * e[0][3] + e[1][2] * e[1][2] + e[1][3] * e[1][3];
    s0 += __shfl_xor_sync(0xffffffffu, s0, 1);
    s1 += __shfl_xor_sync(0xffffffffu, s1, 1);
    s0 += __shfl_xor_sync(0xffffffffu, s0, 2);
    s1 += __shfl_xor_sync(0xffffffffu, s1, 2);
    const float inv0 = rsqrtf(s0 * (1.f / kC) + kEps), inv1 = rsqrtf(s1 * (1.f / kC) + kEps);
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      *reinterpret_cast<uint32_t*>(ob + (size_t)c0 * kC + 8 * nt + 2 * t) = pack(e[nt][0] * inv0, e[nt][1] * inv0);
      *reinterpret_cast<uint32_t*>(ob + (size_t)c1 * kC + 8 * nt + 2 * t) = pack(e[nt][2] * inv1, e[nt][3] * inv1);
    }
    PH(5);
  }
  PH_END(2, blockIdx.x * 8 + blockIdx.y);
}

}  // namespace

namespace bt6t {

size_t triplet_site_workspace(int batch) { return (size_t)batch * 4096 * (kC * 4 + kC * 2 + kVa * 2) + 256; }

int prepare_triplet_site() {
  const int rc = static_cast<int>(cudaFuncSetAttribute(site_triplet<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, kTSmem));
  if (rc != 0) return rc;
  return static_cast<int>(cudaFuncSetAttribute(site_triplet<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, kTSmem));
}

int launch_triplet_site(__half* out, const __half* state, const __half* H, const float* tab, int heads, int contraction,
                        void* workspace, int batch, cudaStream_t stream) {
  if (batch <= 0) return 0;
  if (heads != 24 && heads != 32) return static_cast<int>(cudaErrorInvalidValue);
  char* ws = static_cast<char*>(workspace);
  float* ehat32 = reinterpret_cast<float*>(ws);
  __half* ehat16 = reinterpret_cast<__half*>(ws + (size_t)batch * 4096 * kC * 4);
  __half* va = reinterpret_cast<__half*>(ws + (size_t)batch * 4096 * kC * 6);
  const float* tail = tab + kC * heads;
  const int cells = batch * 4096;
  // k6: kernel 3 on the tensor cores (va in 8 planes). Env BT6_SITE_K5=1 = k5's FP32 kernel 3 (va planar);
  // BT6_SITE_LO = mask of the stages that use the FP16 lo halves of the weights (1 Wo, 2 W1 / Wr, 4 W2; default 7).
  static const bool k5_path = [] { const char* e = std::getenv("BT6_SITE_K5"); return e && e[0] == '1'; }();
  static const int lo = [] { const char* e = std::getenv("BT6_SITE_LO"); return e ? std::atoi(e) & 7 : 7; }();
  if (k5_path) {
    if (heads == 32) site_readback<32, false><<<cells / 256, 256, 0, stream>>>(ehat32, ehat16, state, H, tab);
    else site_readback<24, false><<<cells / 256, 256, 0, stream>>>(ehat32, ehat16, state, H, tab);
    site_triplet<false><<<batch * 8, kTThreads, kTSmem, stream>>>(va, ehat16, tail, contraction);
    site_out_ffn<<<dim3(batch, 64 / kRowsPerCta), kFThreads / 2, 0, stream>>>(out, ehat32, va, tail + kOffWo);
  } else {
    if (heads == 32) site_readback<32, true><<<cells / 256, 256, 0, stream>>>(ehat32, ehat16, state, H, tab);
    else site_readback<24, true><<<cells / 256, 256, 0, stream>>>(ehat32, ehat16, state, H, tab);
    site_triplet<true><<<batch * 8, kTThreads, kTSmem, stream>>>(va, ehat16, tail, contraction);
    site_out_ffn_mma<<<dim3(batch, 8), 512, 0, stream>>>(out, ehat32, va, tail + kOffWo, lo);
  }
#ifdef SITE_PHASE
  if (!k5_path) {
    static const char* names[3][8] = {{"tab+sync", "state-wait", "H+fma", "stores", "", "", "", ""},
                                      {"issue", "load-wait", "maps+stores", "sync1", "softmax", "sync2", "AV+stores", ""},
                                      {"prep+sync", "load-wait", "stage1+n2", "sync", "stage2", "stage3+stores", "", ""}};
    static const char* kn[3] = {"readback", "triplet", "out_ffn_mma"};
    const int ncta[3] = {cells / 256, batch * 8, batch * 8};
    static std::vector<unsigned long long> t(3 * 2 * 2048);
    unsigned long long ph[24], n[3];
    cudaStreamSynchronize(stream);
    cudaMemcpyFromSymbol(ph, g_site_ph, sizeof(ph));
    cudaMemcpyFromSymbol(n, g_site_n, sizeof(n));
    cudaMemcpyFromSymbol(t.data(), g_site_t, sizeof(unsigned long long) * t.size());
    for (int k = 0; k < 3; ++k) {
      const int m = ncta[k] < 2048 ? ncta[k] : 2048;
      unsigned long long t0 = ~0ull, t1 = 0;
      std::vector<double> d;
      for (int c = 0; c < m; ++c) {
        const unsigned long long a = t[(k * 2048 + c) * 2], b = t[(k * 2048 + c) * 2 + 1];
        if (!a) continue;
        t0 = std::min(t0, a); t1 = std::max(t1, b);
        d.push_back((b - a) * 1e-3);
      }
      std::sort(d.begin(), d.end());
      double tot = 0;
      for (int i = 0; i < 8; ++i) tot += ph[k * 8 + i];
      fprintf(stderr, "SITEPH B=%d %s ctas=%d span %.1f us; CTA us p10 %.1f p50 %.1f p90 %.1f; cyc/warp %.0f :", batch, kn[k], ncta[k],
              (t1 - t0) * 1e-3, d.empty() ? 0. : d[d.size() / 10], d.empty() ? 0. : d[d.size() / 2], d.empty() ? 0. : d[d.size() * 9 / 10],
              n[k] ? tot / n[k] : 0.);
      for (int i = 0; i < 7; ++i)
        if (names[k][i][0]) fprintf(stderr, " %s %.0f", names[k][i], n[k] ? (double)ph[k * 8 + i] / n[k] : 0.);
      fprintf(stderr, "\n");
    }
    static const std::vector<unsigned long long> z(3 * 2 * 2048, 0ull);
    cudaMemcpyToSymbol(g_site_ph, z.data(), sizeof(ph));
    cudaMemcpyToSymbol(g_site_n, z.data(), sizeof(n));
    cudaMemcpyToSymbol(g_site_t, z.data(), sizeof(unsigned long long) * z.size());
  }
#endif
  return static_cast<int>(cudaGetLastError());
}

}  // namespace bt6t

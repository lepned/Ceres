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
#include <cuda_fp16.h>
#include <cstdint>
#include "bt6_triplet_site.h"

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
template <int HEADS>
__global__ void __launch_bounds__(256) site_readback(float* __restrict__ ehat32, __half* __restrict__ ehat16,
                                                     const __half* __restrict__ state, const __half* __restrict__ H,
                                                     const float* __restrict__ tab) {
  __shared__ float sR[kC * HEADS];
  for (int k = threadIdx.x; k < kC * HEADS; k += 256) sR[k] = __ldg(tab + k);
  __syncthreads();
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
  const __half* hp = H + b * HEADS * 4096 + cell;
#pragma unroll 8
  for (int h = 0; h < HEADS; ++h) {
    const float v = __half2float(__ldg(hp + (size_t)h * 4096));
#pragma unroll
    for (int c = 0; c < kC; ++c) e[c] = fmaf(sR[c * HEADS + h], v, e[c]);
  }
  float4* d32 = reinterpret_cast<float4*>(ehat32 + idx * kC);
#pragma unroll
  for (int k = 0; k < 4; ++k) d32[k] = make_float4(e[4 * k], e[4 * k + 1], e[4 * k + 2], e[4 * k + 3]);
  uint4 w[2];
  __half2* hw = reinterpret_cast<__half2*>(w);
#pragma unroll
  for (int k = 0; k < 8; ++k) hw[k] = __floats2half2_rn(e[2 * k], e[2 * k + 1]);
  uint4* d16 = reinterpret_cast<uint4*>(ehat16 + idx * kC);
  d16[0] = w[0];
  d16[1] = w[1];
}

// ------------------------------------------------------------------------------------------------------- kernel 2
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
#pragma unroll 1
  for (int base = warp; base < 256; base += 64) {
    uint32_t a[4][4];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const int c0 = (base + 16 * u) * 16 + g;
      a[u][0] = ld32(src + c0 * kC + 2 * t);
      a[u][1] = ld32(src + (c0 + 8) * kC + 2 * t);
      a[u][2] = ld32(src + c0 * kC + 2 * t + 8);
      a[u][3] = ld32(src + (c0 + 8) * kC + 2 * t + 8);
    }
#pragma unroll
    for (int u = 0; u < 4; ++u) {
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
  }
  __syncthreads();

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
  __syncthreads();

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

}  // namespace

namespace bt6t {

size_t triplet_site_workspace(int batch) { return (size_t)batch * 4096 * (kC * 4 + kC * 2 + kVa * 2) + 256; }

int prepare_triplet_site() {
  return static_cast<int>(cudaFuncSetAttribute(site_triplet, cudaFuncAttributeMaxDynamicSharedMemorySize, kTSmem));
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
  if (heads == 32) site_readback<32><<<cells / 256, 256, 0, stream>>>(ehat32, ehat16, state, H, tab);
  else site_readback<24><<<cells / 256, 256, 0, stream>>>(ehat32, ehat16, state, H, tab);
  site_triplet<<<batch * 8, kTThreads, kTSmem, stream>>>(va, ehat16, tail, contraction);
  site_out_ffn<<<dim3(batch, 64 / kRowsPerCta), kFThreads / 2, 0, stream>>>(out, ehat32, va, tail + kOffWo);
  return static_cast<int>(cudaGetLastError());
}

}  // namespace bt6t

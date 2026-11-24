// BT6Prologue (round 2): the whole BT6 prologue of the flat 768x16 trunk in ONE kernel, straight from the record.
//
// Replaces the exported graph from `squares` to (bt6p_edges, S = rsqrt_1, e0 = mul_615): CeresTrain
// bt6_hybrid.pieces_from_squares + bt6_relations.BT6Relations (34 relation channels E) + bt6_net.BT6Prologue
// (pair normaliser S and edge-state seed e0) + round 1's INT32 bit packing of E.
//
//   pieces    p[c][s] = squares[s, 1 + c] > 0.5          (c = 0..11: ours P N B R Q K, theirs P N B R Q K)
//   previous  q[c][s] = squares[s, 14 + c] > 0.5, and q = p when the previous position holds no piece at all
//   E         the 17 base channels as rows (one 64-bit board of targets per source square; the bit logic is lc0ex
//             prologue_egt's rows kernel, ported), channels 17..33 = the transposes
//   Z[d]      = sum_c E[c, i, j] D[c, d] + T[relidx(i, j), d],        S = 1 / sqrt(mean_d Z^2 + eps)
//   e0[d]     = st[d] / sqrt(mean_d st^2 + eps),  st[d] = sum_c E[c, i, j] Pin[c, d] + Toff[relidx(i, j), d]
//   relidx    = 15 (row_i - row_j + 7) + (col_i - col_j + 7)
// FP32 arithmetic from the FP32 trainer parameters (the export rounds D / T / Pin / Toff to FP16 first).
//
// Outputs: edges INT32 [B, 64, 64, 2] (channels 0-16 | 17-33 as bit fields, round 1's layout), S FP16 [B, 64, 64],
// e0 FP16 [B, 64, 64, 16] cell-major (channel innermost: the BT6TermsAttention state input and edge site 1's
// channels-last input, without the transposes).
// Table blob tab32 FP32: D [34][16] | Pin [34][16] | T [225][16] | Toff [225][16].
#include <cuda_fp16.h>
#include <cstdint>
#include "bt6_kernels.h"

namespace {

using u64 = unsigned long long;
constexpr int kRecord = 137;
constexpr u64 kNotA = 0xFEFEFEFEFEFEFEFEull, kNotH = 0x7F7F7F7F7F7F7F7Full;
constexpr u64 kNotAB = 0xFCFCFCFCFCFCFCFCull, kNotGH = 0x3F3F3F3F3F3F3F3Full;

// one (row, column) step of every set bit; bits leaving the board vanish (square = 8 row + column)
template <int R, int C>
__device__ __forceinline__ u64 step(u64 b) {
  constexpr int sh = 8 * R + C;
  u64 m = sh >= 0 ? (b << (sh >= 0 ? sh : 0)) : (b >> (sh < 0 ? -sh : 0));
  if (C == 1) m &= kNotA;
  if (C == 2) m &= kNotAB;
  if (C == -1) m &= kNotH;
  if (C == -2) m &= kNotGH;
  return m;
}
__device__ __forceinline__ u64 king_moves(u64 b) {
  return step<1, 0>(b) | step<-1, 0>(b) | step<0, 1>(b) | step<0, -1>(b) | step<1, 1>(b) | step<1, -1>(b) |
         step<-1, 1>(b) | step<-1, -1>(b);
}
__device__ __forceinline__ u64 knight_moves(u64 b) {
  return step<1, 2>(b) | step<2, 1>(b) | step<-1, 2>(b) | step<-2, 1>(b) | step<1, -2>(b) | step<2, -1>(b) |
         step<-1, -2>(b) | step<-2, -1>(b);
}
__device__ __forceinline__ u64 ring2_moves(u64 b) {
  return step<2, -2>(b) | step<2, -1>(b) | step<2, 0>(b) | step<2, 1>(b) | step<2, 2>(b) | step<-2, -2>(b) |
         step<-2, -1>(b) | step<-2, 0>(b) | step<-2, 1>(b) | step<-2, 2>(b) | step<-1, -2>(b) | step<0, -2>(b) |
         step<1, -2>(b) | step<-1, 2>(b) | step<0, 2>(b) | step<1, 2>(b);
}

// Slider directions (bt6_relations.SLIDER_DIRECTIONS): NE, NW, SE, SW (diagonal), N, S, E, W; opposite = 3 2 1 0 5 4 7 6
__constant__ int kDirRow[8] = {1, 1, -1, -1, 1, -1, 0, 0};
__constant__ int kDirCol[8] = {1, -1, 1, -1, 0, 0, 1, -1};

struct Scan {
  u64 reach_d, reach_o, xray_d, xray_o;
  u64 first[8];
};

// walk the 8 rays of square s: reach (up to and including the first blocker), the first blocker, the x-ray (strictly
// beyond it up to and including the second blocker)
__device__ __forceinline__ void scan(int s, u64 occ, Scan& r, bool want_xray) {
  r.reach_d = r.reach_o = r.xray_d = r.xray_o = 0;
  const int row = s >> 3, col = s & 7;
#pragma unroll
  for (int d = 0; d < 8; ++d) {
    u64 reach = 0, xray = 0, first = 0;
    int state = 0;
    int rr = row, cc = col;
#pragma unroll
    for (int k = 0; k < 7; ++k) {
      rr += kDirRow[d];
      cc += kDirCol[d];
      if (rr < 0 || rr > 7 || cc < 0 || cc > 7 || state == 2) break;
      const u64 bit = 1ull << (8 * rr + cc);
      const bool hit = (occ & bit) != 0;
      if (state == 0) {
        reach |= bit;
        if (hit) { first = bit; state = 1; if (!want_xray) state = 2; }
      } else {
        xray |= bit;
        if (hit) state = 2;
      }
    }
    r.first[d] = first;
    if (d < 4) { r.reach_d |= reach; r.xray_d |= xray; } else { r.reach_o |= reach; r.xray_o |= xray; }
  }
}

// attack row of one side at source square s (bit `src`): pawns toward +row (ours) / -row (theirs), knight, king,
// sliders (bishop | queen diagonal, rook | queen orthogonal)
__device__ __forceinline__ u64 attack_row(u64 src, const u64* side, const Scan& sc, bool ours) {
  u64 a = 0;
  if (side[0] & src) a |= ours ? (step<1, -1>(src) | step<1, 1>(src)) : (step<-1, -1>(src) | step<-1, 1>(src));
  if (side[1] & src) a |= knight_moves(src);
  if (side[5] & src) a |= king_moves(src);
  if ((side[2] | side[4]) & src) a |= sc.reach_d;
  if ((side[3] | side[4]) & src) a |= sc.reach_o;
  return a;
}

__device__ __forceinline__ u64 pin_row(bool here, const Scan& sc, u64 diag, u64 orth) {
  if (!here) return 0;
  constexpr int opp[8] = {3, 2, 1, 0, 5, 4, 7, 6};
  u64 r = 0;
#pragma unroll
  for (int d = 0; d < 8; ++d) {
    const u64 sliders = d < 4 ? diag : orth;
    if (sc.first[opp[d]] & sliders) r |= sc.first[d];
  }
  return r;
}

constexpr int kThreads = 256;
constexpr int kTabD = 0, kTabPin = 34 * 16, kTabT = 2 * 34 * 16, kTabToff = kTabT + 225 * 16;
constexpr int kTab = kTabToff + 225 * 16;

__global__ void __launch_bounds__(kThreads) bt6_prologue(int* __restrict__ edges, __half* __restrict__ norm,
                                                         __half* __restrict__ state, const __half* __restrict__ squares,
                                                         const float* __restrict__ tab, int rows_per_cta, float eps) {
  __shared__ u64 sRows[17][64];
  __shared__ unsigned int sPlane[24][2];   // 12 current + 12 previous piece boards, 32-bit halves
  __shared__ float sDP[34][32];            // per channel: D[c][0..15] | Pin[c][0..15]
  __shared__ float sT[225][32];            // per relative bin: T[r][0..15] | Toff[r][0..15]
  __shared__ int sAnyPrev;
  const int b = blockIdx.x, part = blockIdx.y, tid = threadIdx.x;
  const __half* rec = squares + (size_t)b * 64 * kRecord;

  for (int k = tid; k < 34 * 32; k += kThreads) {
    const int c = k >> 5, d = k & 31;
    sDP[c][d] = __ldg(tab + (d < 16 ? kTabD + c * 16 + d : kTabPin + c * 16 + d - 16));
  }
  for (int k = tid; k < 225 * 32; k += kThreads) {
    const int r = k >> 5, d = k & 31;
    sT[r][d] = __ldg(tab + (d < 16 ? kTabT + r * 16 + d : kTabToff + r * 16 + d - 16));
  }
  if (tid == 0) sAnyPrev = 0;
  __syncthreads();
  // piece boards by ballot: threads 0..63 = squares
  if (tid < 64) {
    const int s = tid, lane = s & 31, half = s >> 5;
    bool anyp = false;
#pragma unroll
    for (int c = 0; c < 12; ++c) {
      const bool cur = __half2float(rec[s * kRecord + 1 + c]) > 0.5f;
      const bool prv = __half2float(rec[s * kRecord + 14 + c]) > 0.5f;
      anyp |= prv;
      const unsigned int bc = __ballot_sync(0xffffffffu, cur), bp = __ballot_sync(0xffffffffu, prv);
      if (lane == 0) { sPlane[c][half] = bc; sPlane[12 + c][half] = bp; }
    }
    if (anyp) atomicOr(&sAnyPrev, 1);
  }
  __syncthreads();
  if (tid < 64) {
    const int s = tid;
    const u64 src = 1ull << s;
    u64 us[6], them[6], pus[6], pthem[6];
    const bool missing = sAnyPrev == 0;
#pragma unroll
    for (int c = 0; c < 6; ++c) {
      us[c] = (u64)sPlane[c][0] | ((u64)sPlane[c][1] << 32);
      them[c] = (u64)sPlane[6 + c][0] | ((u64)sPlane[6 + c][1] << 32);
      pus[c] = missing ? us[c] : ((u64)sPlane[12 + c][0] | ((u64)sPlane[12 + c][1] << 32));
      pthem[c] = missing ? them[c] : ((u64)sPlane[18 + c][0] | ((u64)sPlane[18 + c][1] << 32));
    }
    const u64 occ_us = us[0] | us[1] | us[2] | us[3] | us[4] | us[5];
    const u64 occ_them = them[0] | them[1] | them[2] | them[3] | them[4] | them[5];
    const u64 occ = occ_us | occ_them;
    const bool here = (occ & src) != 0;
    Scan sc;
    scan(s, occ, sc, true);
    const u64 att_us = attack_row(src, us, sc, true), att_them = attack_row(src, them, sc, false);
    const u64 diag_us = us[2] | us[4], orth_us = us[3] | us[4], diag_them = them[2] | them[4], orth_them = them[3] | them[4];
    u64 xr_us = 0, xr_them = 0;
    if (diag_us & src) xr_us |= sc.xray_d;
    if (orth_us & src) xr_us |= sc.xray_o;
    if (diag_them & src) xr_them |= sc.xray_d;
    if (orth_them & src) xr_them |= sc.xray_o;
    sRows[0][s] = att_us;
    sRows[1][s] = att_them;
    sRows[2][s] = att_us & occ_us;
    sRows[3][s] = att_them & occ_them;
    sRows[4][s] = xr_us;
    sRows[5][s] = xr_them;
    sRows[6][s] = pin_row(here, sc, diag_us, orth_us);
    sRows[7][s] = pin_row(here, sc, diag_them, orth_them);
    sRows[8][s] = king_moves(att_us) & them[5];
    sRows[9][s] = king_moves(att_them) & us[5];
    sRows[10][s] = ring2_moves(att_us) & them[5];
    sRows[11][s] = ring2_moves(att_them) & us[5];
    sRows[12][s] = here ? (king_moves(src) & (us[5] | them[5])) : 0ull;
    // previous frame: channels 0/1 with its own occupancy
    const u64 pocc = pus[0] | pus[1] | pus[2] | pus[3] | pus[4] | pus[5] | pthem[0] | pthem[1] | pthem[2] | pthem[3] |
                     pthem[4] | pthem[5];
    Scan ps;
    scan(s, pocc, ps, false);
    const u64 patt_us = attack_row(src, pus, ps, true), patt_them = attack_row(src, pthem, ps, false);
    sRows[13][s] = att_us & ~patt_us;
    sRows[14][s] = att_them & ~patt_them;
    sRows[15][s] = patt_us & ~att_us;
    sRows[16][s] = patt_them & ~att_them;
#ifdef PRO_CONTROL_DROP_CH  // the fidelity gate's CONTROL (must fail): one relation channel zeroed
    sRows[PRO_CONTROL_DROP_CH][s] = 0ull;
#endif
  }
  __syncthreads();

  // cells of this CTA's rows: i in [part * rows_per_cta, +rows_per_cta), j = 0..63
  const int cells = rows_per_cta * 64, i0 = part * rows_per_cta;
  for (int k = tid; k < cells; k += kThreads) {
    const int i = i0 + (k >> 6), j = k & 63;
    unsigned int lo = 0, hi = 0;
#pragma unroll
    for (int c = 0; c < 17; ++c) {
      lo |= static_cast<unsigned int>((sRows[c][i] >> j) & 1ull) << c;
      hi |= static_cast<unsigned int>((sRows[c][j] >> i) & 1ull) << c;
    }
    const int rel = 15 * ((i >> 3) - (j >> 3) + 7) + ((i & 7) - (j & 7) + 7);
    float z[32];
#pragma unroll
    for (int d = 0; d < 32; d += 4) {
      const float4 v = *reinterpret_cast<const float4*>(&sT[rel][d]);
      z[d] = v.x; z[d + 1] = v.y; z[d + 2] = v.z; z[d + 3] = v.w;
    }
    u64 bits = (u64)lo | ((u64)hi << 17);
    while (bits) {
      const int c = __ffsll(static_cast<long long>(bits)) - 1;
      bits &= bits - 1;
#pragma unroll
      for (int d = 0; d < 32; d += 4) {
        const float4 v = *reinterpret_cast<const float4*>(&sDP[c][d]);
        z[d] += v.x; z[d + 1] += v.y; z[d + 2] += v.z; z[d + 3] += v.w;
      }
    }
    float sz = 0.f, ss = 0.f;
#pragma unroll
    for (int d = 0; d < 16; ++d) { sz += z[d] * z[d]; ss += z[16 + d] * z[16 + d]; }
    const float S = rsqrtf(sz * (1.f / 16.f) + eps), r = rsqrtf(ss * (1.f / 16.f) + eps);
    const size_t cell = ((size_t)b * 64 + i) * 64 + j;
    *reinterpret_cast<int2*>(edges + 2 * cell) = make_int2(static_cast<int>(lo), static_cast<int>(hi));
    norm[cell] = __float2half_rn(S);
    uint4 w[2];
    __half2* h = reinterpret_cast<__half2*>(w);
#pragma unroll
    for (int d = 0; d < 8; ++d) h[d] = __floats2half2_rn(z[16 + 2 * d] * r, z[17 + 2 * d] * r);
    uint4* dst = reinterpret_cast<uint4*>(state + cell * 16);
    dst[0] = w[0];
    dst[1] = w[1];
  }
}

}  // namespace

namespace bt6 {

int launch_prologue(int* edges, __half* norm, __half* state, const __half* squares, const float* tab, int batch,
                    int sms, cudaStream_t stream) {
  if (batch <= 0) return 0;
  int split = 1;  // CTAs per position (each rebuilds the rows, then writes 64 / split source rows of cells)
  while (split < 8 && batch * split < 2 * sms) split *= 2;
  dim3 grid(batch, split);
  bt6_prologue<<<grid, kThreads, 0, stream>>>(edges, norm, state, squares, tab, 64 / split, 1e-6f);
  return static_cast<int>(cudaGetLastError());
}

}  // namespace bt6
static_assert(kTab == bt6::kPrologueTab32, "prologue table size");

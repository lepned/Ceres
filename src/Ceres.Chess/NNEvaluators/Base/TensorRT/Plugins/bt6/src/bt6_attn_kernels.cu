// Term-layer attention of the "terms" front end on the lab's v9 head pipelines (`attention_terms_v9.py`).
// This file is `bt4/kernels/egt_v9.cu` (copied, the lab file is not modified) with ONE addition: BIAS, the block's
// shared-smolgen logit bias, FP16 [batch * heads, 64, 64], added to the logits where the trainer adds it -- with the
// edge read, before the door:
//     H = (qk_scale q.k + E terms + S cbias + read . e + bias) * (1 + door . e)
// The terms nets' own layers have no output gate: built with OGATE=0 (packed [q | k | v], the code path the lab file
// already has). Everything else -- the work list, the shared-memory layout, the E prefix form, the arithmetic class
// (FP16 E tables, HFMA2 state math, FP32 softmax / prefix sums) -- is the lab kernel's, described below.
//
// EGT2 attention, v9 (backend lane 09-28/29; `attention_egt_v9.py`, the H3 shape): per-warp head pipelines over a half-position.
// A CTA owns a half-position (32 query rows): the FP16 cell-major edge state is loaded once into shared memory and read by every
// head. It runs 4 heads at a time -- 16 warps = 4 heads x 2 row tiles x 2 key halves -- and each warp runs its head's pipeline in
// registers: q.k, the logit assembly (E, S * cbias, HFMA2 read / door / gate), a local softmax, P as the next MMA's operand, P.V,
// then one merge with its key-half partner. Q, K and V load straight into MMA fragments (V transposed with movmatrix). E is the
// served kernel's prefix-sum form per half-position: the half-position's bits are listed once, each head's 128 threads compute the
// entries' values (balanced whatever the clustering), scan them in FP32 into the head's table region, and a cell reads
// P[end] - P[start]. Only named barriers per head / pair; CTA barriers only when the work list moves to another half-position.
// Measured (4090 microbench, b64, K1's 1,024 real positions): 76.0 us per block against the served kernel's 111.9 (B9 FP16).
// Arithmetic class = B9 (FP16 E tables, HFMA2 state math, FP16 or FP32 MMA accumulate, FP32 softmax / prefix sums).
// Inputs are the builder's own plan tables (FP32, `BLOCK_TABLES` order) and the cell-major FP16 state copy
// (`egt_state_tiles.cast_state_cell_major`). Compile-time: HEADS, HD, HALVES (2 x batch), GATE_SCALE, OGATE, CAP, EXPORT_H,
// OUT_F16 (FP16 output instead of int8 codes), ACC16, GATE_TANH, SKEW, SMEM_BYTES (checked against the layout), H_F16 (the H export
// in FP16, each lane storing 8 consecutive keys: the edge site reads it with `logits_f16`).
//
// ---- TensorRT plugin port (trt_bt6_plugin, 2026-11-24; round 2: optional raw-logit export `hraw`) ------------------------------------------------------------------
// Changes against lc0ex terms_v9.cu, for the BT6TermsAttention plugin (one per term layer of the flat 768x16 trunk):
//   * HALVES is a RUNTIME argument (dynamic batch); the persistent work list is unchanged.
//   * QKV_INTERLEAVED: the trainer's qkv GEMM output as exported, per head [q | k | v] of 3 * HD (head h at h * 3 * HD),
//     instead of lc0ex's [Q | K | V] with the heads side by side.
//   * QKNORM: the layer's QK-norm (RMSNorm over head_dim, eps 1e-6, FP32 math, FP16 result -- the trainer's qLN / kLN)
//     applied to the raw q / k rows as they are loaded into the MMA fragments (one quad reduction per row).
//   * EDGES_SPLIT: the relation bits as INT32 pairs (bits 0-16, bits 17-33) per cell, [B, 64, 64, 2], built by TensorRT
//     from E with exact FP32 arithmetic; recomposed into the uint64 mask the list code reads.
//   * the template bias (AttnTemplates) is written by bt6_template_bias below into the plugin workspace and read as BIAS.
// Defaults below are the plugin build: FP16 output, no output gate, no cap, bias on, FP16 packs.
#include <cuda_fp16.h>
#include <cstdint>
#include "bt6_kernels.h"

// k1 (2026-11-25): the kernel body can be compiled a second time with other compile-time switches (bt6_egt_attn.cu,
// Kovax's 32-head [Q|K|V|G] BT6 flagship): BT6_ATTN_KERNEL names the kernel, BT6_ATTN_KERNEL_ONLY drops the template-bias
// kernel and the host side of the 24-head build.
#ifndef BT6_ATTN_KERNEL
#define BT6_ATTN_KERNEL terms_v9_attention
#endif

#ifndef HEADS
#define HEADS 24
#endif
#ifndef OGATE
#define OGATE 0
#endif
#ifndef OUT_F16
#define OUT_F16 1
#endif
#ifndef BIAS
#define BIAS 1
#endif
#ifndef PACKED
#define PACKED
#endif
#ifndef QKV_INTERLEAVED
#define QKV_INTERLEAVED 1
#endif
#ifndef QKNORM
#define QKNORM 1
#endif
#ifndef QKNORM_EPS
#define QKNORM_EPS 1e-6f
#endif
#ifndef EDGES_SPLIT
#define EDGES_SPLIT 1
#endif
#ifndef HD
#define HD 32
#endif
#ifndef OGATE
#define OGATE 1
#endif
#ifndef CAP
#define CAP 0
#endif
#ifndef EXPORT_H
#define EXPORT_H 0
#endif
#ifndef H_F16
#define H_F16 0
#endif
#ifndef OUT_F16
#define OUT_F16 0
#endif
#ifndef GATE_SCALE
#define GATE_SCALE 2.0f
#endif
#ifdef CONTROL_NO_SIGMA  // the fidelity gate's CONTROL (must fail): E without its scaled-coefficient term
#define SIGMA_TERM(norm_value, coefficient) 0.f
#else
#define SIGMA_TERM(norm_value, coefficient) ((norm_value) * (coefficient))
#endif
#ifndef BIAS
#define BIAS 0
#endif
namespace {

constexpr int kHeads = HEADS, kHd = HD, kWidth = kHeads * kHd, kStride = (OGATE ? 4 : 3) * kWidth;
#if QKV_INTERLEAVED
static_assert(!OGATE, "interleaved qkv: no output gate");
constexpr int kHeadStride = 3 * kHd, kKOff = kHd, kVOff = 2 * kHd;   // per head [q | k | v]
#else
constexpr int kHeadStride = kHd, kKOff = kWidth, kVOff = 2 * kWidth;  // [Q | K | V], heads side by side
#endif
#ifndef GROUP
#define GROUP 4
#endif
constexpr int kGroup = GROUP, kGroups = kHeads / kGroup;       // heads in flight per CTA; groups per half-position
constexpr int kThreads = 128 * kGroup, kRows = 32;             // 4 warps per head in flight (2 row tiles x 2 key halves)
constexpr int kKs = kHd / 16, kNd = kHd / 8, kHalfNd = kNd / 2;  // k-steps over head_dim; P.V n-tiles; the half each partner finalizes
constexpr int kTab = 40, kPar = 128;                           // E channels padded 34 -> 40; FP32 scalars per head
constexpr int kSPitch = 64 * 16 + 16;                          // bytes per row of one state plane (8 channels x FP16 per cell, +16)
constexpr int kPlane = kRows * kSPitch;
constexpr int kState = 2 * kPlane;
constexpr int kTabHead = (kRows + 64) * kTab * 2 + kTab * 4;   // FP16 row table [32][40] + column table [64][40] + FP32 scaled [40]
constexpr int kExWarp = 32 + 16 * (kHd / 2);                   // floats: (max, sum) per row + the partner's half of O
constexpr int kHeadBytes = kTabHead > 4 * kExWarp * 4 ? kTabHead : 4 * kExWarp * 4;
constexpr int kHeadsEnd = kState + kGroup * kHeadBytes;
constexpr int kListCta = 512;                                  // entries per half-position (real positions: at most 321)
constexpr int kOffTot = kHeadsEnd;                             // 16 ints: per-warp bit totals; then 16 floats: per-warp E totals
constexpr int kOffList = kOffTot + 128;                        // uint32 (row << 12 | key << 6 | channel) << 15 | S (FP16, sign dropped)
constexpr int kSmem = kOffList + 4 * kListCta;
static_assert(kListCta * 4 <= kTabHead, "the prefix lives in the head's table region");
static_assert(4 * kListCta / 128 == 16, "4 entries per head thread");
static_assert(kHeads % kGroup == 0 && kHd % 16 == 0 && kHd <= 64, "shape");
static_assert(kSmem <= 101376, "shared memory over the sm_89 / sm_120 opt-in ceiling");
static_assert(kHeadBytes % 16 == 0, "head region alignment");
#ifdef SMEM_BYTES
static_assert(kSmem == SMEM_BYTES, "the builder's shared-memory size disagrees with the kernel's layout");
#endif

#ifdef ACC16
using Acc = uint32_t[2];  // half2 x 2
#else
using Acc = float[4];
#endif

__device__ __forceinline__ uint32_t saddr(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void cp16(void* d, const void* s) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(saddr(d)), "l"(s));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_wait_all() { asm volatile("cp.async.wait_group 0;\n" ::); }
__device__ __forceinline__ void named_bar(int id, int count) { asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(count) : "memory"); }
__device__ __forceinline__ void mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma(uint32_t (&c)[2], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
               : "+r"(c[0]), "+r"(c[1])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void zero(float (&c)[4]) { c[0] = c[1] = c[2] = c[3] = 0.f; }
__device__ __forceinline__ void zero(uint32_t (&c)[2]) { c[0] = c[1] = 0u; }
// element i of an accumulator tile: (row g, col 2t), (g, 2t + 1), (g + 8, 2t), (g + 8, 2t + 1)
__device__ __forceinline__ float elem(const float (&c)[4], int i) { return c[i]; }
__device__ __forceinline__ float elem(const uint32_t (&c)[2], int i) {
  const __half2 h = *reinterpret_cast<const __half2*>(&c[i >> 1]);
  return (i & 1) ? __high2float(h) : __low2float(h);
}
__device__ __forceinline__ uint32_t transpose8x8(uint32_t x) {
  uint32_t y;
  asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(y) : "r"(x));
  return y;
}
__device__ __forceinline__ uint32_t ld32(const __half* p) { return __ldg(reinterpret_cast<const unsigned int*>(p)); }
__device__ __forceinline__ uint32_t pack(float lo, float hi) {
  const __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&h);
}
__device__ __forceinline__ float hsum(__half2 v) {
  const float2 f = __half22float2(v);
  return f.x + f.y;
}
// read / door / gate . e of one cell (16 FP16 channels in two uint4 halves x0, x1). Default: the lab kernel's HFMA2 chains
// (FP16 accumulation, 8 terms per half lane). k1 RDG_F32: FP32 FMAs (the weights converted at use; no extra registers).
__device__ __forceinline__ void rdg_dot(const __half2* x0, const __half2* x1, const __half2* wr, const __half2* wd,
                                        const __half2* wg, float& r, float& d, float& g) {
#if defined(DBG_NO_RDG)  // k5 pricing switch (results wrong): no state dots
  r = d = g = 0.f;
#elif defined(RDG_SPLIT)  // k1: two HFMA2 chains per output (channels 0-7 | 8-15, 4 products per half lane), summed in FP32
  __half2 r0 = __float2half2_rn(0.f), r1 = r0, d0 = r0, d1 = r0, g0 = r0, g1 = r0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    r0 = __hfma2(x0[i], wr[i], r0);
    d0 = __hfma2(x0[i], wd[i], d0);
    g0 = __hfma2(x0[i], wg[i], g0);
    r1 = __hfma2(x1[i], wr[4 + i], r1);
    d1 = __hfma2(x1[i], wd[4 + i], d1);
    g1 = __hfma2(x1[i], wg[4 + i], g1);
  }
  r = hsum(r0) + hsum(r1);
  d = hsum(d0) + hsum(d1);
  g = hsum(g0) + hsum(g1);
#elif defined(RDG_F32)
  r = d = g = 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 a = __half22float2(x0[i]), b = __half22float2(x1[i]);
    float2 w = __half22float2(wr[i]); r = fmaf(a.x, w.x, fmaf(a.y, w.y, r));
    w = __half22float2(wr[4 + i]); r = fmaf(b.x, w.x, fmaf(b.y, w.y, r));
    w = __half22float2(wd[i]); d = fmaf(a.x, w.x, fmaf(a.y, w.y, d));
    w = __half22float2(wd[4 + i]); d = fmaf(b.x, w.x, fmaf(b.y, w.y, d));
    w = __half22float2(wg[i]); g = fmaf(a.x, w.x, fmaf(a.y, w.y, g));
    w = __half22float2(wg[4 + i]); g = fmaf(b.x, w.x, fmaf(b.y, w.y, g));
  }
#else
  __half2 ar = __float2half2_rn(0.f), ad = ar, ag = ar;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    ar = __hfma2(x0[i], wr[i], ar);
    ad = __hfma2(x0[i], wd[i], ad);
    ag = __hfma2(x0[i], wg[i], ag);
    ar = __hfma2(x1[i], wr[4 + i], ar);
    ad = __hfma2(x1[i], wd[4 + i], ad);
    ag = __hfma2(x1[i], wg[4 + i], ag);
  }
  r = hsum(ar);
  d = hsum(ad);
  g = hsum(ag);
#endif
}

__device__ __forceinline__ float two_sigmoid(float x) {
#ifdef GATE_TANH
  float y;
  asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(0.5f * x));
  return 1.f + y;
#else
  return __fdividef(2.f, 1.f + __expf(-x));
#endif
}
__device__ __forceinline__ signed char code_of(float x) { return static_cast<signed char>(fminf(fmaxf(floorf(x + 0.5f), -127.f), 127.f)); }

#ifdef PACKED
// PACKED: the query / key tables as FP16 [heads][40][hd] (channels 34-39 zero), one 32-bit load per pair
__device__ __forceinline__ uint32_t table_pair(const float* __restrict__ table, int head, int channel, int dim) {
  return ld32(reinterpret_cast<const __half*>(table) + ((size_t)head * kTab + channel) * kHd + dim);
}
#else
// FP32 [heads][34][hd] pair of a table row's dims d, d + 1 as half2; channels past 34 read as zero (the tables' padding)
__device__ __forceinline__ uint32_t table_pair(const float* __restrict__ table, int head, int channel, int dim) {
  if (channel >= 34) return 0u;
  const float2 v = __ldg(reinterpret_cast<const float2*>(table + ((size_t)head * 34 + channel) * kHd + dim));
  return pack(v.x, v.y);
}
#endif

// One 16-byte load per lane; quad_words transposes so lane t holds words t, 4 + t, 8 + t, 12 + t.
__device__ __forceinline__ uint4 row_load(const __half* row, int t) { return __ldg(reinterpret_cast<const uint4*>(row) + t); }
__device__ __forceinline__ void quad_words(uint4 v, int t, uint32_t (&w)[4]) {
  uint32_t p[4] = {v.x, v.y, v.z, v.w};
  const bool b1 = t & 2, b0 = t & 1;
  uint32_t x0 = __shfl_xor_sync(0xffffffffu, b1 ? p[0] : p[2], 2);
  uint32_t x1 = __shfl_xor_sync(0xffffffffu, b1 ? p[1] : p[3], 2);
  if (b1) { p[0] = x0; p[1] = x1; } else { p[2] = x0; p[3] = x1; }
  x0 = __shfl_xor_sync(0xffffffffu, b0 ? p[0] : p[1], 1);
  x1 = __shfl_xor_sync(0xffffffffu, b0 ? p[2] : p[3], 1);
  if (b0) { p[0] = x0; p[2] = x1; } else { p[1] = x0; p[3] = x1; }
  w[0] = p[0]; w[1] = p[1]; w[2] = p[2]; w[3] = p[3];
}

// QK-norm of one row held by a quad: lane t holds dims 8t .. 8t + 7 (one row_load). RMS over the row's 32 dims in FP32
// (quad reduction), times the scale, rounded to FP16 -- the trainer's RMSNorm(x) * scale in its FP32 island.
__device__ __forceinline__ uint4 qk_norm_row(uint4 v, int t, const float* __restrict__ scale) {
#if QKNORM
  static_assert(kHd == 32, "qk_norm_row: head_dim 32 (4 lanes x 8 dims)");
  __half2* h = reinterpret_cast<__half2*>(&v);
  float2 f[4];
  float ss = 0.f;
#pragma unroll
  for (int m = 0; m < 4; ++m) {
    f[m] = __half22float2(h[m]);
    ss += f[m].x * f[m].x + f[m].y * f[m].y;
  }
  ss += __shfl_xor_sync(0xffffffffu, ss, 1);
  ss += __shfl_xor_sync(0xffffffffu, ss, 2);
  const float r = rsqrtf(ss * (1.f / kHd) + QKNORM_EPS);
  const float4 s0 = __ldg(reinterpret_cast<const float4*>(scale + 8 * t));
  const float4 s1 = __ldg(reinterpret_cast<const float4*>(scale + 8 * t + 4));
  h[0] = __floats2half2_rn(f[0].x * r * s0.x, f[0].y * r * s0.y);
  h[1] = __floats2half2_rn(f[1].x * r * s0.z, f[1].y * r * s0.w);
  h[2] = __floats2half2_rn(f[2].x * r * s1.x, f[2].y * r * s1.y);
  h[3] = __floats2half2_rn(f[3].x * r * s1.z, f[3].y * r * s1.w);
#endif
  return v;
}

#if EDGES_SPLIT
// relation bits as INT32 pairs (bits 0-16, bits 17-33) per cell
__device__ __forceinline__ unsigned long long compose_bits(int lo, int hi) {
  return static_cast<unsigned long long>(static_cast<unsigned int>(lo)) |
         (static_cast<unsigned long long>(static_cast<unsigned int>(hi)) << 17);
}
__device__ __forceinline__ ulonglong2 load_edges2(const int* __restrict__ edges, size_t cell) {
  const int4 v = __ldg(reinterpret_cast<const int4*>(edges + 2 * cell));
  return make_ulonglong2(compose_bits(v.x, v.y), compose_bits(v.z, v.w));
}
__device__ __forceinline__ unsigned long long load_edge1(const int* __restrict__ edges, size_t cell) {
  const int2 v = __ldg(reinterpret_cast<const int2*>(edges + 2 * cell));
  return compose_bits(v.x, v.y);
}
using EdgeT = int;
#else
__device__ __forceinline__ ulonglong2 load_edges2(const unsigned long long* __restrict__ edges, size_t cell) {
  return *reinterpret_cast<const ulonglong2*>(edges + cell);
}
__device__ __forceinline__ unsigned long long load_edge1(const unsigned long long* __restrict__ edges, size_t cell) {
  return __ldg(edges + cell);
}
using EdgeT = unsigned long long;
#endif

}  // namespace

// kStrideT: the row stride of the qkv input (3 * width, or 3 * width + heads * T with the template mixer's columns
// fused into the same GEMM output: round 2's fused_coef)
// kI8 (round 3): output 0 is INT8 codes round(x * inv_out_scale) clamped to +-127 (the W_h input of an INT8 graph,
// whose DequantizeLinear then reads it directly) instead of FP16.
// k5 (K5_VARIANT, the 32-head build only): kNew selects the k5 fresh path (STATE_LDG, FRESH_SHARE, SKEW) at compile time;
// kNew = false compiles k4's path (env BT6_EGT_K4=1 launches it).
#ifdef K5_VARIANT
template <int kStrideT, bool kI8, bool kSplit = false, bool kNew = true>
#else
template <int kStrideT, bool kI8, bool kSplit = false>
#endif
__global__ void __launch_bounds__(kThreads, 1) BT6_ATTN_KERNEL(
    void* __restrict__ out_ptr, void* __restrict__ hexp_ptr, const __half* __restrict__ qkv,
    const EdgeT* __restrict__ edges, const __half* __restrict__ norm, const __half* __restrict__ state,
    const float* __restrict__ qk_scale_t, const float* __restrict__ attack_t, const float* __restrict__ key_t,
    const float* __restrict__ query_t, const float* __restrict__ scaled_t, const float* __restrict__ cbias,
    const float* __restrict__ read_t, const float* __restrict__ door_t, const float* __restrict__ gate_t,
    const float* __restrict__ gate_b_t, const float* __restrict__ prescale, const __half* __restrict__ bias,
    const float* __restrict__ qn_scale, const float* __restrict__ kn_scale, int halves, __half* __restrict__ hraw,
    float raw_scale, float inv_out_scale
#ifdef HP_SCHED
    , int sched  // k5: > 0 = the remainder's half-positions cut into `sched` parts, one CTA each (kSplit builds only)
#endif
    ) {
  constexpr int kStride = kStrideT;
#ifndef K5_VARIANT
  constexpr bool kNew = true;  // (the k5 switches are not defined in this build)
#endif
  (void)kNew;
  (void)qn_scale;
  (void)kn_scale;
  static_assert(OUT_F16, "round 3: the output type is the template argument kI8");
  __half* out = static_cast<__half*>(out_ptr);
  int8_t* out8 = static_cast<int8_t*>(out_ptr);
  (void)prescale;
#if H_F16
  __half* hexp = static_cast<__half*>(hexp_ptr);
#else
  float* hexp = static_cast<float*>(hexp_ptr);
#endif
  extern __shared__ __align__(16) unsigned char smem[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int hs = warp >> 2, rt = (warp >> 1) & 1, kh = warp & 1;  // head slot, row tile, key half
  const unsigned char* sS = smem;
  unsigned char* head_region = smem + kState + hs * kHeadBytes;
  __half* sR = reinterpret_cast<__half*>(head_region);
  __half* sC = sR + kRows * kTab;
  float* sSig = reinterpret_cast<float*>(sC + 64 * kTab);
  float* sEx = reinterpret_cast<float*>(head_region);
  float* sP = reinterpret_cast<float*>(head_region);  // the head's E prefix over the list, once its tables are dead
  int* sWarpTot = reinterpret_cast<int*>(smem + kOffTot);
  float* sWarpSum = reinterpret_cast<float*>(smem + kOffTot + 64) + hs * 4;  // this head's 4 warps' E totals
  const uint32_t* sList = reinterpret_cast<const uint32_t*>(smem + kOffList);
  uint32_t* sListW = reinterpret_cast<uint32_t*>(smem + kOffList);
  const int bar_head = 1 + hs, bar_pair = 5 + 2 * hs + rt;

  // The work list over (half-position, head group) units, v8's L2-aware order: full rounds of whole half-positions (sister halves on
  // neighbouring CTAs), then the remainder's units in equal contiguous slices.
  const int S = gridDim.x, full = halves / S, rest = halves - full * S;
  const int t0 = static_cast<int>((long long)blockIdx.x * rest * kGroups / S);
  const int t1 = static_cast<int>((long long)(blockIdx.x + 1) * rest * kGroups / S);
#if defined(TAIL_SPLIT) && defined(HP_SCHED)
  // k5 half-position-aligned remainder (sched = m > 0, kSplit builds): each of the `rest` half-positions left after the full
  // rounds is cut into m parts of contiguous heads [kHeads j / m, kHeads (j + 1) / m), one CTA per part (CTA b:
  // half-position b / m, part b % m), run kGroup heads at a time (the last unit of a part may have idle head slots), so a
  // CTA loads ONE state for its remainder work (k4's contiguous slices straddle half-positions: a second / third state
  // load, list build and cold start for the slowest CTAs). When m * rest > S (m divides kGroups, checked by the host) the
  // parts past S are cut into 1-head pieces, one per CTA, as in k3. Per-head arithmetic unchanged.
  const int bx = static_cast<int>(blockIdx.x);
  const int hm = kSplit ? sched : 0;
  const int runits = rest * kGroups, R2 = runits / S, E = runits - R2 * S;
  const int ppu = hm || !kSplit || E == 0 ? 1 : E * kGroup <= S ? kGroup : (kGroup % 2 == 0 && E * 2 <= S) ? 2 : 1;
  int hpn = 0, hpu0 = 0, hph = 0, hsz = 1, hE = 0;
  if (hm) {
    const int parts = hm * rest;
    if (bx < parts) {
      const int j = bx % hm;
      hpu0 = kHeads * j / hm;                                           // first head of the part
      hph = kHeads * (j + 1) / hm - hpu0;                               // its heads
      hpn = (hph + kGroup - 1) / kGroup;
    }
    hsz = kGroups / hm;
    hE = parts > S ? (parts - S) * hsz : 0;
  }
  const int nq = hm ? full * kGroups + hpn + (bx < hE * kGroup ? 1 : 0)
                    : ppu == 1 ? full * kGroups + (t1 - t0) : full * kGroups + R2 + (bx < E * ppu ? 1 : 0);
#elif defined(TAIL_SPLIT)
  // k3 tail split: the remainder's units as R2 contiguous units per CTA, and the E units left over (the last, mostly empty
  // round of k2's list) cut into ppu pieces of kGroup / ppu heads, one piece per CTA (ppu = kGroup when 4E <= S, 2 when
  // 2E <= S; otherwise k2's contiguous slices). Each head's arithmetic is unchanged: only which CTA / slot computes it.
  const int runits = rest * kGroups, R2 = runits / S, E = runits - R2 * S;
  const int ppu = !kSplit || E == 0 ? 1 : E * kGroup <= S ? kGroup : (kGroup % 2 == 0 && E * 2 <= S) ? 2 : 1;
  const int nq = ppu == 1 ? full * kGroups + (t1 - t0) : full * kGroups + R2 + (static_cast<int>(blockIdx.x) < E * ppu ? 1 : 0);
#else
  const int nq = full * kGroups + (t1 - t0);
#endif
  if (nq <= 0) return;
#if defined(EGT_PHASE) && defined(BT6_ATTN_KERNEL_ONLY)  // k5 debug build: per-warp phase clocks (lane 0), summed over warps into g_egt_ph[warp][slot]
  long long ph_acc[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  long long ph_prev = clock64();
#define TICK(slot) do { if (lane == 0) { const long long _n = clock64(); ph_acc[slot] += _n - ph_prev; ph_prev = _n; } } while (0)
#else
#define TICK(slot) do { } while (0)
#endif
#if defined(EGT_TIMING) && defined(TAIL_SPLIT)
  unsigned long long t_start, t_end; int nfresh = 0;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_start));
#endif

  int current = -1;
  __half2 nrm[2][4];              // this lane's cells' S: rows g, g + 8 of the row tile; keys kh * 32 + 8n + 2t, + 1
  uint32_t nz = 0;                // which of its 16 cells (k = 8a + 2n + b) have edge bits
  int estart = 0;                 // this owner lane's first entry (owner = row tile, key half, lane; both head slots share it)
  unsigned long long counts = 0;  // its bits per cell k in 4-bit fields (a cell has at most 8 of 34 on real positions)
  bool eover = false;             // the half-position's bits exceed kListCta: every lane walks its cells' masks (exact, slow)
  int etotal = 0;                 // entries in the half-position's list
  for (int q = 0; q < nq; ++q) {
#ifdef TAIL_SPLIT
    int u, hoff = 0;
    bool idle = false;  // this head slot has no head in this unit (a split tail piece)
    if (q < full * kGroups) {
      u = ((q / kGroups) * S + blockIdx.x) * kGroups + q % kGroups;
#ifdef HP_SCHED
    } else if (hm) {
      const int qq = q - full * kGroups;
      if (qq < hpn) {  // this CTA's part of remainder half-position bx / hm: heads hpu0 + kGroup qq + [0, kGroup)
        u = (full * S + bx / hm) * kGroups;
#ifdef EGT_CONTROL_SPLIT_HEADS  // must-fail control: every unit of a part computes the part's first heads
        hoff = hpu0;
#else
        hoff = hpu0 + kGroup * qq;
#endif
        idle = hs >= hph - kGroup * qq;
      } else {  // a 1-head piece of the parts past S
        const int e = bx / kGroup, lp = S + e / hsz;
        u = (full * S + lp / hm) * kGroups + (lp % hm) * hsz + e % hsz;
#ifdef EGT_CONTROL_SPLIT_HEADS
        hoff = 0;
#else
        hoff = bx % kGroup;
#endif
        idle = hs >= 1;
      }
#endif
    } else if (ppu == 1) {
      u = full * S * kGroups + t0 + (q - full * kGroups);
    } else if (q - full * kGroups < R2) {
      u = full * S * kGroups + blockIdx.x * R2 + (q - full * kGroups);
    } else {  // this CTA's piece of a leftover unit: heads (blockIdx % ppu) * kGroup / ppu + [0, kGroup / ppu)
      u = full * S * kGroups + S * R2 + blockIdx.x / ppu;
      const int nh = kGroup / ppu;
#ifdef EGT_CONTROL_SPLIT_HEADS  // must-fail control: every piece computes the first heads of its unit (the others stay unwritten)
      hoff = 0;
#else
      hoff = (blockIdx.x % ppu) * nh;
#endif
      idle = hs >= nh;
    }
#else
    const int u = q < full * kGroups ? ((q / kGroups) * S + blockIdx.x) * kGroups + q % kGroups
                                     : full * S * kGroups + t0 + (q - full * kGroups);
#endif
    const int hp = u / kGroups, grp = u % kGroups, sample = hp >> 1, row0 = (hp & 1) * kRows;
    const bool fresh = hp != current;
#if defined(EGT_TIMING) && defined(TAIL_SPLIT)
    nfresh += fresh;
#endif
    if (fresh) {
      __syncthreads();  // every warp is past its last read of the previous state
      const __half* src = state + ((size_t)sample * 64 + row0) * 64 * 16;
#if defined(STATE_LDG)
      if (kNew) {
      // k5: the state copy as plain 16-byte loads + shared stores, STATE_LDG at a time per thread (cp.async measured ~15K
      // cycles of every warp per fresh half-position: 8 instructions per thread throttled to ~4 bytes / cycle per SM)
      {
        static_assert(kRows * 64 * 2 % kThreads == 0 && (kRows * 64 * 2 / kThreads) % STATE_LDG == 0, "state copy batches");
#pragma unroll
        for (int j0 = 0; j0 < kRows * 64 * 2 / kThreads; j0 += STATE_LDG) {
          uint4 v[STATE_LDG];
#pragma unroll
          for (int j = 0; j < STATE_LDG; ++j) {
            const int i = tid + (j0 + j) * kThreads, chunk = i & 1, cell = (i >> 1) & 63, r = i >> 7;
            v[j] = __ldg(reinterpret_cast<const uint4*>(src + ((size_t)r * 64 + cell) * 16 + chunk * 8));
          }
#pragma unroll
          for (int j = 0; j < STATE_LDG; ++j) {
            const int i = tid + (j0 + j) * kThreads, chunk = i & 1, cell = (i >> 1) & 63, r = i >> 7;
            *reinterpret_cast<uint4*>(smem + chunk * kPlane + r * kSPitch + cell * 16) = v[j];
          }
        }
      }
      } else
#endif
      {
      for (int i = tid; i < kRows * 64 * 2; i += kThreads) {
        const int chunk = i & 1, cell = (i >> 1) & 63, r = i >> 7;
#ifndef DBG_NO_STATE
        cp16(smem + chunk * kPlane + r * kSPitch + cell * 16, src + ((size_t)r * 64 + cell) * 16 + chunk * 8);
#else
        (void)chunk; (void)cell; (void)r; (void)src;
#endif
      }
      cp_commit();
      }
      current = hp;
    }
    TICK(0);
#ifdef TAIL_SPLIT
    if (idle) {
      if (fresh) {
        cp_wait_all();
        __syncthreads();
        (void)__syncthreads_or(0);
        __syncthreads();
      }
      break;
    }
    const int h = grp * kGroup + hoff + hs;
#else
    const int h = grp * kGroup + hs;
#endif
#ifdef DBG_HOT  // k5 pricing switch (results wrong): the per-unit global loads read one small region (L1 hits)
#define HOTI(x, hot) (hot)
    const __half* base = qkv;
#else
#define HOTI(x, hot) (x)
    const __half* base = qkv + (size_t)sample * 64 * kStride + h * kHeadStride;
#endif

    // ---- loads for q.k and the tables, all issued ahead of the MMAs
    static_assert(kHd == 32 && kKs == 2 && kNd == 4, "ld128 is written for head_dim 32");
    const __half* qrow = base + (size_t)HOTI(row0 + rt * 16 + g, g) * kStride;
    uint4 qraw0 = row_load(qrow, t), qraw1 = row_load(qrow + HOTI(8, 0) * kStride, t);
    uint4 kraw[4];  // keys kh * 32 + 8n + g
#pragma unroll
    for (int n = 0; n < 4; ++n) kraw[n] = row_load(base + kKOff + (size_t)HOTI(kh * 32 + 8 * n + g, g) * kStride, t);

    uint32_t tqf[3][kKs][2], tkf[3][kKs][2];
#if defined(EGT_TAB128) && defined(BT6_ATTN_KERNEL_ONLY)
    // k5 probe: each table tile row (channel 8 nn + g, 32 dims = 64 bytes) as one 16-byte load per lane + the quad transpose
    // (the K-fragment trick): words t, 4 + t, 8 + t, 12 + t = [ks 0][0], [0][1], [1][0], [1][1]. Same values as table_pair.
    uint4 tqr[3], tkr[3];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      const int nn = kh ? (i < 2 ? 3 + i : 4) : i;
      tqr[i] = row_load(reinterpret_cast<const __half*>(query_t) + ((size_t)h * kTab + 8 * nn + g) * kHd, t);
    }
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      const int nn = rt ? (i < 2 ? 3 + i : 4) : i;
      tkr[i] = row_load(reinterpret_cast<const __half*>(key_t) + ((size_t)h * kTab + 8 * nn + g) * kHd, t);
    }
#else
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      const int nn = kh ? (i < 2 ? 3 + i : 4) : i;  // kh = 1 has two tiles; its third slot repeats tile 4 and is not stored
#pragma unroll
      for (int ks = 0; ks < kKs; ++ks) {
        tqf[i][ks][0] = table_pair(query_t, HOTI(h, 0), HOTI(8 * nn + g, g), 16 * ks + 2 * t);
        tqf[i][ks][1] = table_pair(query_t, HOTI(h, 0), HOTI(8 * nn + g, g), 16 * ks + 2 * t + 8);
      }
    }
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      const int nn = rt ? (i < 2 ? 3 + i : 4) : i;  // the row tile picks the column table's channel tiles: 0-2 or 3-4
#pragma unroll
      for (int ks = 0; ks < kKs; ++ks) {
        tkf[i][ks][0] = table_pair(key_t, HOTI(h, 0), HOTI(8 * nn + g, g), 16 * ks + 2 * t);
        tkf[i][ks][1] = table_pair(key_t, HOTI(h, 0), HOTI(8 * nn + g, g), 16 * ks + 2 * t + 8);
      }
    }
#endif

    // Transpose once every load of this unit is in flight (QK-norm first: the rows are still lane-contiguous).
#if QKNORM
    qraw0 = qk_norm_row(qraw0, t, qn_scale);
    qraw1 = qk_norm_row(qraw1, t, qn_scale);
#pragma unroll
    for (int n = 0; n < 4; ++n) kraw[n] = qk_norm_row(kraw[n], t, kn_scale);
#endif
    uint32_t qa[kKs][4];
    {
      uint32_t w0[4], w1[4];
      quad_words(qraw0, t, w0);
      quad_words(qraw1, t, w1);
      qa[0][0] = w0[0]; qa[0][2] = w0[1]; qa[1][0] = w0[2]; qa[1][2] = w0[3];
      qa[0][1] = w1[0]; qa[0][3] = w1[1]; qa[1][1] = w1[2]; qa[1][3] = w1[3];
    }
#if defined(EGT_TAB128) && defined(BT6_ATTN_KERNEL_ONLY)
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      uint32_t w[4];
      quad_words(tqr[i], t, w);
      tqf[i][0][0] = w[0]; tqf[i][0][1] = w[1]; tqf[i][1][0] = w[2]; tqf[i][1][1] = w[3];
      quad_words(tkr[i], t, w);
      tkf[i][0][0] = w[0]; tkf[i][0][1] = w[1]; tkf[i][1][0] = w[2]; tkf[i][1][1] = w[3];
    }
#endif
    uint32_t kb[4][kKs][2];
#pragma unroll
    for (int n = 0; n < 4; ++n) {
      uint32_t w[4];
      quad_words(kraw[n], t, w);
      kb[n][0][0] = w[0]; kb[n][0][1] = w[1]; kb[n][1][0] = w[2]; kb[n][1][1] = w[3];
    }
    // ---- q.k and this warp's share of the head's E tables: row table (its 16 rows; channel tiles 0-2 or 3-4 by key half) and
    // column table (its 32 keys, whose A fragments are its K fragments; channel tiles 0-2 or 3-4 by row tile)
    Acc s[4];
#pragma unroll
    for (int n = 0; n < 4; ++n) {
      zero(s[n]);
#pragma unroll
      for (int ks = 0; ks < kKs; ++ks) mma(s[n], qa[ks], kb[n][ks][0], kb[n][ks][1]);
    }
    Acc racc[3], cacc[2][3];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      zero(racc[i]);
      if (!kh || i < 2) {
#pragma unroll
        for (int ks = 0; ks < kKs; ++ks) mma(racc[i], qa[ks], tqf[i][ks][0], tqf[i][ks][1]);
      }
    }
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int i = 0; i < 3; ++i) {
        zero(cacc[mt][i]);
        if (!rt || i < 2) {
#pragma unroll
          for (int ks = 0; ks < kKs; ++ks) {
            const uint32_t a[4] = {kb[2 * mt][ks][0], kb[2 * mt + 1][ks][0], kb[2 * mt][ks][1], kb[2 * mt + 1][ks][1]};
            mma(cacc[mt][i], a, tkf[i][ks][0], tkf[i][ks][1]);
          }
        }
      }
    TICK(1);
    if (fresh) {  // uniform: every warp of the CTA takes this branch in the same unit
      nz = 0;
      counts = 0;
      int count = 0;
#ifdef FRESH_SHARE
      // k5: only head slot 0 loads the half-position's edge bits and S (the other slots loaded the same 20 KB again:
      // 4 x the global traffic of the fresh block); it leaves each owner lane's S, bit counts, mask and list offset in
      // head region 0 (idle until the tables are written, after the fresh block's last barrier) for the other slots.
      static_assert(128 * 48 <= kHeadBytes, "fresh scratch in a head region");
#ifdef EGT_CONTROL_FRESH  // must-fail control: the other head slots read the neighbouring lane's record
      uint4* scr = reinterpret_cast<uint4*>(smem + kState) + ((warp & 3) * 32 + (hs ? lane ^ 1 : lane)) * 3;
#else
      uint4* scr = reinterpret_cast<uint4*>(smem + kState) + ((warp & 3) * 32 + lane) * 3;
#endif
      int excl = 0;
      if (!kNew || hs == 0) {
#endif
#pragma unroll
      for (int a = 0; a < 2; ++a) {
        const size_t row = (size_t)sample * 64 + row0 + rt * 16 + g + 8 * a;
#pragma unroll
        for (int n = 0; n < 4; ++n) {
          const int key = kh * 32 + 8 * n + 2 * t;
          nrm[a][n] = *reinterpret_cast<const __half2*>(norm + row * 64 + key);
          const ulonglong2 e = load_edges2(edges, row * 64 + key);
#pragma unroll
          for (int b = 0; b < 2; ++b) {
            const int k = a * 8 + n * 2 + b, bits = __popcll(b ? e.y : e.x);
            nz |= (bits ? 1u : 0u) << k;
            counts |= static_cast<unsigned long long>(bits < 15 ? bits : 15) << (4 * k);
            count += bits;
          }
        }
      }
      int inclusive = count;  // warp scan of the owner lanes' bit counts
#pragma unroll
      for (int offset = 1; offset < 32; offset <<= 1) {
        const int v = __shfl_up_sync(0xffffffffu, inclusive, offset);
        if (lane >= offset) inclusive += v;
      }
      if (hs == 0 && lane == 31) sWarpTot[warp] = inclusive;
#ifdef FRESH_SHARE
      excl = inclusive - count;
      if (kNew) {
        const uint32_t* nw = reinterpret_cast<const uint32_t*>(&nrm[0][0]);
        scr[0] = make_uint4(nw[0], nw[1], nw[2], nw[3]);
        scr[1] = make_uint4(nw[4], nw[5], nw[6], nw[7]);
        scr[2] = make_uint4(static_cast<uint32_t>(counts), static_cast<uint32_t>(counts >> 32), nz, static_cast<uint32_t>(excl));
      }
      }
#endif
      bool cell_over = false;
#pragma unroll
      for (int k = 0; k < 16; ++k) cell_over |= ((counts >> (4 * k)) & 15u) == 15u;
      cp_wait_all();
      __syncthreads();  // the state is in; head slot 0's warp totals are visible
#ifdef FRESH_SHARE
      if (kNew && hs != 0) {  // read before the next CTA barrier: slot 0 writes its tables into this region only after the fresh block
        const uint4 x0 = scr[0], x1 = scr[1], x2 = scr[2];
        uint32_t* nw = reinterpret_cast<uint32_t*>(&nrm[0][0]);
        nw[0] = x0.x; nw[1] = x0.y; nw[2] = x0.z; nw[3] = x0.w;
        nw[4] = x1.x; nw[5] = x1.y; nw[6] = x1.z; nw[7] = x1.w;
        counts = static_cast<unsigned long long>(x2.x) | (static_cast<unsigned long long>(x2.y) << 32);
        nz = x2.z;
        excl = static_cast<int>(x2.w);
      }
#endif
      int base_count = 0, total = 0;
#pragma unroll
      for (int w = 0; w < 4; ++w) {
        const int v = sWarpTot[w];
        base_count += w < (warp & 3) ? v : 0;
        total += v;
      }
      eover = __syncthreads_or(total > kListCta || cell_over) != 0;
#ifdef FRESH_SHARE
      estart = base_count + excl;
#else
      estart = base_count + inclusive - count;
#endif
      etotal = total;
      if (hs == 0 && !eover) {  // the owner's entries, in cell order, with the cell's S
        int at = estart;
#pragma unroll
        for (int a = 0; a < 2; ++a) {
          const int rl = rt * 16 + g + 8 * a;
          const size_t row = (size_t)sample * 64 + row0 + rl;
#pragma unroll
          for (int n = 0; n < 4; ++n) {
            const ulonglong2 e = load_edges2(edges, row * 64 + kh * 32 + 8 * n + 2 * t);
#pragma unroll
            for (int b = 0; b < 2; ++b) {
              const int key = kh * 32 + 8 * n + 2 * t + b;
              const __half nb = b ? __high2half(nrm[a][n]) : __low2half(nrm[a][n]);
              const uint32_t sbits = static_cast<uint32_t>(__half_as_ushort(nb)) & 0x7FFFu;
              unsigned long long m = b ? e.y : e.x;
              while (m) {
                const uint32_t c = static_cast<uint32_t>(__ffsll(static_cast<long long>(m)) - 1);
                sListW[at++] = ((static_cast<uint32_t>(rl) << 12 | static_cast<uint32_t>(key) << 6 | c) << 15) | sbits;
                m &= m - 1;
              }
            }
          }
        }
      }
      __syncthreads();  // the list is visible to every head slot
#ifdef SKEW
      // probe (09-28): delay the odd head slots by SKEW cycles so that, on every SMSP, two warps run the FMA-heavy assembly while
      // the other two run their MMAs (the four head slots are otherwise in lockstep and the pipes take turns)
#ifdef SKEW_GRADED  // slot hs waits hs * SKEW / 4
      const long long wait = (long long)hs * SKEW / 4;
#else
      const long long wait = (hs & 1) ? SKEW : 0;
#endif
      if (wait && kNew) {
        const long long start = clock64();
        while (clock64() - start < wait) {
        }
      }
#endif
    }
    TICK(7);
    // the head's scalars, state weights and cbias: issued before the barrier so their latency hides under it
    const float qk_scale = __ldg(qk_scale_t + h), gate_b = __ldg(gate_b_t + h);
    __half2 wr[8], wd[8], wg[8];
#ifdef PACKED  // read_t = 24 packed half2 per head: read, door, gate x 8 channel pairs
    {
      const uint4* wv = reinterpret_cast<const uint4*>(read_t) + HOTI(h, 0) * 6;
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const uint4 x = __ldg(wv + i), y = __ldg(wv + 2 + i), z = __ldg(wv + 4 + i);
        const __half2* hx = reinterpret_cast<const __half2*>(&x);
        const __half2* hy = reinterpret_cast<const __half2*>(&y);
        const __half2* hz = reinterpret_cast<const __half2*>(&z);
#pragma unroll
        for (int m = 0; m < 4; ++m) { wr[4 * i + m] = hx[m]; wd[4 * i + m] = hy[m]; wg[4 * i + m] = hz[m]; }
      }
      (void)door_t;
      (void)gate_t;
    }
#else
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float2 r = __ldg(reinterpret_cast<const float2*>(read_t + h * 16) + i);
      const float2 d = __ldg(reinterpret_cast<const float2*>(door_t + h * 16) + i);
      const float2 x = __ldg(reinterpret_cast<const float2*>(gate_t + h * 16) + i);
      wr[i] = __floats2half2_rn(r.x, r.y);
      wd[i] = __floats2half2_rn(d.x, d.y);
      wg[i] = __floats2half2_rn(x.x, x.y);
    }
#endif
    const float* cbh = cbias + (size_t)h * 4096;
    __half2 cb[2][4];
#pragma unroll
    for (int a = 0; a < 2; ++a)
#pragma unroll
      for (int n = 0; n < 4; ++n)
#ifdef PACKED  // cbias as FP16 [heads][64][64]
        cb[a][n] = __ldg(reinterpret_cast<const __half2*>(cbias) + HOTI((h * 4096 + (row0 + rt * 16 + g + 8 * a) * 64 + kh * 32 + 8 * n + 2 * t) / 2, t));
#else
        cb[a][n] = __float22half2_rn(
            __ldg(reinterpret_cast<const float2*>(cbh + (row0 + rt * 16 + g + 8 * a) * 64 + kh * 32 + 8 * n + 2 * t)));
#endif
#if BIAS  // the smolgen bias of this lane's cells, issued with the other head loads
    __half2 bs[2][4];
#pragma unroll
    for (int a = 0; a < 2; ++a)
#pragma unroll
      for (int n = 0; n < 4; ++n)
        bs[a][n] = __ldg(reinterpret_cast<const __half2*>(bias) +
                         (((size_t)sample * kHeads + h) * 4096 + (row0 + rt * 16 + g + 8 * a) * 64 + kh * 32 + 8 * n + 2 * t) / 2);
#else
    (void)bias;
#endif
    named_bar(bar_head, 128);  // B0: the head slot's previous exchange has been read
    {
      const int r = rt * 16 + g;
#pragma unroll
      for (int i = 0; i < 3; ++i) {
        if (!kh || i < 2) {
          const int c = 8 * (kh ? 3 + i : i) + 2 * t;
          const float a0 = c < 34 ? __ldg(attack_t + h * 34 + c) : 0.f, a1 = c + 1 < 34 ? __ldg(attack_t + h * 34 + c + 1) : 0.f;
          *reinterpret_cast<uint32_t*>(sR + r * kTab + c) = pack(elem(racc[i], 0) + a0, elem(racc[i], 1) + a1);
          *reinterpret_cast<uint32_t*>(sR + (r + 8) * kTab + c) = pack(elem(racc[i], 2) + a0, elem(racc[i], 3) + a1);
        }
      }
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) {
        const int kr = kh * 32 + 16 * mt + g;
#pragma unroll
        for (int i = 0; i < 3; ++i) {
          if (!rt || i < 2) {
            const int c = 8 * (rt ? 3 + i : i) + 2 * t;
            *reinterpret_cast<uint32_t*>(sC + kr * kTab + c) = pack(elem(cacc[mt][i], 0), elem(cacc[mt][i], 1));
            *reinterpret_cast<uint32_t*>(sC + (kr + 8) * kTab + c) = pack(elem(cacc[mt][i], 2), elem(cacc[mt][i], 3));
          }
        }
      }
      if (rt == 0 && kh == 0) {
        sSig[lane] = __ldg(scaled_t + h * 34 + lane);                                         // channels 0-31
        if (lane < kTab - 32) sSig[32 + lane] = lane < 2 ? __ldg(scaled_t + h * 34 + 32 + lane) : 0.f;  // 32, 33, padding
      }
    }
    named_bar(bar_head, 128);  // B1: the tables are visible
    TICK(2);
#if BIAS
    // Round 2: the RAW logits the edge sites read, Hraw = q.k / sqrt(hd) + template bias (QK-normed q, k; FP32 sum,
    // FP16 store), [B, heads, 64, 64]; the quad transpose of the H_F16 export (8 consecutive keys per lane, 16 bytes).
    if (hraw != nullptr) {
#pragma unroll
      for (int a = 0; a < 2; ++a) {
        const int rl = rt * 16 + g + 8 * a;
        uint32_t p[4];
#pragma unroll
        for (int n = 0; n < 4; ++n) {
#ifdef HRAW_CONTROL_NO_BIAS  // the fidelity gate's CONTROL (must fail): raw logits without the template bias
          const float2 bsf = make_float2(0.f, 0.f);
#else
          const float2 bsf = __half22float2(bs[a][n]);
#endif
          p[n] = pack(elem(s[n], 2 * a) * raw_scale + bsf.x, elem(s[n], 2 * a + 1) * raw_scale + bsf.y);
        }
        const bool b1 = t & 2, b0 = t & 1;
        uint32_t x0 = __shfl_xor_sync(0xffffffffu, b1 ? p[0] : p[2], 2);
        uint32_t x1 = __shfl_xor_sync(0xffffffffu, b1 ? p[1] : p[3], 2);
        if (b1) { p[0] = x0; p[1] = x1; } else { p[2] = x0; p[3] = x1; }
        x0 = __shfl_xor_sync(0xffffffffu, b0 ? p[0] : p[1], 1);
        x1 = __shfl_xor_sync(0xffffffffu, b0 ? p[2] : p[3], 1);
        if (b0) { p[0] = x0; p[2] = x1; } else { p[1] = x0; p[3] = x1; }
        *reinterpret_cast<uint4*>(hraw + (((size_t)sample * kHeads + h) * 64 + row0 + rl) * 64 + kh * 32 + 8 * t) =
            make_uint4(p[0], p[1], p[2], p[3]);
      }
    }
#endif

    // ---- the logit assembly per cell: E entries, norm * cbias, read / door / gate from the shared state (HFMA2),
    // H = (F + read)(1 + door)
    // V for P.V, loaded now so its latency hides under the assembly (transposed at use)
    const __half* vrow = base + kVOff + (size_t)HOTI(kh * 32 + g, g) * kStride;
    uint4 vload[2][2];  // rows kh * 32 + g + 16 kk + 8 j, transposed into fragment words just before P.V
#pragma unroll
    for (int kk = 0; kk < 2; ++kk)
#pragma unroll
      for (int j = 0; j < 2; ++j) vload[kk][j] = row_load(vrow + (size_t)HOTI(16 * kk + 8 * j, 0) * kStride, t);
    const bool walk = eover;  // the half-position's bits did not fit the list: every lane reads its cells' masks
#ifndef DBG_NO_E
    if (!walk) {  // this head's E values: 4 consecutive entries per head thread, then an FP32 prefix over the list
      const int ht = (warp & 3) * 32 + lane;  // thread within the head
      float v[4], run = 0.f;
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        // slots past the list decode as entry 0 (cell 0, channel 0): every read stays inside this head's region (a stale word could
        // name channel 63, past the head's table; racecheck 09-29) and their value is selected out below
        const bool live = 4 * ht + r < etotal;
        const uint32_t entry = live ? sList[4 * ht + r] : 0u;
        const int c = static_cast<int>((entry >> 15) & 63u), key = static_cast<int>((entry >> 21) & 63u);
        const int rl = static_cast<int>(entry >> 27);
        const float nb = __half2float(__ushort_as_half(static_cast<unsigned short>(entry & 0x7FFFu)));
        const float x = __half2float(sR[rl * kTab + c]) + __half2float(sC[key * kTab + c]) + SIGMA_TERM(nb, sSig[c]);
        run += live ? x : 0.f;
        v[r] = run;
      }
      float incl = run;  // warp scan of the threads' sums
#pragma unroll
      for (int offset = 1; offset < 32; offset <<= 1) {
        const float x = __shfl_up_sync(0xffffffffu, incl, offset);
        if (lane >= offset) incl += x;
      }
      if (lane == 31) sWarpSum[warp & 3] = incl;
      named_bar(bar_head, 128);  // the head's values are computed (its tables are dead) and its warp totals visible
      float offset_sum = incl - run;
#pragma unroll
      for (int w = 0; w < 4; ++w) offset_sum += w < (warp & 3) ? sWarpSum[w] : 0.f;
#pragma unroll
      for (int r = 0; r < 4; ++r) sP[4 * ht + r] = offset_sum + v[r];
      named_bar(bar_head, 128);  // the prefix is visible
    }
#endif
    TICK(8);
    int cell_start = estart;  // running start of cell k's entries
    float hv[4][4];
    __half2 dg[4][4];  // (door, 2 sigmoid(gate)) per cell
    // The rare overflow half-position takes the copy below.
    if (!walk) {
#pragma unroll
    for (int a = 0; a < 2; ++a) {
      const int rl = rt * 16 + g + 8 * a;
#pragma unroll
      for (int n = 0; n < 4; ++n) {
        const float2 nr = __half22float2(nrm[a][n]), cbf = __half22float2(cb[a][n]);
#if BIAS
        const float2 bsf = __half22float2(bs[a][n]);
#else
        const float2 bsf = make_float2(0.f, 0.f);
#endif
#pragma unroll
        for (int b = 0; b < 2; ++b) {
          const int k = a * 8 + n * 2 + b, key = kh * 32 + 8 * n + 2 * t + b;
#ifdef DBG_NO_STATE  // pricing switches (results wrong): drop one term
          const uint4 c0 = make_uint4(0, 0, 0, 0), c1 = c0;
#else
          const uint4 c0 = *reinterpret_cast<const uint4*>(sS + rl * kSPitch + key * 16);
          const uint4 c1 = *reinterpret_cast<const uint4*>(sS + kPlane + rl * kSPitch + key * 16);
#endif
          const __half2* x0 = reinterpret_cast<const __half2*>(&c0);
          const __half2* x1 = reinterpret_cast<const __half2*>(&c1);
          float f_ar, f_ad, f_ag;
          rdg_dot(x0, x1, wr, wd, wg, f_ar, f_ad, f_ag);
          const float nb = b ? nr.y : nr.x;
          const int cnt = static_cast<int>((counts >> (4 * k)) & 15u);
#ifdef DBG_NO_E
          hv[n][2 * a + b] = 0.f;
#else
          hv[n][2 * a + b] = cnt ? sP[cell_start + cnt - 1] - (cell_start ? sP[cell_start - 1] : 0.f) : 0.f;
#endif
          cell_start += cnt;
          hv[n][2 * a + b] += elem(s[n], 2 * a + b) * qk_scale + nb * (b ? cbf.y : cbf.x) + f_ar + (b ? bsf.y : bsf.x);
#ifdef DOOR_F32  // k1: the door applied here in FP32 (not via its FP16 copy in dg below)
          hv[n][2 * a + b] *= 1.f + f_ad;
#endif
          const float x = f_ag + gate_b;
#if CAP
          dg[n][2 * a + b] = __floats2half2_rn(f_ad, x >= 0.f ? 1.f : two_sigmoid(x));
#else
          dg[n][2 * a + b] = __floats2half2_rn(f_ad, two_sigmoid(x));
#endif
        }
      }
    }
    } else {
#pragma unroll
    for (int a = 0; a < 2; ++a) {
      const int rl = rt * 16 + g + 8 * a;
#pragma unroll
      for (int n = 0; n < 4; ++n) {
        const float2 nr = __half22float2(nrm[a][n]), cbf = __half22float2(cb[a][n]);
#if BIAS
        const float2 bsf = __half22float2(bs[a][n]);
#else
        const float2 bsf = make_float2(0.f, 0.f);
#endif
#pragma unroll
        for (int b = 0; b < 2; ++b) {
          const int k = a * 8 + n * 2 + b, key = kh * 32 + 8 * n + 2 * t + b;
#ifdef DBG_NO_STATE  // pricing switches (results wrong): drop one term
          const uint4 c0 = make_uint4(0, 0, 0, 0), c1 = c0;
#else
          const uint4 c0 = *reinterpret_cast<const uint4*>(sS + rl * kSPitch + key * 16);
          const uint4 c1 = *reinterpret_cast<const uint4*>(sS + kPlane + rl * kSPitch + key * 16);
#endif
          const __half2* x0 = reinterpret_cast<const __half2*>(&c0);
          const __half2* x1 = reinterpret_cast<const __half2*>(&c1);
          float f_ar, f_ad, f_ag;
          rdg_dot(x0, x1, wr, wd, wg, f_ar, f_ad, f_ag);
          const float nb = b ? nr.y : nr.x;
          if (walk && ((nz >> k) & 1u)) {  // the rare lane whose bits did not fit the list
            unsigned long long m = load_edge1(edges, ((size_t)sample * 64 + row0 + rl) * 64 + key);
            float e = 0.f;
            while (m) {
              const int c = __ffsll(static_cast<long long>(m)) - 1;
              m &= m - 1;
              e += __half2float(sR[rl * kTab + c]) + __half2float(sC[key * kTab + c]) + SIGMA_TERM(nb, sSig[c]);
            }
            hv[n][2 * a + b] = e;
          } else {
            const int cnt = static_cast<int>((counts >> (4 * k)) & 15u);
#ifdef DBG_NO_E
            hv[n][2 * a + b] = 0.f;
#else
            hv[n][2 * a + b] = cnt ? sP[cell_start + cnt - 1] - (cell_start ? sP[cell_start - 1] : 0.f) : 0.f;
#endif
            cell_start += cnt;
          }
          hv[n][2 * a + b] += elem(s[n], 2 * a + b) * qk_scale + nb * (b ? cbf.y : cbf.x) + f_ar + (b ? bsf.y : bsf.x);
#ifdef DOOR_F32  // k1: the door applied here in FP32 (not via its FP16 copy in dg below)
          hv[n][2 * a + b] *= 1.f + f_ad;
#endif
          const float x = f_ag + gate_b;
#if CAP
          dg[n][2 * a + b] = __floats2half2_rn(f_ad, x >= 0.f ? 1.f : two_sigmoid(x));
#else
          dg[n][2 * a + b] = __floats2half2_rn(f_ad, two_sigmoid(x));
#endif
        }
      }
    }
    }
#pragma unroll
    for (int a = 0; a < 2; ++a) {
      const int rl = rt * 16 + g + 8 * a;
#pragma unroll
      for (int n = 0; n < 4; ++n) {
#pragma unroll
#ifndef DOOR_F32
        for (int b = 0; b < 2; ++b) hv[n][2 * a + b] *= 1.f + __low2float(dg[n][2 * a + b]);
#endif
#if EXPORT_H && !H_F16
        *reinterpret_cast<float2*>(hexp + (((size_t)sample * kHeads + h) * 64 + row0 + rl) * 64 + kh * 32 + 8 * n + 2 * t) =
            make_float2(hv[n][2 * a], hv[n][2 * a + 1]);
#endif
      }
#if EXPORT_H && H_F16
      if (hexp != nullptr) {  // FP16 H (k1: only when the caller asks for it): a 4 x 4 transpose across the quad (lane t, register n) -> (lane n, register t) gives lane t the key half's
         // n-tile t, 8 consecutive keys stored as one 16-byte write (a row's 32 keys = 64 contiguous bytes per instruction)
        uint32_t p[4];
#pragma unroll
        for (int n = 0; n < 4; ++n) p[n] = pack(hv[n][2 * a], hv[n][2 * a + 1]);
        const bool b1 = t & 2, b0 = t & 1;
        uint32_t x0 = __shfl_xor_sync(0xffffffffu, b1 ? p[0] : p[2], 2);  // registers whose bit 1 differs from the lane's
        uint32_t x1 = __shfl_xor_sync(0xffffffffu, b1 ? p[1] : p[3], 2);
        if (b1) { p[0] = x0; p[1] = x1; } else { p[2] = x0; p[3] = x1; }
        x0 = __shfl_xor_sync(0xffffffffu, b0 ? p[0] : p[1], 1);  // then bit 0
        x1 = __shfl_xor_sync(0xffffffffu, b0 ? p[2] : p[3], 1);
        if (b0) { p[0] = x0; p[2] = x1; } else { p[1] = x0; p[3] = x1; }
        *reinterpret_cast<uint4*>(hexp + (((size_t)sample * kHeads + h) * 64 + row0 + rl) * 64 + kh * 32 + 8 * t) =
            make_uint4(p[0], p[1], p[2], p[3]);
      }
#endif
    }
    TICK(3);

    // ---- local softmax over this key half (quad shuffles only); the gate multiplies the unnormalised exponentials
    float mrow[2], zrow[2];
#pragma unroll
    for (int a = 0; a < 2; ++a) {
      float mx = fmaxf(fmaxf(hv[0][2 * a], hv[0][2 * a + 1]), fmaxf(hv[1][2 * a], hv[1][2 * a + 1]));
      mx = fmaxf(mx, fmaxf(fmaxf(hv[2][2 * a], hv[2][2 * a + 1]), fmaxf(hv[3][2 * a], hv[3][2 * a + 1])));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
      float z = 0.f;
#pragma unroll
      for (int n = 0; n < 4; ++n)
#pragma unroll
        for (int b = 0; b < 2; ++b) {
          const float p = __expf(hv[n][2 * a + b] - mx);
          z += p;
          hv[n][2 * a + b] = p * __high2float(dg[n][2 * a + b]);
        }
      z += __shfl_xor_sync(0xffffffffu, z, 1);
      z += __shfl_xor_sync(0xffffffffu, z, 2);
      mrow[a] = mx;
      zrow[a] = z;
    }

    // the output-gate lane and the prescale of the n-tiles this warp finalizes, ahead of the exchange
    const int nd0 = kh * kHalfNd;
    float2 gl[kHalfNd][2], ps[kHalfNd];
#pragma unroll
    for (int i = 0; i < kHalfNd; ++i) {
#if OUT_F16
      ps[i] = make_float2(1.f, 1.f);
#else
      ps[i] = __ldg(reinterpret_cast<const float2*>(prescale + h * kHd + 8 * (nd0 + i) + 2 * t));
#endif
#pragma unroll
      for (int a = 0; a < 2; ++a) {
#if OGATE
        gl[i][a] = __half22float2(__ldg(reinterpret_cast<const __half2*>(
            base + 3 * kWidth + (size_t)HOTI(row0 + rt * 16 + g + 8 * a, g) * kStride + HOTI(8 * (nd0 + i), 0) + 2 * t)));
#else
        gl[i][a] = make_float2(0.f, 0.f);
#endif
      }
    }
    // ---- P.V: P as A fragments from the accumulator layout; V fragments transposed in registers
    uint32_t pa[2][4];
#pragma unroll
    for (int kk = 0; kk < 2; ++kk) {
      pa[kk][0] = pack(hv[2 * kk][0], hv[2 * kk][1]);
      pa[kk][1] = pack(hv[2 * kk][2], hv[2 * kk][3]);
      pa[kk][2] = pack(hv[2 * kk + 1][0], hv[2 * kk + 1][1]);
      pa[kk][3] = pack(hv[2 * kk + 1][2], hv[2 * kk + 1][3]);
    }
    uint32_t vraw[kNd][2][2];
#pragma unroll
    for (int kk = 0; kk < 2; ++kk)
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        uint32_t w[4];
        quad_words(vload[kk][j], t, w);
#pragma unroll
        for (int nd = 0; nd < kNd; ++nd) vraw[nd][kk][j] = w[nd];
      }
    Acc o[kNd];
#pragma unroll
    for (int nd = 0; nd < kNd; ++nd) {
      zero(o[nd]);
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) mma(o[nd], pa[kk], transpose8x8(vraw[nd][kk][0]), transpose8x8(vraw[nd][kk][1]));
    }
    TICK(4);

    // ---- merge with the key-half partner through the head's region (its tables are dead once all 4 warps pass B2)
    named_bar(bar_head, 128);  // B2
    float* mine = sEx + (rt * 2 + kh) * kExWarp;
    const float* theirs = sEx + (rt * 2 + (1 - kh)) * kExWarp;
    if (t == 0) {
      mine[g] = mrow[0];
      mine[g + 8] = mrow[1];
      mine[16 + g] = zrow[0];
      mine[16 + g + 8] = zrow[1];
    }
#pragma unroll
    for (int i = 0; i < kHalfNd; ++i) {  // the partner finalizes n-tiles (1 - kh) * kHalfNd + i
      float v[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) v[j] = kh ? elem(o[i], j) : elem(o[kHalfNd + i], j);
      float* dst = mine + 32 + g * (kHd / 2) + 8 * i + 2 * t;
      *reinterpret_cast<float2*>(dst) = make_float2(v[0], v[1]);
      *reinterpret_cast<float2*>(dst + 8 * (kHd / 2)) = make_float2(v[2], v[3]);
    }
    named_bar(bar_pair, 64);  // B3
    TICK(5);
    float fm[2], fp[2], inv[2];
#pragma unroll
    for (int a = 0; a < 2; ++a) {
      const float mp = theirs[g + 8 * a], zp = theirs[16 + g + 8 * a];
      const float mm = fmaxf(mrow[a], mp);
      fm[a] = __expf(mrow[a] - mm);
      fp[a] = __expf(mp - mm);
      inv[a] = __frcp_rn(zrow[a] * fm[a] + zp * fp[a]);
    }
    constexpr float gate_scale = GATE_SCALE;
#pragma unroll
    for (int i = 0; i < kHalfNd; ++i) {
      float v[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) v[j] = kh ? elem(o[kHalfNd + i], j) : elem(o[i], j);
      const float* src = theirs + 32 + g * (kHd / 2) + 8 * i + 2 * t;
      const float2 p0 = *reinterpret_cast<const float2*>(src), p1 = *reinterpret_cast<const float2*>(src + 8 * (kHd / 2));
      const int d = h * kHd + 8 * (nd0 + i) + 2 * t;
#pragma unroll
      for (int a = 0; a < 2; ++a) {
        const int row = row0 + rt * 16 + g + 8 * a;
        float x0 = (v[2 * a] * fm[a] + (a ? p1.x : p0.x) * fp[a]) * inv[a];
        float x1 = (v[2 * a + 1] * fm[a] + (a ? p1.y : p0.y) * fp[a]) * inv[a];
#if OGATE
        x0 *= 0.5f * gate_scale * two_sigmoid(gl[i][a].x);  // gate_scale * sigmoid
        x1 *= 0.5f * gate_scale * two_sigmoid(gl[i][a].y);
#endif
        if constexpr (kI8) {
          char2 code;
          code.x = code_of(x0 * inv_out_scale);
          code.y = code_of(x1 * inv_out_scale);
          *reinterpret_cast<char2*>(out8 + ((size_t)sample * 64 + row) * kWidth + d) = code;
        } else {
          *reinterpret_cast<__half2*>(out + ((size_t)sample * 64 + row) * kWidth + d) = __floats2half2_rn(x0, x1);
        }
      }
    }
    TICK(6);
  }
#if defined(EGT_PHASE) && defined(BT6_ATTN_KERNEL_ONLY)
  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < 10; ++i) atomicAdd(&g_egt_ph[warp * 10 + i], static_cast<unsigned long long>(ph_acc[i]));
    atomicAdd(&g_egt_ph[160 + warp], static_cast<unsigned long long>(nq));
  }
#endif
#if defined(EGT_TIMING) && defined(TAIL_SPLIT)
  if (tid == 0) {
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_end));
    g_egt_t[3 * blockIdx.x] = t_start; g_egt_t[3 * blockIdx.x + 1] = t_end; g_egt_t[3 * blockIdx.x + 2] = nq | (nfresh << 16);
  }
#endif
}

#ifndef BT6_ATTN_KERNEL_ONLY
// ---------------------------------------------------------------------------------------------------------------------
// Template bias (AttnTemplates, attn_templates.AttnTemplateMixer): bias[b, h, i, j] = sum_t c[b, i, h, t] * table[t, i, j]
// c = the layer's mix output [B, 64, heads * T] (FP16, with bias), table [T, 64, 64] FP16 shared by every layer.
// One CTA per (b, i): FP32 sums, FP16 result in [B, heads, 64, 64] (the BIAS layout of the attention kernel).
template <int T>
__global__ void __launch_bounds__(256) bt6_template_bias(__half* __restrict__ bias, const __half* __restrict__ coef,
                                                         const __half* __restrict__ table, int heads, int cstride,
                                                         const __half* __restrict__ cbias) {
  const int bi = blockIdx.x, b = bi >> 6, i = bi & 63;
  __shared__ float sc[HEADS * T];
  __shared__ float st[T][64];
  // coef row stride `cstride`; cbias: the mixer bias when the coefficients come straight from a fused GEMM (else null)
  for (int k = threadIdx.x; k < heads * T; k += blockDim.x)
#ifdef COEF_CONTROL_NO_BIAS  // the fidelity gate's CONTROL (must fail): the fused mixer's bias dropped
    sc[k] = __half2float(coef[(size_t)bi * cstride + k]);
#else
    sc[k] = __half2float(coef[(size_t)bi * cstride + k]) + (cbias ? __half2float(cbias[k]) : 0.f);
#endif
  for (int k = threadIdx.x; k < T * 64; k += blockDim.x) st[k >> 6][k & 63] = __half2float(table[((k >> 6) * 64 + i) * 64 + (k & 63)]);
  __syncthreads();
  // each thread: one head and 2 consecutive keys per step (half2 stores)
  for (int k = threadIdx.x; k < heads * 32; k += blockDim.x) {
    const int h = k >> 5, j = 2 * (k & 31);
    float a0 = 0.f, a1 = 0.f;
#pragma unroll
    for (int tt = 0; tt < T; ++tt) {
      const float c = sc[h * T + tt];
      a0 += c * st[tt][j];
      a1 += c * st[tt][j + 1];
    }
    *reinterpret_cast<__half2*>(bias + (((size_t)b * heads + h) * 64 + i) * 64 + j) = __floats2half2_rn(a0, a1);
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Host side. Table blobs (built by the ONNX rewrite, tools/bt6_plugin_onnx.py):
//   tab16 FP16: query [H][40][HD] | key [H][40][HD] | read|door|gate [H][48] | constant bias [H][64][64]
//   tab32 FP32: qk_scale [H] | attack [H][34] | scaled [H][34] | gate_b [H] | qn_scale [HD] | kn_scale [HD]
namespace bt6 {

constexpr int kSmemBytes = kSmem;
constexpr int kThreadsPerCta = kThreads;
constexpr int kTab16Elems = 2 * kHeads * 40 * kHd + kHeads * 48 + kHeads * 4096;
constexpr int kTab32Elems = kHeads + 2 * kHeads * 34 + kHeads + 2 * kHd;

int prepare_attention() {
  const cudaError_t r[4] = {
      cudaFuncSetAttribute(terms_v9_attention<kStride, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem),
      cudaFuncSetAttribute(terms_v9_attention<kStride + kHeads * 8, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem),
      cudaFuncSetAttribute(terms_v9_attention<kStride, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem),
      cudaFuncSetAttribute(terms_v9_attention<kStride + kHeads * 8, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem)};
  for (cudaError_t e : r)
    if (e != cudaSuccess) return static_cast<int>(e);
  return 0;
}

int launch_attention(void* out, const __half* qkv, const __half* coef, const int* edges, const __half* norm,
                     const __half* state, const __half* tab16, const float* tab32, const __half* tpl, int templates,
                     __half* bias_ws, int batch, int sms, cudaStream_t stream, __half* hraw, int fused_coef,
                     float out_int8_scale) {
  if (batch <= 0) return 0;
  if (templates == 8) {
    // fused_coef: qkv is [B, 64, 3 * width + heads * 8] (the mixer columns after q | k | v), coef = the mixer bias
    if (fused_coef)
      bt6_template_bias<8><<<batch * 64, 256, 0, stream>>>(bias_ws, qkv + kStride, tpl, kHeads, kStride + kHeads * 8, coef);
    else
      bt6_template_bias<8><<<batch * 64, 256, 0, stream>>>(bias_ws, coef, tpl, kHeads, kHeads * 8, nullptr);
  } else {
    return 1;
  }
  const __half* q16 = tab16;
  const __half* k16 = q16 + kHeads * 40 * kHd;
  const __half* rdg = k16 + kHeads * 40 * kHd;
  const __half* cb = rdg + kHeads * 48;
  const float* qk_scale = tab32;
  const float* attack = qk_scale + kHeads;
  const float* scaled = attack + kHeads * 34;
  const float* gate_b = scaled + kHeads * 34;
  const float* qn = gate_b + kHeads;
  const float* kn = qn + kHd;
  const int halves = 2 * batch;
  const int units = halves * kGroups;
  const int grid = units < sms ? units : sms;
  const bool i8 = out_int8_scale > 0.f;
  const float inv = i8 ? 1.f / out_int8_scale : 1.f;
#define BT6_LAUNCH(STRIDE, I8)                                                                                          terms_v9_attention<STRIDE, I8><<<grid, kThreads, kSmem, stream>>>(                                                       out, nullptr, qkv, reinterpret_cast<const EdgeT*>(edges), norm, state, qk_scale, attack,                              reinterpret_cast<const float*>(k16), reinterpret_cast<const float*>(q16), scaled, reinterpret_cast<const float*>(cb),       reinterpret_cast<const float*>(rdg), nullptr, nullptr, gate_b, nullptr, bias_ws, qn, kn, halves, hraw,               0.17677669529663687f /* 1 / sqrt(head_dim 32) */, inv)
  if (fused_coef) {
    if (i8) BT6_LAUNCH(kStride + kHeads * 8, true); else BT6_LAUNCH(kStride + kHeads * 8, false);
  } else {
    if (i8) BT6_LAUNCH(kStride, true); else BT6_LAUNCH(kStride, false);
  }
#undef BT6_LAUNCH
  return static_cast<int>(cudaGetLastError());
}

}  // namespace bt6
static_assert(bt6::kTab16Elems == bt6::kPluginTab16 && bt6::kTab32Elems == bt6::kPluginTab32, "table blob sizes");
static_assert(kHeads == bt6::kPluginHeads && kHd == bt6::kPluginHeadDim, "compiled shape");
#endif  // BT6_ATTN_KERNEL_ONLY

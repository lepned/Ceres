// k1 (2026-11-25): BT6EgtAttention -- the EGT2 attention of Kovax's BT6 flagship (1024 wide, 32 heads x 32, lc0 "BT6"),
// the lab kernel of bt6_attn_kernels.cu compiled a second time with that net's switches:
//   HEADS 32, packed [Q | K | V | G] qkvg GEMM output (heads side by side per lane, QKV_INTERLEAVED 0), output gate
//   gate_scale * sigmoid(g) fused (OGATE 1, GATE_SCALE 2), no QK-norm, no template bias (BIAS 0), and the H export
//   H = (qk_scale q.k + E terms + S (sigma . E + cbias) + read . e) * (1 + door . e) in FP16 [B, heads, 64, 64]
//   (EXPORT_H 1, H_F16 1; written only when the caller passes an H buffer: the edge-site layers).
// Math = bt6_net.served_layer (the attention part), exact against the lc0ex egt kernels:
//   A = softmax(H) * 2 sigmoid(gate . e + gate_b), out = (A @ V) * 2 sigmoid(G).
// Tables (built by kovax_bt6_trt/plugin/bt6_k1_onnx.py from the carrier, FP32 -> FP16 once):
//   tab16 FP16: query [H][40][32] | key [H][40][32] | read|door|gate [H][48] | constant_bias [H][64][64]
//   tab32 FP32: qk_scale [H] | attack [H][34] | scaled_coefficients [H][34] | gate_b [H]
#define HEADS 32
#define HD 32
#define OGATE 1
#define GATE_SCALE 2.0f
#define QKV_INTERLEAVED 0
#define QKNORM 0
#define BIAS 0
#define EXPORT_H 1
#define H_F16 1
#define OUT_F16 1
#define BT6_ATTN_KERNEL egt32_attention
// numerics of this build (k1, gated on Kovax's net): the door applied in FP32 where it is computed (DOOR_F32; its
// FP16 copy cost the gate's worst position KL 1.5e-2 -> 1.9e-4); the read / door / gate dots stay the lab kernel's
// HFMA2 chains (FP32 dots: +25 % kernel time for no gate gain; two-chain split: no gain). Probe switches:
// EGT_NO_DOOR_F32, EGT_RDG_F32, EGT_RDG_SPLIT.
#ifndef EGT_NO_DOOR_F32
#define DOOR_F32
#endif
#if defined(EGT_RDG_F32)
#define RDG_F32
#elif defined(EGT_RDG_SPLIT)
#define RDG_SPLIT
#endif
// k2 probe switches: EGT_GROUP (heads per CTA, default 4) and EGT_CTAS (resident CTAs per SM the grid assumes, default 1)
#ifdef EGT_GROUP
#define GROUP EGT_GROUP
#endif
#ifndef EGT_CTAS
#define EGT_CTAS 1
#endif
// k3: TAIL_SPLIT -- the last, mostly empty round of the persistent work list cut into 1- or 2-head pieces (kernel
// argument split; env BT6_EGT_SPLIT=0 restores k2's schedule). Numerics unchanged.
#define TAIL_SPLIT
// k5 (all four bit-identical to k4; NOTES_k5). The kernel is compiled in both forms (template flag kNew); env BT6_EGT_K4=1
// launches k4's form and schedule. Build switches drop single changes:
//   STATE_LDG: the half-position's state copied with plain 16-byte loads + shared stores instead of cp.async
//     (-DEGT_STATE_CPASYNC restores cp.async);
//   FRESH_SHARE: only head slot 0 loads the half-position's edge bits / S; the other slots read its per-lane results
//     from shared memory (-DEGT_NO_FRESH_SHARE restores the four loads);
//   SKEW: lc0ex's head-slot skew, odd slots start EGT_SKEW cycles late after a fresh block (default 4500; -DEGT_SKEW=0 off);
//   HP_SCHED: half-position-aligned remainder schedule (kernel argument sched; env BT6_EGT_SCHED=0 restores k4's
//     schedule, =m forces m parts per half-position where valid).
#define K5_VARIANT
#ifndef EGT_STATE_CPASYNC
#define STATE_LDG 4
#endif
#ifndef EGT_NO_FRESH_SHARE
#define FRESH_SHARE
#endif
#define HP_SCHED
#ifndef EGT_SKEW
#define EGT_SKEW 4500
#endif
#if EGT_SKEW > 0
#define SKEW EGT_SKEW
#endif
#ifdef EGT_TIMING  // debug build only: per-CTA start / end (globaltimer) and unit / fresh counts, printed by the host
__device__ unsigned long long g_egt_t[3 * 4096];
#include <cstdio>
#include <vector>
#include <algorithm>
#endif
#ifdef EGT_PHASE  // k5 debug build only: per-warp phase clocks
__device__ unsigned long long g_egt_ph[16 * 10 + 16];
#include <cstdio>
#endif
#define BT6_ATTN_KERNEL_ONLY
#include "bt6_attn_kernels.cu"
#include <cstdlib>

namespace bt6 {

constexpr int kEgtTab16Elems = 2 * kHeads * 40 * kHd + kHeads * 48 + kHeads * 4096;
constexpr int kEgtTab32Elems = kHeads + 2 * kHeads * 34 + kHeads;
static_assert(kEgtTab16Elems == kEgtTab16 && kEgtTab32Elems == kEgtTab32, "egt table blob sizes");
static_assert(kHeads == kEgtHeads && kHd == kEgtHeadDim && kStride == kEgtQkvgWidth, "egt compiled shape");

// k5: parts per remainder half-position (0 = k4's schedule). m0 = CTAs available per remainder half-position.
// Measured on the 5090 (170 SMs, NOTES_k5): m0 = 5 (b16) -19 % kernel span, m0 = 21 (b4) -14 %, m0 = 32 (b1) -23 %;
// m0 = 10 (b8) +5 %, m0 = 2 (b32) +5 %, m0 = 1 (b64) +9 %, b128 (m = 2 with leftover pieces) 0 %.
// After a full round it does not pay (b96: m = 7 +2.5 %, b192: m = 3 +1 %): used only when full == 0.
static int egt_auto_sched(int full, int rest, int slots) {
  if (rest <= 0 || full > 0) return 0;
  int m0 = slots / rest;
  if (m0 > kHeads) m0 = kHeads;
  return (m0 >= 3 && m0 <= 7) || m0 >= 16 ? m0 : 0;
}

int prepare_egt_attention() {
#ifdef BT6_WITH_T5
  (void)egt_t5_usable();  // t5's shared-memory attributes, outside any graph capture (GB300 only; no-op elsewhere)
#endif
  const cudaError_t r[8] = {
      cudaFuncSetAttribute(egt32_attention<kStride, false, false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem),
      cudaFuncSetAttribute(egt32_attention<kStride, true, false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem),
      cudaFuncSetAttribute(egt32_attention<kStride, false, true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem),
      cudaFuncSetAttribute(egt32_attention<kStride, true, true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem),
      cudaFuncSetAttribute(egt32_attention<kStride, false, false, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem),
      cudaFuncSetAttribute(egt32_attention<kStride, true, false, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem),
      cudaFuncSetAttribute(egt32_attention<kStride, false, true, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem),
      cudaFuncSetAttribute(egt32_attention<kStride, true, true, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem)};
  for (cudaError_t e : r)
    if (e != cudaSuccess) return static_cast<int>(e);
  return 0;
}

int launch_egt_attention(void* out, const __half* qkvg, const int* edges, const __half* norm, const __half* state,
                         const __half* tab16, const float* tab32, int batch, int sms, cudaStream_t stream, __half* h_out,
                         float out_int8_scale) {
  if (batch <= 0) return 0;
#ifdef BT6_WITH_T5
  // GB300 (sm_103): the tcgen05 kernel (bt6_egt_t5.cu) for FP16 output; env BT6_EGT_T5=0 keeps v9
  if (out_int8_scale <= 0.f && egt_t5_usable())
    return launch_egt_t5(out, qkvg, edges, norm, state, tab16, tab32, batch, sms, stream, h_out);
#endif
  const __half* q16 = tab16;
  const __half* k16 = q16 + kHeads * 40 * kHd;
  const __half* rdg = k16 + kHeads * 40 * kHd;
  const __half* cb = rdg + kHeads * 48;
  const float* qk_scale = tab32;
  const float* attack = qk_scale + kHeads;
  const float* scaled = attack + kHeads * 34;
  const float* gate_b = scaled + kHeads * 34;
  const int halves = 2 * batch;
  const int units = halves * kGroups;
  const int slots = sms * EGT_CTAS;
  // k3 tail split (TAIL_SPLIT, kSplit = true) when the leftover round can be cut into 1- or 2-head pieces AND the CTAs
  // have at least kSplitMinUnits units each (measured, egt timing build: b128 -3.8 %, b256 -1 %, b64 +1.5 %); else k2's
  // schedule and kernel. env BT6_EGT_SPLIT=0 never splits, =2 splits whenever the pieces fit.
  static const int split = [] { const char* e = std::getenv("BT6_EGT_SPLIT"); return !e ? 1 : e[0] == '0' ? 0 : e[0] == '2' ? 2 : 1; }();
  constexpr int kSplitMinUnits = 12;
  int grid = units < slots ? units : slots;
  bool use_split = false;
  if (split) {
    const int gs = units * kGroup < slots ? units * kGroup : slots;
    const int full = halves / gs, rest = halves - full * gs, runits = rest * kGroups, R2 = runits / gs, E = runits - R2 * gs;
    const bool fits = E > 0 && (E * kGroup <= gs || E * 2 <= gs);
    if (fits && (split == 2 || full * kGroups + R2 >= kSplitMinUnits)) {
      use_split = true;
      grid = gs;
    }
  }
  // k5 half-position-aligned remainder: m CTAs per remainder half-position (see the kernel). Auto rule = measured
  // (NOTES_k5): used when the remainder's contiguous slices are short enough that a second state load matters.
  static const bool k4_path = [] { const char* e = std::getenv("BT6_EGT_K4"); return e && e[0] == '1'; }();
  static const int sched_env = [] { const char* e = std::getenv("BT6_EGT_SCHED"); return k4_path ? 0 : e ? std::atoi(e) : -1; }();
  int sched = 0;
  {
    const int full = halves / slots, rest = halves - full * slots;
    auto valid = [&](int m) {
      if (m < 1 || m > kHeads || rest == 0) return false;
      const int parts = m * rest;
      if (parts <= slots) return true;
      return kGroups % m == 0 && (parts - slots) * (kGroups / m) * kGroup <= slots;
    };
    if (sched_env > 0) {
      if (valid(sched_env)) sched = sched_env;
    } else if (sched_env < 0) {
      sched = egt_auto_sched(full, rest, slots);
      if (!valid(sched)) sched = 0;
    }
    if (sched) {
      use_split = true;
      grid = slots;
    }
  }
  const bool i8 = out_int8_scale > 0.f;
  const float inv = i8 ? 1.f / out_int8_scale : 1.f;
#define EGT_LAUNCH(I8, SP, NEW)                                                                                            \
  egt32_attention<kStride, I8, SP, NEW><<<grid, kThreads, kSmem, stream>>>(                                                \
      out, h_out, qkvg, reinterpret_cast<const EdgeT*>(edges), norm, state, qk_scale, attack,                       \
      reinterpret_cast<const float*>(k16), reinterpret_cast<const float*>(q16), scaled, reinterpret_cast<const float*>(cb), \
      reinterpret_cast<const float*>(rdg), nullptr, nullptr, gate_b, nullptr, nullptr, nullptr, nullptr, halves, nullptr, \
      1.f, inv, sched)
  if (k4_path) {
    if (use_split) { if (i8) EGT_LAUNCH(true, true, false); else EGT_LAUNCH(false, true, false); }
    else { if (i8) EGT_LAUNCH(true, false, false); else EGT_LAUNCH(false, false, false); }
  } else {
    if (use_split) { if (i8) EGT_LAUNCH(true, true, true); else EGT_LAUNCH(false, true, true); }
    else { if (i8) EGT_LAUNCH(true, false, true); else EGT_LAUNCH(false, false, true); }
  }
#ifdef EGT_TIMING
  {
    std::vector<unsigned long long> v(3 * grid);
    cudaStreamSynchronize(stream);
    cudaMemcpyFromSymbol(v.data(), g_egt_t, sizeof(unsigned long long) * 3 * grid);
    unsigned long long t0 = ~0ull, t1 = 0;
    std::vector<double> d(grid);
    for (int b = 0; b < grid; ++b) if (v[3 * b]) { t0 = std::min(t0, v[3 * b]); t1 = std::max(t1, v[3 * b + 1]); }  // CTAs without work write nothing
    for (int b = 0; b < grid; ++b) d[b] = v[3 * b] ? (v[3 * b + 1] - t0) * 1e-3 : 0.;
    { static const std::vector<unsigned long long> z(3 * 4096, 0ull); cudaMemcpyToSymbol(g_egt_t, z.data(), sizeof(unsigned long long) * 3 * 4096); }
    std::vector<int> idx(grid);
    for (int b = 0; b < grid; ++b) idx[b] = b;
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return d[a] > d[b]; });
    std::vector<double> ds(d); std::sort(ds.begin(), ds.end());
    fprintf(stderr, "EGT B=%d grid=%d sched=%d split=%d/%d span %.1f us; end p10 %.1f p50 %.1f p90 %.1f; slowest:", batch, grid, sched, split, (int)use_split,
            (t1 - t0) * 1e-3, ds[grid / 10], ds[grid / 2], ds[grid * 9 / 10]);
    for (int k = 0; k < 6 && k < grid; ++k) {
      const int b = idx[k];
      fprintf(stderr, " [%d nq%d f%d st%.1f end%.1f]", b, (int)(v[3 * b + 2] & 0xffff), (int)(v[3 * b + 2] >> 16),
              (v[3 * b] - t0) * 1e-3, d[b]);
    }
    fprintf(stderr, "\n");
  }
#endif
#ifdef EGT_PHASE
  {
    static const char* names[10] = {"start/fresh", "ld+mma", "tabwrite", "assembly+H", "softmax+PV", "exchange", "merge+out", "list", "Eprefix", "-"};
    unsigned long long v[176];
    cudaStreamSynchronize(stream);
    cudaMemcpyFromSymbol(v, g_egt_ph, sizeof(v));
    unsigned long long z[176] = {0};
    cudaMemcpyToSymbol(g_egt_ph, z, sizeof(z));
    double tot[10] = {0}, nq = 0, all = 0;
    for (int w = 0; w < 16; ++w) { nq += v[160 + w]; for (int i = 0; i < 10; ++i) tot[i] += v[w * 10 + i]; }
    for (int i = 0; i < 10; ++i) all += tot[i];
    fprintf(stderr, "EGTPH B=%d grid=%d cyc/warp-unit %.0f :", batch, grid, all / nq);
    for (int i : {0, 1, 7, 2, 8, 3, 4, 5, 6}) fprintf(stderr, " %s %.0f", names[i], tot[i] / nq);
    fprintf(stderr, "\n");
  }
#endif
#undef EGT_LAUNCH
  return static_cast<int>(cudaGetLastError());
}

}  // namespace bt6

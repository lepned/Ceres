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
#define BT6_ATTN_KERNEL_ONLY
#include "bt6_attn_kernels.cu"

namespace bt6 {

constexpr int kEgtTab16Elems = 2 * kHeads * 40 * kHd + kHeads * 48 + kHeads * 4096;
constexpr int kEgtTab32Elems = kHeads + 2 * kHeads * 34 + kHeads;
static_assert(kEgtTab16Elems == kEgtTab16 && kEgtTab32Elems == kEgtTab32, "egt table blob sizes");
static_assert(kHeads == kEgtHeads && kHd == kEgtHeadDim && kStride == kEgtQkvgWidth, "egt compiled shape");

int prepare_egt_attention() {
  const cudaError_t r[2] = {
      cudaFuncSetAttribute(egt32_attention<kStride, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem),
      cudaFuncSetAttribute(egt32_attention<kStride, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem)};
  for (cudaError_t e : r)
    if (e != cudaSuccess) return static_cast<int>(e);
  return 0;
}

int launch_egt_attention(void* out, const __half* qkvg, const int* edges, const __half* norm, const __half* state,
                         const __half* tab16, const float* tab32, int batch, int sms, cudaStream_t stream, __half* h_out,
                         float out_int8_scale) {
  if (batch <= 0) return 0;
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
  const int grid = units < slots ? units : slots;
  const bool i8 = out_int8_scale > 0.f;
  const float inv = i8 ? 1.f / out_int8_scale : 1.f;
#define EGT_LAUNCH(I8)                                                                                             \
  egt32_attention<kStride, I8><<<grid, kThreads, kSmem, stream>>>(                                                  \
      out, h_out, qkvg, reinterpret_cast<const EdgeT*>(edges), norm, state, qk_scale, attack,                       \
      reinterpret_cast<const float*>(k16), reinterpret_cast<const float*>(q16), scaled, reinterpret_cast<const float*>(cb), \
      reinterpret_cast<const float*>(rdg), nullptr, nullptr, gate_b, nullptr, nullptr, nullptr, nullptr, halves, nullptr, \
      1.f, inv)
  if (i8) EGT_LAUNCH(true); else EGT_LAUNCH(false);
#undef EGT_LAUNCH
  return static_cast<int>(cudaGetLastError());
}

}  // namespace bt6

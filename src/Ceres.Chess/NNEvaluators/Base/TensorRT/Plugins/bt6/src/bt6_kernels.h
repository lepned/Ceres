// Host entry points of the round-4 plugin kernels (round 3 + move-token decoder attention / head, bt6_dec_kernels.cu)
// (round 2 + INT8 attention output) (bt6_attn_kernels.cu, bt6_prologue_kernels.cu).
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace bt6 {

// 24 heads x 32, 8 templates: the shapes the kernels are compiled for (checked by the plugin against its attributes).
constexpr int kPluginHeads = 24;
constexpr int kPluginHeadDim = 32;
constexpr int kPluginTab16 = 2 * 24 * 40 * 32 + 24 * 48 + 24 * 4096;
constexpr int kPluginTab32 = 24 + 2 * 24 * 34 + 24 + 2 * 32;
// BT6Prologue table blob: D [34][16] | Pin [34][16] | T [225][16] | Toff [225][16] (FP32)
constexpr int kPrologueTab32 = 2 * 34 * 16 + 2 * 225 * 16;

// BT6EdgeSite table blob: R [16][24] | W1 [64][16] | Wr [64][16] | b1 [64] | W2 [16][64] (FP32)
constexpr int kSiteTab32 = 16 * 24 + 2 * 64 * 16 + 64 + 16 * 64;

int prepare_attention();
// hraw: nullptr, or FP16 [B, heads, 64, 64] for the raw logits q.k / sqrt(hd) + template bias (edge-site layers)
// fused_coef: qkv = [B, 64, 3 * width + heads * 8] (q | k | v | template-mixer columns of one GEMM), coef = mixer bias [heads * 8]
// out_int8_scale > 0 (round 3, v3 out_int8): out is INT8 [B, 64, H * hd], codes round(x / out_int8_scale); else FP16
int launch_attention(void* out, const __half* qkv, const __half* coef, const int* edges, const __half* norm,
                     const __half* state, const __half* tab16, const float* tab32, const __half* tpl, int templates,
                     __half* bias_ws, int batch, int sms, cudaStream_t stream, __half* hraw, int fused_coef,
                     float out_int8_scale);
int launch_prologue(int* edges, __half* norm, __half* state, const __half* squares, const float* tab, int batch,
                    int sms, cudaStream_t stream);
// BT6GluQuant (round 3): FFN width of the SwiGLU + per-token INT8 quantization kernel (bt6_glu_kernels.cu)
constexpr int kGluChannels = 1536;
int launch_glu_quant(int8_t* codes, __half* rmax, const int8_t* qg, const int8_t* qu, const __half* bias, float gate_scale,
                     float up_scale, int rows, cudaStream_t stream);
int launch_site(__half* out, const __half* state, const __half* hraw, const float* tab, int batch, cudaStream_t stream);

// Round 4 (bt6_dec_kernels.cu): move-token decoder, 80 tokens, dm 256, 8 heads x 32.
// mode 0 self (q = qkv[:, :, 0:256], k at k_off, v at v_off of the same [B, 80, q_stride] tensor, kbias FP16 [B, 80]),
// mode 1 cross / mode 2 post-move (q [B, 80, q_stride]; k, v from kv [B, 64, kv_stride] at k_off / v_off;
// mode 2: idx INT32 [B, 80, 3] = from, to, mover; tab FP16 [2][7][256] = dk | dv).
int launch_dec_attn(__half* out, const __half* q, int q_stride, const __half* kv, int kv_stride, int k_off, int v_off,
                    const __half* kbias, const int* idx, const __half* tab, float scale, int mode, int batch,
                    cudaStream_t stream);
// BT6DecHead: x FP16 [B, 80, 256] (residual after the last block), kbias FP16 [B, 80], sel INT32 [B, 80],
// tab FP32 [256 + 4 * 256 + 1858] (out_ln scale | pol W | per-move bias), slot INT32 [1858]
// -> policy FP16 [B, 1858], pooled FP16 [B, 512] (mean | max over valid tokens).
constexpr int kDecHeadTab32 = 256 + 4 * 256 + 1858;
int launch_dec_head(__half* policy, __half* pooled, const __half* x, const __half* kbias, const int* sel, const float* tab,
                    const int* slot, float eps, int batch, cudaStream_t stream);

}  // namespace bt6

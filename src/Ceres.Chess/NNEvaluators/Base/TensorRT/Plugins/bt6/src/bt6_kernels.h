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
// k2: BT6RowQuant / BT6SigGluQuant (bt6_rowquant_kernels.cu): per-token INT8 codes + row scale, 1024 wide (glu = 1:
// input [rows, 2048] = [gate | up], m = sigmoid(gate) * up)
constexpr int kRowQuantChannels = 1024;
int launch_row_quant(int8_t* codes, __half* rmax, const __half* x, int glu, float eps, int rows, cudaStream_t stream);
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

// ---- k1 (2026-11-25): Kovax's BT6 flagship (1024 x 24, 32 heads x 32, lc0 planes input) ------------------------------
// BT6EgtAttention (bt6_egt_attn.cu): qkvg FP16 [B, 64, 4096] = [Q | K | V | G] (heads side by side in each lane),
// edges INT32 [B, 64, 64, 2], norm S FP16 [B, 64, 64], state FP16 [B, 64, 64, 16] cell-major, tab16 / tab32
// -> out FP16 [B, 64, 1024] = (A @ V) * 2 sigmoid(G) (or INT8 codes with out_int8_scale > 0), and, when h_out is
// not null, H FP16 [B, 32, 64, 64] (the door-scaled logits the edge sites read).
constexpr int kEgtHeads = 32;
constexpr int kEgtHeadDim = 32;
constexpr int kEgtQkvgWidth = 4 * 32 * 32;
constexpr int kEgtTab16 = 2 * 32 * 40 * 32 + 32 * 48 + 32 * 4096;
constexpr int kEgtTab32 = 32 + 2 * 32 * 34 + 32;
int prepare_egt_attention();
int launch_egt_attention(void* out, const __half* qkvg, const int* edges, const __half* norm, const __half* state,
                         const __half* tab16, const float* tab32, int batch, int sms, cudaStream_t stream, __half* h_out,
                         float out_int8_scale);
// t5 (bt6_egt_t5.cu, built with BT6_WITH_T5 for sm_103a): the tcgen05 form of BT6EgtAttention for the GB300, FP16 output
// only; same arguments as launch_egt_attention. egt_t5_usable(): compute capability 10.3 and env BT6_EGT_T5 != 0.
bool egt_t5_supported();  // compute capability 10.3 and the kernel attributes set (no env check)
bool egt_t5_usable();
int launch_egt_t5(void* out, const __half* qkvg, const int* edges, const __half* norm, const __half* state,
                  const __half* tab16, const float* tab32, int batch, int sms, cudaStream_t stream, __half* h_out);
// BT6PlanesPrologue (bt6_prologue_kernels.cu, kPlanes): planes FP32 [B, 112, 64] + tab FP32 [kPlanesPrologueTab32]
// (the BT6Prologue blob | policy psc [34] | policy pcb [4096]) -> edges, S, e0 (as BT6Prologue) + pol FP32 [B, 64, 64]
// (+ e0f FP32 [B, 16, 64, 64] channels-first when not null).
constexpr int kPlanesPrologueTab32 = kPrologueTab32 + 34 + 4096;
int launch_prologue_planes(int* edges, __half* norm, __half* state, float* pol, const float* planes, const float* tab,
                           int batch, int sms, float eps, cudaStream_t stream, float* e0f = nullptr);

// k3: BT6LnQuant (bt6_lnquant_kernels.cu): res_out = res + alpha (yn r + bias) (FP16),
// codes = saturate(rint(LN(res_out) / scale)); tab FP32 [3 * 1024] = bias | gamma | beta; r may be null (row scale 1).
int launch_ln_quant(__half* res_out, int8_t* codes, const __half* yn, const __half* r, const __half* res, const float* tab,
                    float alpha, float eps, float scale, int rows, cudaStream_t stream);

// k4: BT6I8GemmGlu (bt6_i8gemm_kernels.cu): out INT8 [M, N / 2] = GLU-int8 epilogue of a [M, K] x [N, K]^T int8 GEMM;
// tab FP32 = scale [N] | bias [N] | r [N / 2]; tile -1 = env BT6_I8GLU_TILE or the default by M.
int i8gemm_glu_tiles();
int launch_i8gemm_glu(int8_t* out, const int8_t* a, const int8_t* w, const float* tab, int M, int N, int K, int tile,
                      cudaStream_t stream);
// Hopper form of BT6I8GemmGlu (bt6_i8gemm_sm90.cu, built with BT6_WITH_SM90GLU for sm_90a): same arguments and codes;
// cfg < 0 = env BT6_I8GLU_SM90_CFG or the default by M. launch_i8gemm_glu calls it on compute capability 9.0.
bool i8gemm_glu_sm90_supported();
int i8gemm_glu_sm90_cfgs();
int launch_i8gemm_glu_sm90(int8_t* out, const int8_t* a, const int8_t* w, const float* tab, int M, int N, int K, int cfg,
                           cudaStream_t stream);

}  // namespace bt6

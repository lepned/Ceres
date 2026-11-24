// Round 4: move-token decoder kernels (BT6DecAttn, BT6DecHead).
//
// The decoder of the 768x16 flat BT6 net (move_tokens.py, MoveTokenDecoder with post_move in every block): 4 blocks,
// dm 256, 8 heads x 32, M = 80 move tokens per position. Each block has three attentions, each between two GEMMs that
// stay in TensorRT:
//   self   q, k, v from the block's qkv GEMM [B, 80, 768] (q | k | v, head h at +32h), additive key bias [B, 80]
//          (-1e4 on invalid tokens, as the exported graph), keys = the 80 tokens.
//   cross  q = xq GEMM [B, 80, 256]; k, v = the block's columns of the shared square K|V GEMM [B, 64, 4096]
//          (k at koff + 32h, v at koff + 256 + 32h), keys = the 64 squares.
//   pm     post-move: q = pm_q GEMM; k, v from the same K|V GEMM (koff = 2048 + 512 i); the to-square key / value get the
//          mover's delta dk[mover], dv[mover] and the from-square the 'empty' delta dk[0], dv[0] (one-hot edits):
//          s_tj = q.k_j + [j == to] q.dk[mover] + [j == from] q.dk[0];  o = sum_j a_j v_j + a_to dv[mover] + a_fr dv[0]
// Control builds (must fail the gate): DEC_CONTROL_HALF_OUT halves head 0 of every attention output,
// DEC_CONTROL_NO_DV drops the post-move value edits (a_to dv[mover] + a_fr dv[0]).
// One CTA per (position, head), 5 warps (one 16-query tile each), mma.sync m16n8k16 FP16 inputs with FP32 accumulation,
// FP32 softmax, P normalised before it is rounded to FP16 for P V. Output o FP16 [B, 80, 256] (heads side by side), the
// input of the block's projection GEMM.
//
// BT6DecHead: after the last block: out_ln (RMSNorm) -> xo; policy = scatter of the 4 slot logits per token
// (pol 256 -> 4, floor -30 on invalid / absent tokens) through the [1858] pair*4+slot table + per-move bias; pooled =
// [mean over valid tokens | max over valid tokens (0 when none)] of xo, FP16 [B, 512]. One CTA per position.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "bt6_kernels.h"

namespace bt6 {
namespace {

constexpr int kQ = 80;      // move tokens per position
constexpr int kHd = 32;     // head dim
constexpr int kHeads = 8;
constexpr int kDm = 256;
constexpr int kLdK = 40;    // smem row pitch (halves) of K [key][32]: conflict-free 32-bit fragment loads

__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ uint32_t pack_half2(float lo, float hi) {
  __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ uint32_t ld32(const __half* p) { return *reinterpret_cast<const uint32_t*>(p); }

// MODE 0 self (NK = 80), 1 cross (NK = 64), 2 post-move (NK = 64).
template <int MODE, int NK>
__global__ void __launch_bounds__(160) dec_attn(__half* __restrict__ out, const __half* __restrict__ qsrc, int q_stride,
                                                const __half* __restrict__ kvsrc, int kv_stride, int k_off, int v_off,
                                                const __half* __restrict__ kbias, const int* __restrict__ idx,
                                                const __half* __restrict__ tab, float scale) {
  constexpr int NT = NK / 8;       // n8 tiles of S
  constexpr int LDV = NK + 8;      // pitch of V^T [32][NK] (halves)
  __shared__ __align__(16) __half sK[NK * kLdK];
  __shared__ __align__(16) __half sVt[kHd * LDV];
  const int h = blockIdx.x, b = blockIdx.y;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t = lane & 3;
  const int kv_rows = (MODE == 0) ? kQ : 64;
  const __half* kvb = kvsrc + static_cast<size_t>(b) * kv_rows * kv_stride;
  // K and V of this head: NK rows x 32 halves each = 4 chunks of 16 B per row
  for (int c = tid; c < NK * 4; c += 160) {
    const int r = c >> 2, q4 = c & 3;
    const uint4 kk = *reinterpret_cast<const uint4*>(kvb + static_cast<size_t>(r) * kv_stride + k_off + h * kHd + q4 * 8);
    *reinterpret_cast<uint4*>(&sK[r * kLdK + q4 * 8]) = kk;
    const uint4 vv = *reinterpret_cast<const uint4*>(kvb + static_cast<size_t>(r) * kv_stride + v_off + h * kHd + q4 * 8);
    const __half* vh = reinterpret_cast<const __half*>(&vv);
#pragma unroll
    for (int e = 0; e < 8; ++e) sVt[(q4 * 8 + e) * LDV + r] = vh[e];
  }
  // Q fragments of this warp's 16 rows (rows r0 = 16 warp + g, r1 = r0 + 8), straight from global
  const int r0 = warp * 16 + g, r1 = r0 + 8;
  const __half* qb = qsrc + static_cast<size_t>(b) * kQ * q_stride + h * kHd;
  const __half* q0 = qb + static_cast<size_t>(r0) * q_stride;
  const __half* q1 = qb + static_cast<size_t>(r1) * q_stride;
  uint32_t qa[2][4];
#pragma unroll
  for (int ks = 0; ks < 2; ++ks) {
    qa[ks][0] = ld32(q0 + ks * 16 + 2 * t);
    qa[ks][1] = ld32(q1 + ks * 16 + 2 * t);
    qa[ks][2] = ld32(q0 + ks * 16 + 8 + 2 * t);
    qa[ks][3] = ld32(q1 + ks * 16 + 8 + 2 * t);
  }
  // post-move: c_to = scale q.dk[mover], c_fr = scale q.dk[0] per row (quad reduction over the A fragments)
  int to0 = -1, to1 = -1, fr0 = -1, fr1 = -1, mv0 = 0, mv1 = 0;
  float cto0 = 0.f, cto1 = 0.f, cfr0 = 0.f, cfr1 = 0.f;
  if (MODE == 2) {
    const int* ib = idx + static_cast<size_t>(b) * kQ * 3;
    fr0 = ib[r0 * 3 + 0]; to0 = ib[r0 * 3 + 1]; mv0 = ib[r0 * 3 + 2];
    fr1 = ib[r1 * 3 + 0]; to1 = ib[r1 * 3 + 1]; mv1 = ib[r1 * 3 + 2];
    const __half* dk = tab;              // [7][256]
    const __half* dkm0 = dk + mv0 * kDm + h * kHd;
    const __half* dkm1 = dk + mv1 * kDm + h * kHd;
    const __half* dkf = dk + h * kHd;
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
#pragma unroll
      for (int hi = 0; hi < 2; ++hi) {
        const int col = ks * 16 + hi * 8 + 2 * t;
        const float2 qr0 = __half22float2(*reinterpret_cast<const __half2*>(&qa[ks][hi * 2 + 0]));
        const float2 qr1 = __half22float2(*reinterpret_cast<const __half2*>(&qa[ks][hi * 2 + 1]));
        const float2 a0 = __half22float2(*reinterpret_cast<const __half2*>(dkm0 + col));
        const float2 a1 = __half22float2(*reinterpret_cast<const __half2*>(dkm1 + col));
        const float2 f = __half22float2(*reinterpret_cast<const __half2*>(dkf + col));
        cto0 += qr0.x * a0.x + qr0.y * a0.y;
        cto1 += qr1.x * a1.x + qr1.y * a1.y;
        cfr0 += qr0.x * f.x + qr0.y * f.y;
        cfr1 += qr1.x * f.x + qr1.y * f.y;
      }
    }
#pragma unroll
    for (int o = 1; o <= 2; o <<= 1) {
      cto0 += __shfl_xor_sync(0xffffffffu, cto0, o);
      cto1 += __shfl_xor_sync(0xffffffffu, cto1, o);
      cfr0 += __shfl_xor_sync(0xffffffffu, cfr0, o);
      cfr1 += __shfl_xor_sync(0xffffffffu, cfr1, o);
    }
    cto0 *= scale; cto1 *= scale; cfr0 *= scale; cfr1 *= scale;
  }
  __syncthreads();
  // S = Q K^T (FP32)
  float s[NT][4];
#pragma unroll
  for (int j = 0; j < NT; ++j) {
    s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.f;
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
      uint32_t bf[2];
      const __half* kr = &sK[(j * 8 + g) * kLdK + ks * 16 + 2 * t];
      bf[0] = ld32(kr);
      bf[1] = ld32(kr + 8);
      mma16816(s[j], qa[ks], bf);
    }
  }
  // scale, bias / edits, row max
  float m0 = -3.0e38f, m1 = -3.0e38f;
  const __half* kbb = (MODE == 0) ? kbias + static_cast<size_t>(b) * kQ : nullptr;
#pragma unroll
  for (int j = 0; j < NT; ++j) {
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      const int key = j * 8 + 2 * t + e;
      float v0 = s[j][e] * scale, v1 = s[j][2 + e] * scale;
      if (MODE == 0) {
        const float kb = __half2float(kbb[key]);
        v0 += kb; v1 += kb;
      }
      if (MODE == 2) {
        if (key == to0) v0 += cto0;
        if (key == fr0) v0 += cfr0;
        if (key == to1) v1 += cto1;
        if (key == fr1) v1 += cfr1;
      }
      s[j][e] = v0; s[j][2 + e] = v1;
      m0 = fmaxf(m0, v0); m1 = fmaxf(m1, v1);
    }
  }
#pragma unroll
  for (int o = 1; o <= 2; o <<= 1) {
    m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, o));
    m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, o));
  }
  float l0 = 0.f, l1 = 0.f;
#pragma unroll
  for (int j = 0; j < NT; ++j) {
    s[j][0] = __expf(s[j][0] - m0); s[j][1] = __expf(s[j][1] - m0);
    s[j][2] = __expf(s[j][2] - m1); s[j][3] = __expf(s[j][3] - m1);
    l0 += s[j][0] + s[j][1]; l1 += s[j][2] + s[j][3];
  }
#pragma unroll
  for (int o = 1; o <= 2; o <<= 1) {
    l0 += __shfl_xor_sync(0xffffffffu, l0, o);
    l1 += __shfl_xor_sync(0xffffffffu, l1, o);
  }
  const float i0 = 1.f / l0, i1 = 1.f / l1;
  float ato0 = 0.f, ato1 = 0.f, afr0 = 0.f, afr1 = 0.f;
#pragma unroll
  for (int j = 0; j < NT; ++j) {
    s[j][0] *= i0; s[j][1] *= i0; s[j][2] *= i1; s[j][3] *= i1;
    if (MODE == 2) {
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int key = j * 8 + 2 * t + e;
        if (key == to0) ato0 += s[j][e];
        if (key == fr0) afr0 += s[j][e];
        if (key == to1) ato1 += s[j][2 + e];
        if (key == fr1) afr1 += s[j][2 + e];
      }
    }
  }
  if (MODE == 2) {
#pragma unroll
    for (int o = 1; o <= 2; o <<= 1) {
      ato0 += __shfl_xor_sync(0xffffffffu, ato0, o);
      ato1 += __shfl_xor_sync(0xffffffffu, ato1, o);
      afr0 += __shfl_xor_sync(0xffffffffu, afr0, o);
      afr1 += __shfl_xor_sync(0xffffffffu, afr1, o);
    }
  }
  // O = P V
  float acc[4][4];
#pragma unroll
  for (int n = 0; n < 4; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f;
#pragma unroll
  for (int kk = 0; kk < NK / 16; ++kk) {
    uint32_t pa[4];
    pa[0] = pack_half2(s[2 * kk][0], s[2 * kk][1]);
    pa[1] = pack_half2(s[2 * kk][2], s[2 * kk][3]);
    pa[2] = pack_half2(s[2 * kk + 1][0], s[2 * kk + 1][1]);
    pa[3] = pack_half2(s[2 * kk + 1][2], s[2 * kk + 1][3]);
#pragma unroll
    for (int n = 0; n < 4; ++n) {
      uint32_t bf[2];
      const __half* vr = &sVt[(n * 8 + g) * LDV + kk * 16 + 2 * t];
      bf[0] = ld32(vr);
      bf[1] = ld32(vr + 8);
      mma16816(acc[n], pa, bf);
    }
  }
  __half* o0 = out + (static_cast<size_t>(b) * kQ + r0) * kDm + h * kHd;
  __half* o1 = out + (static_cast<size_t>(b) * kQ + r1) * kDm + h * kHd;
#pragma unroll
  for (int n = 0; n < 4; ++n) {
    const int col = n * 8 + 2 * t;
    float x0 = acc[n][0], x1 = acc[n][1], y0 = acc[n][2], y1 = acc[n][3];
#ifndef DEC_CONTROL_NO_DV
    if (MODE == 2) {
      const __half* dv = tab + 7 * kDm;   // [7][256]
      const float2 dm0 = __half22float2(*reinterpret_cast<const __half2*>(dv + mv0 * kDm + h * kHd + col));
      const float2 dm1 = __half22float2(*reinterpret_cast<const __half2*>(dv + mv1 * kDm + h * kHd + col));
      const float2 df = __half22float2(*reinterpret_cast<const __half2*>(dv + h * kHd + col));
      x0 += ato0 * dm0.x + afr0 * df.x; x1 += ato0 * dm0.y + afr0 * df.y;
      y0 += ato1 * dm1.x + afr1 * df.x; y1 += ato1 * dm1.y + afr1 * df.y;
    }
#endif
#ifdef DEC_CONTROL_HALF_OUT
    if (h == 0) { x0 *= 0.5f; x1 *= 0.5f; y0 *= 0.5f; y1 *= 0.5f; }
#endif
    *reinterpret_cast<uint32_t*>(o0 + col) = pack_half2(x0, x1);
    *reinterpret_cast<uint32_t*>(o1 + col) = pack_half2(y0, y1);
  }
}

// ------------------------------------------------------------------------------------------------------------ head
// tab FP32: out_ln scale [256] | pol W [4][256] | pol bias [1858]; slot INT32 [1858] = pair * 4 + slot.
__global__ void __launch_bounds__(256) dec_head(__half* __restrict__ policy, __half* __restrict__ pooled,
                                                const __half* __restrict__ x, const __half* __restrict__ kbias,
                                                const int* __restrict__ sel, const float* __restrict__ tab,
                                                const int* __restrict__ slot, float eps) {
  extern __shared__ __align__(16) unsigned char dsm[];
  __half* sX = reinterpret_cast<__half*>(dsm);                       // raw x [80][256]
  short* sMap = reinterpret_cast<short*>(dsm + kQ * kDm * 2);         // pair -> token
  float* sR = reinterpret_cast<float*>(dsm + kQ * kDm * 2 + 4096 * 2);  // rsqrt per token
  float* sValid = sR + kQ;
  float* sTok = sValid + kQ;                                          // [80][4]
  const int b = blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const __half* xb = x + static_cast<size_t>(b) * kQ * kDm;
  for (int i = tid; i < 4096; i += 256) sMap[i] = -1;
  if (tid < kQ) sValid[tid] = (__half2float(kbias[static_cast<size_t>(b) * kQ + tid]) > -1.f) ? 1.f : 0.f;
  const float* W = tab + kDm;
  // warp per token: RMSNorm (8 channels per lane) and the 4 slot logits
  for (int tk = warp; tk < kQ; tk += 8) {
    const uint4 raw = *reinterpret_cast<const uint4*>(xb + tk * kDm + lane * 8);
    *reinterpret_cast<uint4*>(sX + tk * kDm + lane * 8) = raw;
    const __half* hv = reinterpret_cast<const __half*>(&raw);
    float v[8], ss = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) { v[e] = __half2float(hv[e]); ss += v[e] * v[e]; }
#pragma unroll
    for (int o = 16; o >= 1; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    const float r = rsqrtf(ss * (1.f / kDm) + eps);
    float a[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int c = lane * 8 + e;
      const float xo = v[e] * r * tab[c];
#pragma unroll
      for (int k = 0; k < 4; ++k) a[k] += xo * W[k * kDm + c];
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) {
#pragma unroll
      for (int o = 16; o >= 1; o >>= 1) a[k] += __shfl_xor_sync(0xffffffffu, a[k], o);
    }
    if (lane == 0) sR[tk] = r;
    if (lane < 4) {
      float mine = a[0];
      if (lane == 1) mine = a[1];
      if (lane == 2) mine = a[2];
      if (lane == 3) mine = a[3];
      sTok[tk * 4 + lane] = mine;
    }
  }
  __syncthreads();
  if (tid < kQ) sMap[sel[static_cast<size_t>(b) * kQ + tid]] = static_cast<short>(tid);
  // pools: thread = channel, over the valid tokens of xo = x r scale
  {
    const int c = tid;
    const float sc = tab[c];
    float sum = 0.f, mx = -3.0e38f, n = 0.f;
    for (int tk = 0; tk < kQ; ++tk) {
      if (sValid[tk] > 0.5f) {
        const float v = __half2float(sX[tk * kDm + c]) * sR[tk] * sc;
        sum += v; mx = fmaxf(mx, v); n += 1.f;
      }
    }
    __half* pb = pooled + static_cast<size_t>(b) * 2 * kDm;
    pb[c] = __float2half_rn(sum / fmaxf(n, 1.f));
    pb[kDm + c] = __float2half_rn(n > 0.f ? mx : 0.f);
  }
  __syncthreads();
  const float* pbias = tab + kDm + 4 * kDm;
  __half* pol = policy + static_cast<size_t>(b) * 1858;
  for (int j = tid; j < 1858; j += 256) {
    const int ps = slot[j];
    const int tk = sMap[ps >> 2];
    const float v = (tk >= 0 && sValid[tk] > 0.5f ? sTok[tk * 4 + (ps & 3)] : -30.f) + pbias[j];
    pol[j] = __float2half_rn(v);
  }
}

constexpr int kHeadSmem = kQ * kDm * 2 + 4096 * 2 + kQ * 4 * 6;

}  // namespace

int launch_dec_attn(__half* out, const __half* q, int q_stride, const __half* kv, int kv_stride, int k_off, int v_off,
                    const __half* kbias, const int* idx, const __half* tab, float scale, int mode, int batch,
                    cudaStream_t stream) {
  if (batch <= 0) return 0;
  const dim3 grid(kHeads, batch);
  if (mode == 0) dec_attn<0, 80><<<grid, 160, 0, stream>>>(out, q, q_stride, kv, kv_stride, k_off, v_off, kbias, idx, tab, scale);
  else if (mode == 1) dec_attn<1, 64><<<grid, 160, 0, stream>>>(out, q, q_stride, kv, kv_stride, k_off, v_off, kbias, idx, tab, scale);
  else dec_attn<2, 64><<<grid, 160, 0, stream>>>(out, q, q_stride, kv, kv_stride, k_off, v_off, kbias, idx, tab, scale);
  return static_cast<int>(cudaGetLastError());
}

int launch_dec_head(__half* policy, __half* pooled, const __half* x, const __half* kbias, const int* sel, const float* tab,
                    const int* slot, float eps, int batch, cudaStream_t stream) {
  if (batch <= 0) return 0;
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(dec_head, cudaFuncAttributeMaxDynamicSharedMemorySize, kHeadSmem);
    attr = true;
  }
  dec_head<<<batch, 256, kHeadSmem, stream>>>(policy, pooled, x, kbias, sel, tab, slot, eps);
  return static_cast<int>(cudaGetLastError());
}

}  // namespace bt6

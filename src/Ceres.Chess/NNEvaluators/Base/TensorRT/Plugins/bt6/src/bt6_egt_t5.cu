// BT6EgtAttention on the GB300 (sm_103a): the "t5" EGT2 attention kernel, every contraction on the 5th-generation tensor cores
// (tcgen05), the softmax operand kept in Tensor Memory. Ported into the bt6 TensorRT plugin library from the lc0ex builder tree
// (lczero-triton-tcgen05 commit 5fcc749, bt4/kernels/egt_t5.cu release 4 with t5a.h and tmem_ldst.h).
// launch_egt_attention (bt6_egt_attn.cu) calls it on compute
// capability 10.3 for FP16 output; every other GPU, and the INT8-output form, keep the v9 kernel. Env BT6_EGT_T5=0 forces v9.
//
// Changes against release 4 (the math, the pipeline, the shared-memory and Tensor Memory layouts are unchanged):
//   * HALVES (2 x batch) is a RUNTIME value: it only ever sized the grid (one CTA per half-position and GPC head groups).
//   * GPC (head groups per CTA) and EXPORT_H are template parameters (8 instantiations); the E stage form follows GPC as
//     before (rank list for GPC 8, cell list otherwise). GPC per launch: the wave-count rule in pick_gpc below.
//   * edges: the plugin's INT32 pairs [B, 64, 64, 2] (bits 0-16 | 17-33, BT6PlanesPrologue's layout), recomposed on load,
//     instead of one uint64 per cell.
//   * the constant bias is read from the plugin's table blob as it is, FP16 [heads][64][64] (four 2-byte loads per cell,
//     issued a stage ahead like the pack's one 8-byte load), instead of lc0ex's t5 pack [8 groups][64][64][4 heads].
//   * the H export is FP16 [B, heads, 64, 64] (what BT6TripletSite reads), not FP32.
//   * the gates' 2 sigmoid(x) = 1 + tanh.approx(x / 2) as in lc0ex's t5 (GATE_TANH); the plugin's v9 build uses 2 / (1 + exp(-x)).
//     Measured on the GB300 against a float64 reference (test/test_t5.cu): output rms error 8.357e-5 vs 8.363e-5, t5 6-9 %
//     faster. -DT5_GATE_EXACT restores the exact form.
//   * debug probes (T5_TRACE, T5_PROF) and the pricing repeat loops are left out.
// Inputs as launch_egt_attention: qkvg FP16 [B, 64, 4096] = [Q | K | V | G], edges INT32 [B, 64, 64, 2], norm S FP16 [B, 64, 64],
// state FP16 [B, 64, 64, 16] cell-major, tab16 FP16 (query [H][40][32] | key [H][40][32] | read|door|gate [H][48] |
// constant_bias [H][64][64]), tab32 FP32 (qk_scale [H] | attack [H][34] | scaled_coefficients [H][34] | gate_b [H]).
// Output FP16 [B, 64, 1024]; H FP16 [B, 32, 64, 64] when h_out is not null.
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdlib>
#include "t5a.h"
#include "bt6_kernels.h"

#define T5_HEADS 32
#define T5_HD 32
#define T5_GATE_SCALE 2.0f

namespace {
using namespace t5a;

constexpr int kHeads = T5_HEADS, kHd = T5_HD, kWidth = kHeads * kHd, kStride = 4 * kWidth;   // OGATE: packed [Q | K | V | G]
constexpr int kGroup = 4, kGroups = kHeads / kGroup;
// warps 0-3 compute, warps 4-5 request operands, warp 6 issues every product
constexpr int kWorkers = 128, kLoaders = 64, kThreads = kWorkers + kLoaders + 32;
constexpr int kRows = 32, kCells = kRows * 64, kChunks = kCells / 128, kTab = 40;
constexpr int kCellCap = 256;  // cell-list form: entries of one warp's list; past it the warp walks all its cells (exact)
constexpr int kBitCap = 512;   // rank-list form: entries of one warp's bit list; past it the same walk
constexpr uint32_t kNoBit = 0xffffffffu;
constexpr int kStagePitch = 66 * 8, kTabPitch = 21 * 16;  // 132 and 84 words: 16-byte accesses of the 32 lanes tile the banks
static_assert(kHeads == 32 && kHd == 32, "written for 32 heads x 32");

// ---- shared memory (bytes; the base is aligned to 1024, every operand block to its swizzle period)
constexpr int kOffTmem = 0, kOffMbar = 64, kOffAG = 256;                       // header: Tensor Memory base; mbarriers; {attack, G} [40][2][4] FP32
constexpr int kHdr = 2048;
constexpr int kOffState = kHdr;                                                // [2048 cells][16] FP16, K-major 32B-swizzled
constexpr int kOffStage = kOffState + kCells * 32;                             // [4 heads][32 rows][64 keys] {bias f32, door f16, gate f16}
constexpr int kOffQ = kOffStage + 67584;                                       // [128 = (head, row)][32] FP16, K-major 64B-swizzled
constexpr int kOffK = kOffQ + 128 * 64;                                        // [256 = (head, key)][32] FP16, K-major 64B-swizzled
constexpr int kOffT = kOffK + 256 * 64;                                        // query tables [160][32], then key tables [160][32]
constexpr int kOffKT = kOffT + 160 * 64;                                       // (V takes K's block: 2 planes [64 keys][64] MN-major 128B-swizzled)
constexpr int kOffW = kOffT + 2 * 160 * 64;                                    // [16 = (head, term)][16] FP16, K-major 32B-swizzled
constexpr int kOffQq = kOffW + 1024;                                           // [32 rows][40][4 heads] FP16
constexpr int kOffQk = kOffQq + 10752;                                         // [64 keys][40][4 heads] FP16
constexpr int kOffNz = kOffQk + 21504;                                         // [4 warps] x 2 KB: the warp's cell list (uint64) or bit list (uint32)
constexpr int kOffS = kOffNz + 4 * kCellCap * 8;                               // [2048 cells] FP16: S of the half-position
static_assert(kCellCap * 8 == kBitCap * 4, "the two lists share one block");
constexpr int kSmemUsed = kOffS + kCells * 2;
constexpr int kSmem = kSmemUsed + 1024;                                        // + the alignment pad
static_assert(kOffT % 1024 == 0 && kOffQ % 512 == 0 && kOffK % 512 == 0 && kOffState % 256 == 0 && kOffW % 256 == 0, "swizzle alignment");
static_assert(2 * 64 * 128 <= 256 * 64 && kOffK % 1024 == 0, "V fits K's block");
static_assert(kGroup * kRows * kStagePitch <= 67584 && kRows * kTabPitch <= 10752 && 64 * kTabPitch <= 21504, "padded blocks");
static_assert(kSmem <= 232448, "shared memory over the sm_103 opt-in ceiling");
static_assert(kSmem == 226816, "release 4's shared-memory size");

// ---- Tensor Memory columns (512)
// Round A of the NEXT group is issued while this group's rows run, so it must not touch this group's P and O.
constexpr int kColQq = 0, kColSt = 160;                     // round A: Qq [0, 160), chunks 0-15 [160, 416)
constexpr int kColQK = 0, kColQk = 256;                     // round B: qk [0, 256), Qk [256, 416)
constexpr int kColP = 416, kColO = 448;                     // rows: P [416, 448), O [448, 512) (one head pair's 64 columns, lane-masked)
// mbarriers. Rounds of products (committed by the driver): A (Qq and the 16 state chunks), B (q.k, Qk), C (P V).
// Signals to the driver: R (every worker warp has loaded q.k), L (both loader warps have the next group's Q, tables and W in).
constexpr int kMbA = 0, kMbB = 1, kMbC = 2, kMbR = 3, kMbL = 4;

__device__ __forceinline__ void cp16(void* d, const void* s) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(saddr(d)), "l"(s) : "memory");
}
__device__ __forceinline__ void named_bar(int id, int count) { asm volatile("bar.sync %0, %1;\n" :: "r"(id), "r"(count) : "memory"); }
__device__ __forceinline__ void cp_wait_all() { asm volatile("cp.async.wait_all;\n" ::: "memory"); }
__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float two_sigmoid(float x) {
#ifndef T5_GATE_EXACT  // lc0ex's t5 form (GATE_TANH): one special-function op; -DT5_GATE_EXACT = the plugin v9 build's 2 / (1 + exp(-x))
  float y;
  asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(0.5f * x));
  return 1.f + y;
#else
  return __fdividef(2.f, 1.f + __expf(-x));
#endif
}
__device__ __forceinline__ uint32_t pack(float lo, float hi) {
  const __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&h);
}
__device__ __forceinline__ float lo_f(uint32_t w) { return __low2float(*reinterpret_cast<const __half2*>(&w)); }
__device__ __forceinline__ float hi_f(uint32_t w) { return __high2float(*reinterpret_cast<const __half2*>(&w)); }
// the plugin's relation bits: INT32 pairs (bits 0-16, bits 17-33) per cell
__device__ __forceinline__ unsigned long long load_edge(const int* __restrict__ edges, size_t cell) {
  const int2 v = __ldg(reinterpret_cast<const int2*>(edges + 2 * cell));
  return static_cast<unsigned long long>(static_cast<unsigned int>(v.x)) |
         (static_cast<unsigned long long>(static_cast<unsigned int>(v.y)) << 17);
}

template <int kGpc, bool kExportH>
__global__ void __launch_bounds__(kThreads, 1) egt_t5_attention(
    __half* __restrict__ out, __half* __restrict__ hexp, const __half* __restrict__ qkv, const int* __restrict__ edges,
    const __half* __restrict__ norm, const __half* __restrict__ state, const float* __restrict__ qk_scale_t,
    const float* __restrict__ attack_t, const __half* __restrict__ key16, const __half* __restrict__ query16,
    const float* __restrict__ scaled_t, const __half* __restrict__ cbias16, const __half* __restrict__ weights16,
    const float* __restrict__ gate_b_t) {
  static_assert(kGpc == 1 || kGpc == 2 || kGpc == 4 || kGpc == 8, "GPC");
  constexpr int kCtasPerHalf = kGroups / kGpc;
  constexpr bool kRank = kGpc >= 8;   // E stage form: rank list where a CTA owns all 8 groups, cell list otherwise
  // query16 / key16 [heads][40][32]; weights16 [heads][read 16 | door 16 | gate 16]; cbias16 [heads][64 rows][64 keys]
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const int tid = threadIdx.x, q = tid >> 5, lane = tid & 31;
  const uint32_t s0 = saddr(smem_raw), pad = (1024u - (s0 & 1023u)) & 1023u;
  unsigned char* smem = smem_raw + pad;
  unsigned char* sState = smem + kOffState;
  unsigned char* sStage = smem + kOffStage;
  unsigned char* sQ = smem + kOffQ;
  unsigned char* sK = smem + kOffK;
  unsigned char* sT = smem + kOffT;
  unsigned char* sW = smem + kOffW;
  unsigned char* sQq = smem + kOffQq;  // FP16 (row, head, channel) at row * kTabPitch + head * 80 + channel * 2
  unsigned char* sQk = smem + kOffQk;  // FP16 (key, head, channel)
  unsigned long long* sCell = reinterpret_cast<unsigned long long*>(smem + kOffNz) + q * kCellCap;
  uint32_t* sBits = reinterpret_cast<uint32_t*>(smem + kOffNz) + q * kBitCap;
  const __half* sS = reinterpret_cast<const __half*>(smem + kOffS);
  float4* sAG = reinterpret_cast<float4*>(smem + kOffAG);
  uint32_t* s_tmem = reinterpret_cast<uint32_t*>(smem + kOffTmem);
  const uint32_t mbar = saddr(smem + kOffMbar);

  const int hp = blockIdx.x / kCtasPerHalf, g_first = (blockIdx.x % kCtasPerHalf) * kGpc;
  const int sample = hp >> 1, row0 = (hp & 1) * kRows;
  const size_t cell0 = ((size_t)sample * 64 + row0) * 64;  // first cell of the half-position in [batch][64][64] arrays
  const __half* row_base = qkv + (size_t)sample * 64 * kStride;

  // ---- requests of one group's operands in 16-byte chunks: requester `me` of `of` takes the chunks me, me + of, ...
  auto load_q = [&](int h0, int me, int of) {  // Q rows: row R = (head a, row i)
#pragma unroll 2
    for (int i = me; i < 128 * 4; i += of) {
      const int R = i >> 2, ch = i & 3;
      cp16(sQ + swz_off(R, ch, 64), row_base + (size_t)(row0 + (R & 31)) * kStride + (h0 + (R >> 5)) * kHd + ch * 8);
    }
  };
  auto load_k = [&](int h0, int me, int of) {  // K rows: row R = (head a, key j)
#pragma unroll 2
    for (int i = me; i < 256 * 4; i += of) {
      const int R = i >> 2, ch = i & 3;
      cp16(sK + swz_off(R, ch, 64), row_base + (size_t)(R & 63) * kStride + kWidth + (h0 + (R >> 6)) * kHd + ch * 8);
    }
  };
  auto load_tables = [&](int h0, int me, int of) {  // rows R = (head a, channel c) of both tables
#pragma unroll 2
    for (int i = me; i < 640; i += of) {
      const int R = i >> 2, ch = i & 3;
      cp16(sT + swz_off(R, ch, 64), query16 + ((size_t)h0 * kTab + R) * kHd + ch * 8);
      cp16(smem + kOffKT + swz_off(R, ch, 64), key16 + ((size_t)h0 * kTab + R) * kHd + ch * 8);
    }
    if (me < 24) cp16(sW + swz_off(me >> 1, me & 1, 32), weights16 + (size_t)h0 * 48 + (me >> 1) * 16 + (me & 1) * 8);
  };
  auto load_v = [&](int h0, int me, int of) {  // V into K's block: plane p = heads (2p, 2p + 1), row = key j, 128 bytes
#pragma unroll 2
    for (int i = me; i < 2 * 64 * 8; i += of) {
      const int p = i >> 9, j = (i >> 3) & 63, ch = i & 7;
      cp16(sK + p * 8192 + swz_off(j, ch, 128), row_base + (size_t)j * kStride + 2 * kWidth + (h0 + 2 * p) * kHd + ch * 8);
    }
  };

  // ---- once per CTA: Tensor Memory, the mbarriers, the state and S into shared memory, this warp's list of cells with edge bits,
  //      and the first group's operands
  if (q == 0) tmem_alloc(s_tmem, 512);
  if (tid == 0) {
    mbar_init(mbar + 8 * kMbA, 1);
    mbar_init(mbar + 8 * kMbB, 1);
    mbar_init(mbar + 8 * kMbC, 1);
    mbar_init(mbar + 8 * kMbR, 4);
    mbar_init(mbar + 8 * kMbL, 2);
    mbar_init_fence();
  }
  if (q == 6) {  // ---- the driver: every product, as early as its operands and its Tensor Memory columns allow.
    // Lane 0 issues; the whole warp walks the same path (a barrier counts a warp as soon as any of its threads arrives).
    __syncthreads();  // setup
    const uint32_t tm = *s_tmem;
    const uint64_t d_state = smem_desc(saddr(sState), 1, 256, k32), d_w = smem_desc(saddr(sW), 1, 256, k32),
                   d_q = smem_desc(saddr(sQ), 1, 512, k64), d_k = smem_desc(saddr(sK), 1, 512, k64),
                   d_qt = smem_desc(saddr(sT), 1, 512, k64), d_kt = smem_desc(saddr(smem + kOffKT), 1, 512, k64),
                   d_v = smem_desc(saddr(sK), 8192 >> 4, 1024, k128);
    const uint32_t id_state = idesc(128, 16, 1, 0, 0), id_qq = idesc(128, 160, 1, 0, 0), id_qk = idesc(128, 80, 1, 0, 0),
                   id_qkk = idesc(128, 256, 1, 0, 0), id_pv = idesc(128, 64, 1, 0, 1);
#define T5_ROUND_A()                                                                                              \
    if (lane == 0) {                                                                                              \
      fence_async();                                                                                              \
      mma_ss(tm + kColQq, d_q, d_qt, id_qq, 0);                                                                   \
      mma_ss(tm + kColQq, d_q + 2, d_qt + 2, id_qq, 1);                                                           \
      _Pragma("unroll 1") for (int c = 0; c < kChunks; ++c) mma_ss(tm + kColSt + 16 * c, d_state + 256 * c, d_w, id_state, 0); \
      commit(mbar + 8 * kMbA);                                                                                    \
    }
    T5_ROUND_A()
#pragma unroll 1
    for (int gi = 0; gi < kGpc; ++gi) {
      const uint32_t parity = gi & 1;
      __syncthreads();  // S1: the workers are past round A's columns; K is in
      if (lane == 0) {
        fence_async();
        fence_after();
        mma_ss(tm + kColQK, d_q, d_k, id_qkk, 0);
        mma_ss(tm + kColQK, d_q + 2, d_k + 2, id_qkk, 1);
#pragma unroll
        for (int r = 0; r < 2; ++r) {  // head pair r: A = K rows of the pair, B = its key tables
          mma_ss(tm + kColQk + 80 * r, d_k + 512 * r, d_kt + 320 * r, id_qk, 0);
          mma_ss(tm + kColQk + 80 * r, d_k + 512 * r + 2, d_kt + 320 * r + 2, id_qk, 1);
        }
        commit(mbar + 8 * kMbB);
      }
      if (gi + 1 < kGpc) {  // the next group's round A, under this group's rows
        mbar_wait(mbar + 8 * kMbR, parity);  // every worker has loaded this group's q.k (and Qk long before)
        mbar_wait(mbar + 8 * kMbL, parity);  // the next group's Q, tables and W are in
        fence_after();
        T5_ROUND_A()
      }
      __syncthreads();  // S4: every lane's P is in Tensor Memory; V is in
      if (lane == 0) {
        fence_async();
        fence_after();
#pragma unroll
        for (int pr = 0; pr < 2; ++pr) {  // head pair pr: its plane of V, its 64 lanes
          const uint32_t off = pr ? 0xffffffffu : 0u, on = ~off;
#pragma unroll
          for (int st = 0; st < 4; ++st)
            mma_ts(tm + kColO, tm + kColP + 8 * st, d_v + 512 * pr + 128 * st, id_pv, st > 0, off, off, on, on);
        }
        commit(mbar + 8 * kMbC);
      }
    }
#undef T5_ROUND_A
    __syncthreads();  // the end
    return;
  }
  if (q >= 4) {  // ---- the loaders: every operand request after the state, timed by the products' barriers
    const int me = tid - kWorkers;
    load_q(g_first * kGroup, me, kLoaders);
    load_tables(g_first * kGroup, me, kLoaders);
    load_k(g_first * kGroup, me, kLoaders);
    cp_wait_all();
    __syncthreads();  // setup
#pragma unroll 1
    for (int gi = 0; gi < kGpc; ++gi) {
      const int h0 = (g_first + gi) * kGroup;
      const uint32_t parity = gi & 1;
      const bool more = gi + 1 < kGpc;
      __syncthreads();  // S1: K is in; the driver issues round B
      mbar_wait(mbar + 8 * kMbB, parity);  // every product that reads Q, K, the tables or W has completed
      if (more) {
        load_q(h0 + kGroup, me, kLoaders);
        load_tables(h0 + kGroup, me, kLoaders);
        cp_wait_all();
        if (lane == 0) mbar_arrive(mbar + 8 * kMbL);  // the driver may issue the next group's round A
      }
      load_v(h0, me, kLoaders);
      cp_wait_all();
      __syncthreads();  // S4: V is in (and the next group's Q, tables, W); the driver issues P V
      mbar_wait(mbar + 8 * kMbC, parity);  // P V has completed: K's block is free
      if (more) load_k(h0 + kGroup, me, kLoaders);
      cp_wait_all();
    }
    __syncthreads();  // the end
    return;
  }
  int nz_count = 0;  // warp-uniform: this warp's cells with edge bits (cell list) or its padded bit-list length (rank list)
  {
    unsigned long long mask[kChunks];
    __half shalf[kChunks];
#pragma unroll
    for (int c = 0; c < kChunks; ++c) {
      mask[c] = load_edge(edges, cell0 + (2 * c + (q >> 1)) * 64 + 32 * (q & 1) + lane);
      shalf[c] = __ldg(norm + cell0 + (2 * c + (q >> 1)) * 64 + 32 * (q & 1) + lane);
    }
    const __half* src = state + cell0 * 16;
#pragma unroll 8
    for (int i = tid; i < kCells * 2; i += kWorkers) cp16(sState + swz_off(i >> 1, i & 1, 32), src + (size_t)(i >> 1) * 16 + (i & 1) * 8);
#pragma unroll
    for (int i = tid; i < kCells / 8; i += kWorkers) cp16(smem + kOffS + i * 16, norm + cell0 + i * 8);
    if (tid < 8) *reinterpret_cast<uint4*>(sW + 12 * 32 + tid * 16) = make_uint4(0, 0, 0, 0);  // the 4 unused rows of W's 16
    if constexpr (kRank) {
      // The set bits of this warp's 512 cells as (row, key, channel) entries, grouped by RANK: first every cell's lowest bit, then
      // every cell's second bit, ... each rank padded to 32 entries. A pass over 32 consecutive entries then never holds a cell
      // twice, so the passes add into the staged bias without loops or conflicts.
      (void)shalf;
#pragma unroll 1
      for (;;) {
        uint32_t mine = 0u;  // the chunks in which this lane's cell still has a bit
#pragma unroll
        for (int c = 0; c < kChunks; ++c) mine |= (mask[c] != 0ull ? 1u : 0u) << c;
        const uint32_t todo = __reduce_or_sync(0xffffffffu, mine);
        if (!todo) break;
#pragma unroll
        for (int c = 0; c < kChunks; ++c) {
          if (todo >> c & 1u) {
            const bool has = mask[c] != 0ull;
            const uint32_t ballot = __ballot_sync(0xffffffffu, has);
            const int at = nz_count + __popc(ballot & ((1u << lane) - 1u));
            if (has && at < kBitCap)
              sBits[at] = static_cast<uint32_t>((2 * c + (q >> 1)) << 12 | (32 * (q & 1) + lane) << 6 | (__ffsll(static_cast<long long>(mask[c])) - 1));
            mask[c] &= mask[c] - 1ull;
            nz_count += __popc(ballot);
          }
        }
        const int padded = (nz_count + 31) & ~31;
        if (nz_count + lane < padded && nz_count + lane < kBitCap) sBits[nz_count + lane] = kNoBit;
        nz_count = padded;
      }
    } else {
      // This warp's cells with edge bits, in chunk order: 8 bytes each = the cell's 34-bit mask | (row << 6 | key) << 34 |
      // S (FP16) << 48, padded with empty entries to 32. The E stage takes 32 cells per pass and walks their bits in
      // registers, so a cell with many bits costs iterations of one pass, not passes.
#pragma unroll
      for (int c = 0; c < kChunks; ++c) {
        const bool has = mask[c] != 0ull;
        const uint32_t ballot = __ballot_sync(0xffffffffu, has);
        const int at = nz_count + __popc(ballot & ((1u << lane) - 1u));
        if (has && at < kCellCap)
          sCell[at] = mask[c] | (static_cast<unsigned long long>((2 * c + (q >> 1)) << 6 | (32 * (q & 1) + lane)) << 34)
                      | (static_cast<unsigned long long>(__half_as_ushort(shalf[c])) << 48);
        nz_count += __popc(ballot);
      }
      if (nz_count <= kCellCap) {
        const int padded = (nz_count + 31) & ~31;
        if (nz_count + lane < padded && nz_count + lane < kCellCap) sCell[nz_count + lane] = 0ull;
      }
    }
  }
  cp_wait_all();
  __syncthreads();  // setup: the state, S and (the loaders) the first group's operands are in
  const uint32_t tm = *s_tmem, lane0 = static_cast<uint32_t>(q * 32) << 16;
  const int cj = 32 * (q & 1) + lane;  // this thread's key in the cells stages
  // ---- a group's constants, requested a stage before their use (the first group's here, the next group's under P V):
  //      this thread's 16 cells' constant bias (4 FP16 per cell, one per head of the group, from the [heads][64][64] table),
  //      the gate biases, the q.k scale of this thread's head, and (40 threads) the {attack, G} table of the E terms
#ifdef T5_CB_PACK  // experiment: the constant bias as lc0ex's t5 pack [8 groups][64 rows][64 keys][4 heads], one 8-byte load per cell
  uint2 cbw[kChunks];
#define T5_CB_LOAD(g_)                                                                                                         \
    _Pragma("unroll") for (int c = 0; c < kChunks; ++c)                                                                        \
      cbw[c] = __ldg(reinterpret_cast<const uint2*>(cbias16 + ((size_t)(g_) * 4096 + (row0 + 2 * c + (q >> 1)) * 64 + cj) * 4));
#define T5_CB(c, a) ((a) == 0 ? lo_f(cbw[c].x) : (a) == 1 ? hi_f(cbw[c].x) : (a) == 2 ? lo_f(cbw[c].y) : hi_f(cbw[c].y))
#else
  __half cbh[kChunks][4];
#define T5_CB_LOAD(g_)                                                                                                         \
    _Pragma("unroll") for (int c = 0; c < kChunks; ++c)                                                                        \
      _Pragma("unroll") for (int a = 0; a < 4; ++a)                                                                            \
        cbh[c][a] = __ldg(cbias16 + ((size_t)((g_) * kGroup + a) * 64 + (row0 + 2 * c + (q >> 1))) * 64 + cj);
#define T5_CB(c, a) __half2float(cbh[c][a])
#endif
  float gb[4], qk_scale;
#define T5_REQUEST(g_, gb_, scale_)                                                                                            \
  {                                                                                                                            \
    T5_CB_LOAD(g_)                                                                                                             \
    _Pragma("unroll") for (int a = 0; a < 4; ++a) gb_[a] = __ldg(gate_b_t + (g_) * kGroup + a);                                \
    scale_ = __ldg(qk_scale_t + (g_) * kGroup + q);                                                                            \
  }
#define T5_AG_LOAD(g_, att_, sig_)                                                                                             \
  if (tid < 34) {                                                                                                              \
    const int h_ = (g_) * kGroup;                                                                                              \
    att_ = make_float4(__ldg(attack_t + h_ * 34 + tid), __ldg(attack_t + (h_ + 1) * 34 + tid), __ldg(attack_t + (h_ + 2) * 34 + tid), \
                       __ldg(attack_t + (h_ + 3) * 34 + tid));                                                                 \
    sig_ = make_float4(__ldg(scaled_t + h_ * 34 + tid), __ldg(scaled_t + (h_ + 1) * 34 + tid), __ldg(scaled_t + (h_ + 2) * 34 + tid), \
                       __ldg(scaled_t + (h_ + 3) * 34 + tid));                                                                 \
  }
#define T5_AG_STORE(att_, sig_)                                                                                                \
  if (tid < kTab) {                                                                                                            \
    sAG[2 * tid] = att_;                                                                                                       \
    sAG[2 * tid + 1] = sig_;                                                                                                   \
  }
  {
    float4 att = make_float4(0.f, 0.f, 0.f, 0.f), sig = att;
    T5_REQUEST(g_first, gb, qk_scale)
    T5_AG_LOAD(g_first, att, sig)
    T5_AG_STORE(att, sig)
  }

#pragma unroll 1
  for (int gi = 0; gi < kGpc; ++gi) {
    const int g = g_first + gi, h0 = g * kGroup;
    const uint32_t parity = gi & 1;
    const bool more = gi + 1 < kGpc;

    // ---- round A (the driver, issued under the previous group's rows): Qq and the 16 state chunks
    mbar_wait(mbar + 8 * kMbA, parity);
    fence_after();
    {  // Qq as FP16 [row][head][channel]: this lane = (head q, row lane), its head's 40 columns = 80 bytes
      float t[kTab];
      tmem_ldf_x32(t, tm + lane0 + kColQq + kTab * q);
      tmem_ldf_x8(t + 32, tm + lane0 + kColQq + kTab * q + 32);
      uint4* dst = reinterpret_cast<uint4*>(sQq + lane * kTabPitch + q * 80);
#pragma unroll
      for (int v = 0; v < 5; ++v)
        dst[v] = make_uint4(pack(t[8 * v], t[8 * v + 1]), pack(t[8 * v + 2], t[8 * v + 3]), pack(t[8 * v + 4], t[8 * v + 5]),
                            pack(t[8 * v + 6], t[8 * v + 7]));
    }
    // ---- cells: this thread owns cell (row 2c + (q >> 1), key cj) of every chunk c; four chunks (64 adjacent columns) at a time
#define T5_CELLS(b, column)                                                                                                  \
    {                                                                                                                        \
      float t[64];                                                                                                           \
      tmem_ldf_x64(t, tm + lane0 + (column));                                                                                \
      _Pragma("unroll") for (int cc = 0; cc < 4; ++cc) {                                                                     \
        const int c = 4 * (b) + cc, r = 2 * c + (q >> 1);                                                                    \
        const float s = __half2float(sS[r * 64 + cj]);                                                                       \
        _Pragma("unroll") for (int a = 0; a < 4; ++a) {                                                                      \
          const float bias = fmaf(s, T5_CB(c, a), t[16 * cc + 3 * a]), x = t[16 * cc + 3 * a + 2] + gb[a];      \
          const float g2 = two_sigmoid(x);                                                                                   \
          *reinterpret_cast<uint2*>(sStage + (a * kRows + r) * kStagePitch + cj * 8) =                                        \
              make_uint2(__float_as_uint(bias), pack(t[16 * cc + 3 * a + 1], g2));                                           \
        }                                                                                                                    \
      }                                                                                                                      \
    }
    T5_CELLS(0, kColSt)
    T5_CELLS(1, kColSt + 64)
    T5_CELLS(2, kColSt + 128)
    T5_CELLS(3, kColSt + 192)
#undef T5_CELLS
    wait_ld();
    fence_before();
    __syncthreads();  // S1 (with the loaders and the driver: K is in): Qq is visible; every lane is past round A's columns

    // ---- round B (the driver): q.k and Qk (two head pairs)
    mbar_wait(mbar + 8 * kMbB, parity);
    fence_after();
    {  // Qk as FP16 [key][head][channel]: this lane = (head 2r + a2, key cj) of head pair r
      float u[2][kTab];
      const int a2 = q >> 1;
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        tmem_ldf_x32(u[r], tm + lane0 + kColQk + 80 * r + kTab * a2);
        tmem_ldf_x8(u[r] + 32, tm + lane0 + kColQk + 80 * r + kTab * a2 + 32);
      }
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        uint4* dst = reinterpret_cast<uint4*>(sQk + cj * kTabPitch + (2 * r + a2) * 80);
#pragma unroll
        for (int v = 0; v < 5; ++v)
          dst[v] = make_uint4(pack(u[r][8 * v], u[r][8 * v + 1]), pack(u[r][8 * v + 2], u[r][8 * v + 3]),
                              pack(u[r][8 * v + 4], u[r][8 * v + 5]), pack(u[r][8 * v + 6], u[r][8 * v + 7]));
      }
    }
    wait_ld();
    named_bar(1, kWorkers);  // S2 (workers): Qk is visible
    if (nz_count <= (kRank ? kBitCap : kCellCap)) {
      if constexpr (kRank) {
        // ---- the E terms: one pass per 32 entries of this warp's bit list, added into the staged bias
#pragma unroll 1
        for (int k = 0; k < nz_count; k += 32) {
          const uint32_t entry = sBits[k + lane];
          const bool live = entry != kNoBit;
          const int r = live ? static_cast<int>(entry >> 12) : 0, j = live ? static_cast<int>((entry >> 6) & 63u) : 0,
                    ch = live ? static_cast<int>(entry & 63u) : 0;
          // every load first (their lanes are scattered), then the arithmetic
          const __half* rq = reinterpret_cast<const __half*>(sQq + r * kTabPitch) + ch;
          const __half* ck = reinterpret_cast<const __half*>(sQk + j * kTabPitch) + ch;
          float* b0 = reinterpret_cast<float*>(sStage + r * kStagePitch + j * 8);
          const __half sh = sS[r * 64 + j];
          const __half q0 = rq[0], q1 = rq[40], q2 = rq[80], q3 = rq[120], k0 = ck[0], k1 = ck[40], k2 = ck[80], k3 = ck[120];
          const float4 att = sAG[2 * ch], sig = sAG[2 * ch + 1];
          const float o0 = b0[0], o1 = b0[kRows * kStagePitch / 4], o2 = b0[2 * kRows * kStagePitch / 4], o3 = b0[3 * kRows * kStagePitch / 4];
          const float s = __half2float(sh);
          const float v0 = o0 + (__half2float(q0) + __half2float(k0) + fmaf(s, sig.x, att.x));
          const float v1 = o1 + (__half2float(q1) + __half2float(k1) + fmaf(s, sig.y, att.y));
          const float v2 = o2 + (__half2float(q2) + __half2float(k2) + fmaf(s, sig.z, att.z));
          const float v3 = o3 + (__half2float(q3) + __half2float(k3) + fmaf(s, sig.w, att.w));
          if (live) {
            b0[0] = v0;
            b0[kRows * kStagePitch / 4] = v1;
            b0[2 * kRows * kStagePitch / 4] = v2;
            b0[3 * kRows * kStagePitch / 4] = v3;
          }
        }
      } else {
        // ---- the E terms: 32 of this warp's non-empty cells per pass; the staged biases are read first, the cells' bits are
        //      walked together (the warp loops while any lane has a bit left; a lane without one adds nothing), then written back
#pragma unroll 1
        for (int k = 0; k < nz_count; k += 32) {
          const unsigned long long entry = sCell[k + lane];
          unsigned long long m = entry & 0x3ffffffffull;
          const bool live = m != 0ull;
          const int r = static_cast<int>(entry >> 40) & 31, j = static_cast<int>(entry >> 34) & 63;
          const float s = __half2float(__ushort_as_half(static_cast<unsigned short>(entry >> 48)));
          float* b0 = reinterpret_cast<float*>(sStage + r * kStagePitch + j * 8);
          const float o0 = b0[0], o1 = b0[kRows * kStagePitch / 4], o2 = b0[2 * kRows * kStagePitch / 4], o3 = b0[3 * kRows * kStagePitch / 4];
          const __half* rq0 = reinterpret_cast<const __half*>(sQq + r * kTabPitch);
          const __half* ck0 = reinterpret_cast<const __half*>(sQk + j * kTabPitch);
          float e0 = 0.f, e1 = 0.f, e2 = 0.f, e3 = 0.f;
#pragma unroll 1
          while (__any_sync(0xffffffffu, m != 0ull)) {
            const bool on = m != 0ull;
            const int ch = on ? __ffsll(static_cast<long long>(m)) - 1 : 0;
            m &= m - 1ull;
            const __half q0 = rq0[ch], q1 = rq0[ch + 40], q2 = rq0[ch + 80], q3 = rq0[ch + 120];
            const __half k0 = ck0[ch], k1 = ck0[ch + 40], k2 = ck0[ch + 80], k3 = ck0[ch + 120];
            const float4 att = sAG[2 * ch], sig = sAG[2 * ch + 1];
            const float v0 = __half2float(q0) + __half2float(k0) + fmaf(s, sig.x, att.x);
            const float v1 = __half2float(q1) + __half2float(k1) + fmaf(s, sig.y, att.y);
            const float v2 = __half2float(q2) + __half2float(k2) + fmaf(s, sig.z, att.z);
            const float v3 = __half2float(q3) + __half2float(k3) + fmaf(s, sig.w, att.w);
            e0 += on ? v0 : 0.f;
            e1 += on ? v1 : 0.f;
            e2 += on ? v2 : 0.f;
            e3 += on ? v3 : 0.f;
          }
          if (live) {
            b0[0] = o0 + e0;
            b0[kRows * kStagePitch / 4] = o1 + e1;
            b0[2 * kRows * kStagePitch / 4] = o2 + e2;
            b0[3 * kRows * kStagePitch / 4] = o3 + e3;
          }
        }
      }
    } else {  // more entries than the list holds: every lane walks its own 16 cells' masks (exact, slow)
#pragma unroll 1
      for (int c = 0; c < kChunks; ++c) {
        const int r = 2 * c + (q >> 1);
        unsigned long long m = load_edge(edges, cell0 + r * 64 + cj);
        const float s = __half2float(sS[r * 64 + cj]);
        float e[4] = {0.f, 0.f, 0.f, 0.f};
        while (m) {
          const int ch = __ffsll(static_cast<long long>(m)) - 1;
          m &= m - 1ull;
          const __half* rq = reinterpret_cast<const __half*>(sQq + r * kTabPitch) + ch;
          const __half* ck = reinterpret_cast<const __half*>(sQk + cj * kTabPitch) + ch;
          const float4 att = sAG[2 * ch], sig = sAG[2 * ch + 1];
          e[0] += __half2float(rq[0]) + __half2float(ck[0]) + fmaf(s, sig.x, att.x);
          e[1] += __half2float(rq[40]) + __half2float(ck[40]) + fmaf(s, sig.y, att.y);
          e[2] += __half2float(rq[80]) + __half2float(ck[80]) + fmaf(s, sig.z, att.z);
          e[3] += __half2float(rq[120]) + __half2float(ck[120]) + fmaf(s, sig.w, att.w);
        }
#pragma unroll
        for (int a = 0; a < 4; ++a) *reinterpret_cast<float*>(sStage + (a * kRows + r) * kStagePitch + cj * 8) += e[a];
      }
    }
    named_bar(1, kWorkers);  // S3 (workers): the staging buffer is complete

    // ---- rows: this thread owns (head q, row lane)
    const int h = h0 + q, row = row0 + lane;
    uint4 gq[4];  // the output-gate lane of this (row, head), requested after the softmax
    {
      float hv[64];
      uint32_t dg[64];
      tmem_ldf_x64(hv, tm + lane0 + kColQK + 64 * q);
      wait_ld();
      fence_before();
      if (lane == 0) mbar_arrive(mbar + 8 * kMbR);  // this warp is past q.k (and Qk): the next group's round A may overwrite them
      {
        const uint4* st = reinterpret_cast<const uint4*>(sStage + (q * kRows + lane) * kStagePitch);
#pragma unroll
        for (int p = 0; p < 32; ++p) {
          const uint4 w = st[p];  // {bias, door | gate} of keys 2p, 2p + 1
          dg[2 * p] = w.y;
          dg[2 * p + 1] = w.w;
          const float f0 = fmaf(hv[2 * p], qk_scale, __uint_as_float(w.x)), f1 = fmaf(hv[2 * p + 1], qk_scale, __uint_as_float(w.z));
          hv[2 * p] = fmaf(f0, lo_f(w.y), f0);
          hv[2 * p + 1] = fmaf(f1, lo_f(w.w), f1);
        }
      }
      if constexpr (kExportH) {  // H FP16 [B, heads, 64, 64]: this thread's row of 64 keys, 128 bytes
        uint4* dst = reinterpret_cast<uint4*>(hexp + (((size_t)sample * kHeads + h) * 64 + row) * 64);
#pragma unroll
        for (int p = 0; p < 8; ++p)
          dst[p] = make_uint4(pack(hv[8 * p], hv[8 * p + 1]), pack(hv[8 * p + 2], hv[8 * p + 3]), pack(hv[8 * p + 4], hv[8 * p + 5]),
                              pack(hv[8 * p + 6], hv[8 * p + 7]));
      }
      float z[8];
      constexpr float kLog2e = 1.4426950408889634f;
      {
        float mx[4] = {hv[0], hv[1], hv[2], hv[3]};
#pragma unroll
        for (int j = 4; j < 64; ++j) mx[j & 3] = fmaxf(mx[j & 3], hv[j]);
        const float m = fmaxf(fmaxf(mx[0], mx[1]), fmaxf(mx[2], mx[3]));
        const float shift = -m * kLog2e;
#pragma unroll
        for (int i = 0; i < 8; ++i) z[i] = 0.f;
#pragma unroll
        for (int j = 0; j < 64; ++j) hv[j] = fmaf(hv[j], kLog2e, shift);
#pragma unroll
        for (int j = 0; j < 64; ++j) {  // the exponentials occupy the special-function unit; the conversions run beside them
          hv[j] = ex2(hv[j]);
          dg[j] = __float_as_uint(hi_f(dg[j]));
        }
#pragma unroll
        for (int j = 0; j < 64; ++j) {
          z[j & 7] += hv[j];
          hv[j] *= __uint_as_float(dg[j]);
        }
      }
      {
        const uint4* gsrc = reinterpret_cast<const uint4*>(row_base + (size_t)row * kStride + 3 * kWidth + h * kHd);
#pragma unroll
        for (int i = 0; i < 4; ++i) gq[i] = __ldg(gsrc + i);
      }
      uint32_t pk[32];
      {
        const float inv = 1.f / (((z[0] + z[1]) + (z[2] + z[3])) + ((z[4] + z[5]) + (z[6] + z[7])));
#pragma unroll
        for (int p = 0; p < 32; ++p) pk[p] = pack(hv[2 * p] * inv, hv[2 * p + 1] * inv);
        wait_ld();
        tmem_st_x32(tm + lane0 + kColP, pk);
        wait_st();
      }
      fence_before();
    }
    // the next group's constants, requested while the driver finishes the next round A: they arrive under the gate factors
    float4 att_next = make_float4(0.f, 0.f, 0.f, 0.f), sig_next = att_next;
    if (more) {
      T5_REQUEST(g + 1, gb, qk_scale)
      T5_AG_LOAD(g + 1, att_next, sig_next)
    }
    __syncthreads();  // S4 (with the loaders: V is in): every lane's P is in Tensor Memory
    // ---- round C (the driver): P V, each head pair's plane of V into the same 64 columns of its own lanes
    // ---- O: this lane's head's 32 columns; the output gate (its factors computed while the product runs); FP16.
    {
      float o[32];
      constexpr float half_scale = 0.5f * T5_GATE_SCALE;
      float gf[32];
      {
        const uint32_t gw[16] = {gq[0].x, gq[0].y, gq[0].z, gq[0].w, gq[1].x, gq[1].y, gq[1].z, gq[1].w,
                                 gq[2].x, gq[2].y, gq[2].z, gq[2].w, gq[3].x, gq[3].y, gq[3].z, gq[3].w};
#pragma unroll
        for (int p = 0; p < 16; ++p) {
          gf[2 * p] = lo_f(gw[p]);
          gf[2 * p + 1] = hi_f(gw[p]);
        }
#pragma unroll
        for (int d = 0; d < 32; ++d) gf[d] = two_sigmoid(gf[d]);
#pragma unroll
        for (int d = 0; d < 32; ++d) gf[d] *= half_scale;
      }
      if (more) { T5_AG_STORE(att_next, sig_next) }  // this group's E terms are long done
      mbar_wait(mbar + 8 * kMbC, parity);
      fence_after();
      tmem_ldf_x32(o, tm + lane0 + kColO + 32 * (q & 1));
      wait_ld();
      uint32_t ow[16];
#pragma unroll
      for (int p = 0; p < 16; ++p) ow[p] = pack(o[2 * p] * gf[2 * p], o[2 * p + 1] * gf[2 * p + 1]);
      uint4* dst = reinterpret_cast<uint4*>(out + ((size_t)sample * 64 + row) * kWidth + h * kHd);
#pragma unroll
      for (int p = 0; p < 4; ++p) dst[p] = make_uint4(ow[4 * p], ow[4 * p + 1], ow[4 * p + 2], ow[4 * p + 3]);
    }
    named_bar(1, kWorkers);  // S5 (workers): every lane is past its loads: Tensor Memory is free for the next group
  }
#undef T5_REQUEST
#undef T5_CB_LOAD
#undef T5_CB
#undef T5_AG_LOAD
#undef T5_AG_STORE
  __syncthreads();  // the end (with the loaders and the driver)
  if (q == 0) tmem_free(tm, 512);
}

// ---- host side
template <int G, bool H>
cudaError_t t5_attr() {
  return cudaFuncSetAttribute(egt_t5_attention<G, H>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem);
}

// Head groups per CTA for a batch: grid = 2 x batch x (8 / GPC) CTAs at one per SM. A CTA's time is about (1 + GPC) group
// steps (the half-position's state load and pipeline start cost about one group, ATTENTION_T5.md), so pick the GPC with the
// fewest waves x (1 + GPC); ties go to the larger GPC (fewer state loads). On 152 SMs this gives lc0ex's auto rule at batch
// 8 / 16 / 32 / 64 (1 / 2 / 4 / 8) and GPC 4 where 2 x batch CTAs would leave a mostly empty second wave (e.g. batch 96).
int pick_gpc(int batch, int sms) {
  static const int forced = [] { const char* e = std::getenv("BT6_EGT_T5_GPC"); return e ? std::atoi(e) : 0; }();
  if (forced == 1 || forced == 2 || forced == 4 || forced == 8) return forced;
  if (sms <= 0) sms = 152;
  int best = 8;
  long best_cost = -1;
  for (int g = 8; g >= 1; g >>= 1) {
    const long ctas = 2L * batch * (8 / g), waves = (ctas + sms - 1) / sms, cost = waves * (1 + g);
    if (best_cost < 0 || cost < best_cost) { best = g; best_cost = cost; }
  }
  return best;
}

}  // namespace

namespace bt6 {

bool egt_t5_supported() {
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) return false;
  // per device: 0 unknown, 1 usable, 2 not (another architecture, or the attributes failed)
  static int state[64] = {0};
  if (state[dev] == 0) {
    int major = 0, minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
    bool ok = major == 10 && minor == 3;
    if (ok) {
      const cudaError_t r[8] = {t5_attr<1, false>(), t5_attr<2, false>(), t5_attr<4, false>(), t5_attr<8, false>(),
                                t5_attr<1, true>(), t5_attr<2, true>(), t5_attr<4, true>(), t5_attr<8, true>()};
      for (cudaError_t e : r) ok = ok && e == cudaSuccess;
      if (!ok) cudaGetLastError();
    }
    state[dev] = ok ? 1 : 2;
  }
  return state[dev] == 1;
}

bool egt_t5_usable() {
  static const bool disabled = [] { const char* e = std::getenv("BT6_EGT_T5"); return e && e[0] == '0'; }();
  return !disabled && egt_t5_supported();
}

int launch_egt_t5(void* out, const __half* qkvg, const int* edges, const __half* norm, const __half* state,
                  const __half* tab16, const float* tab32, int batch, int sms, cudaStream_t stream, __half* h_out) {
  if (batch <= 0) return 0;
  const __half* q16 = tab16;
  const __half* k16 = q16 + kHeads * 40 * kHd;
  const __half* rdg = k16 + kHeads * 40 * kHd;
  const __half* cb = rdg + kHeads * 48;
  const float* qk_scale = tab32;
  const float* attack = qk_scale + kHeads;
  const float* scaled = attack + kHeads * 34;
  const float* gate_b = scaled + kHeads * 34;
  const int gpc = pick_gpc(batch, sms);
  const int grid = 2 * batch * (kGroups / gpc);
  __half* o = static_cast<__half*>(out);
#define T5_LAUNCH(G, H) egt_t5_attention<G, H><<<grid, kThreads, kSmem, stream>>>(o, h_out, qkvg, edges, norm, state, qk_scale, \
                                                                                 attack, k16, q16, scaled, cb, rdg, gate_b)
  if (h_out) {
    switch (gpc) { case 1: T5_LAUNCH(1, true); break; case 2: T5_LAUNCH(2, true); break; case 4: T5_LAUNCH(4, true); break; default: T5_LAUNCH(8, true); }
  } else {
    switch (gpc) { case 1: T5_LAUNCH(1, false); break; case 2: T5_LAUNCH(2, false); break; case 4: T5_LAUNCH(4, false); break; default: T5_LAUNCH(8, false); }
  }
#undef T5_LAUNCH
  return static_cast<int>(cudaGetLastError());
}

}  // namespace bt6

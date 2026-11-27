// k4 (2026-11-25): BT6I8GemmGlu -- INT8 x INT8 -> INT32 tensor-core GEMM (CUTLASS 2.x API, CUTLASS 4.8 headers,
// third_party/cutlass48_include) with a sigmoid-GLU epilogue that writes INT8 codes (lc0ex's Q1 `glu` / `i8` epilogue,
// lczero_triton bt4/kernels/cutlass_gemm_i8.py, transcribed: same visitor, same math, same rounding rule).
//
//   acc[m, n] = sum_k a[m, k] * w[n, k]                  a INT8 [M, K] row-major (codes), w INT8 [N, K] row-major
//   v[m, n]   = fmaf(float(acc), scale[n], bias[n])      FP32
//   columns are interleaved gate / up pairs (n = 2 j gate, 2 j + 1 up):
//   h[m, j]   = sigmoid(v[m, 2j]) * v[m, 2j + 1]         sigmoid = __fdividef(1, 1 + __expf(-g))
//   out[m, j] = clamp(floor(fmaf(h, r[j], 0.5)), -127, 127)  INT8 [M, N / 2]
// tab FP32 = scale [N] | bias [N] | r [N / 2].
// N must be a multiple of the tile's N, K a multiple of 16; M is arbitrary (predicated).
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/threadblock/default_mma.h>
#include <cutlass/epilogue/threadblock/default_epilogue_tensor_op.h>
#include <cutlass/epilogue/threadblock/epilogue_with_visitor.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>
#ifdef BT6_WITH_SM90GLU
#include "bt6_kernels.h"  // launch_i8gemm_glu_sm90 (bt6_i8gemm_sm90.cu)
#endif

namespace bt6i8 {

struct Params {
  int8_t* out;
  const int8_t* a;
  const int8_t* w;
  const float* scale;
  const float* bias;
  const float* prescale;
  int M, N, K;
};

template <int TM, int TN, int TK, int WM, int WN, int WK, int S>
struct GluGemm {
  using ElementA = int8_t;
  using ElementB = int8_t;
  using ElementAccumulator = int32_t;
  using ElementTile = cutlass::half_t;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using ThreadblockShape = cutlass::gemm::GemmShape<TM, TN, TK>;
  using WarpShape = cutlass::gemm::GemmShape<WM, WN, WK>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
  static constexpr int kThreads = (TM / WM) * (TN / WN) * (TK / WK) * 32;
  static constexpr int kTM = TM, kTN = TN;

  using Mma = typename cutlass::gemm::threadblock::DefaultMma<
      ElementA, LayoutA, 16, ElementB, LayoutB, 16, ElementAccumulator, LayoutC, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, ThreadblockShape, WarpShape, InstructionShape, S,
      cutlass::arch::OpMultiplyAddSaturate>::ThreadblockMma;

  using PlaceholderOp = cutlass::epilogue::thread::LinearCombination<
      ElementTile, 128 / cutlass::sizeof_bits<ElementTile>::value, ElementAccumulator, float>;
  using DefaultEpilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
      ThreadblockShape, typename Mma::Operator, ThreadblockShape::kK / WarpShape::kK, PlaceholderOp,
      PlaceholderOp::kCount>::Epilogue;
  using OutputTileIterator = typename DefaultEpilogue::OutputTileIterator;

  struct Visitor {
    static int const kIterations = OutputTileIterator::kIterations;
    static int const kElementsPerAccess = OutputTileIterator::kElementsPerAccess;
    static_assert(kElementsPerAccess == 8, "the visitor is written for 8 accumulator columns per access");
    using ElementAccumulator = int32_t;
    using AccumulatorFragment = cutlass::Array<ElementAccumulator, kElementsPerAccess>;
    struct SharedStorage {};
    static int const kColumnIterations = OutputTileIterator::ThreadMap::Iterations::kColumn;

    OutputTileIterator coordinates_;
    Params p_;
    float scale_cache_[kColumnIterations][8];
    float bias_cache_[kColumnIterations][8];

    CUTLASS_DEVICE Visitor(int thread_index, cutlass::MatrixCoord const& threadblock_offset, Params const& p)
        : coordinates_(typename OutputTileIterator::Params(LayoutC(p.N)), nullptr, {p.M, p.N}, thread_index,
                       threadblock_offset),
          p_(p) {}

    CUTLASS_DEVICE void set_k_partition(int, int) {}
    CUTLASS_DEVICE void set_batch_index(int) {}
    CUTLASS_DEVICE void begin_epilogue() {
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < kColumnIterations; ++c) {
        const int column = coordinates_.thread_start().column() + OutputTileIterator::ThreadMap::iteration_offset(c).column();
        if (column < p_.N) {
          const float4 s0 = __ldg(reinterpret_cast<const float4*>(p_.scale + column));
          const float4 s1 = __ldg(reinterpret_cast<const float4*>(p_.scale + column + 4));
          const float4 b0 = __ldg(reinterpret_cast<const float4*>(p_.bias + column));
          const float4 b1 = __ldg(reinterpret_cast<const float4*>(p_.bias + column + 4));
          scale_cache_[c][0] = s0.x; scale_cache_[c][1] = s0.y; scale_cache_[c][2] = s0.z; scale_cache_[c][3] = s0.w;
          scale_cache_[c][4] = s1.x; scale_cache_[c][5] = s1.y; scale_cache_[c][6] = s1.z; scale_cache_[c][7] = s1.w;
          bias_cache_[c][0] = b0.x; bias_cache_[c][1] = b0.y; bias_cache_[c][2] = b0.z; bias_cache_[c][3] = b0.w;
          bias_cache_[c][4] = b1.x; bias_cache_[c][5] = b1.y; bias_cache_[c][6] = b1.z; bias_cache_[c][7] = b1.w;
        }
      }
    }
    CUTLASS_DEVICE void begin_step(int) {}
    CUTLASS_DEVICE void begin_row(int) {}
    CUTLASS_DEVICE void end_row(int) {}
    CUTLASS_DEVICE void end_step(int) { ++coordinates_; }
    CUTLASS_DEVICE void end_epilogue() {}

    CUTLASS_DEVICE void visit(int, int, int, int fragment_index, AccumulatorFragment const& accumulators) {
      const cutlass::MatrixCoord coordinate =
          coordinates_.thread_start() + OutputTileIterator::ThreadMap::iteration_offset(fragment_index);
      const int row = coordinate.row();
      const int column = coordinate.column();
      if (row >= p_.M || column >= p_.N) return;
      const int group = fragment_index % kColumnIterations;
      float value[8];
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < 8; ++i) value[i] = fmaf(static_cast<float>(accumulators[i]), scale_cache_[group][i], bias_cache_[group][i]);
      const int hidden = column / 2;
      float h[4];
      CUTLASS_PRAGMA_UNROLL
      for (int q = 0; q < 4; ++q) {
#ifdef BT6_I8GLU_CONTROL_NO_SIGMOID
        const float gate = value[2 * q];   // must-fail control: GLU without the sigmoid
#else
        const float gate = __fdividef(1.0f, 1.0f + __expf(-value[2 * q]));
#endif
        h[q] = gate * value[2 * q + 1];
      }
      const float4 r = __ldg(reinterpret_cast<const float4*>(p_.prescale + hidden));
      const float rs[4] = {r.x, r.y, r.z, r.w};
      uint32_t word = 0;
      CUTLASS_PRAGMA_UNROLL
      for (int q = 0; q < 4; ++q) {
        const float code = fminf(fmaxf(floorf(fmaf(h[q], rs[q], 0.5f)), -127.0f), 127.0f);
        word |= (static_cast<uint32_t>(static_cast<int32_t>(code)) & 0xffu) << (8 * q);
      }
      *reinterpret_cast<uint32_t*>(p_.out + static_cast<int64_t>(row) * (p_.N / 2) + hidden) = word;
    }
  };

  using Epilogue =
      typename cutlass::epilogue::threadblock::EpilogueWithVisitorFromExistingEpilogue<Visitor, DefaultEpilogue>::Epilogue;

  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };
};

template <class G>
__global__ void __launch_bounds__(G::kThreads) glu_gemm_kernel(Params p) {
  extern __shared__ __align__(16) char smem_base[];
  typename G::SharedStorage& shared = *reinterpret_cast<typename G::SharedStorage*>(smem_base);
  const int tile_row = blockIdx.x;
  const int tile_column = blockIdx.y;
  const cutlass::MatrixCoord offset_a{tile_row * G::kTM, 0};
  const cutlass::MatrixCoord offset_b{0, tile_column * G::kTN};
  const int k_iterations = (p.K + G::ThreadblockShape::kK - 1) / G::ThreadblockShape::kK;
  typename G::Mma::IteratorA iterator_a(typename G::Mma::IteratorA::Params(typename G::LayoutA(p.K)),
                                        const_cast<int8_t*>(p.a), {p.M, p.K}, threadIdx.x, offset_a);
  typename G::Mma::IteratorB iterator_b(typename G::Mma::IteratorB::Params(typename G::LayoutB(p.K)),
                                        const_cast<int8_t*>(p.w), {p.K, p.N}, threadIdx.x, offset_b);
  const int warp_index = threadIdx.x / 32;
  const int lane_index = threadIdx.x % 32;
  typename G::Mma::FragmentC accumulators;
  accumulators.clear();
  typename G::Mma mma(shared.main_loop, threadIdx.x, warp_index, lane_index);
  mma(k_iterations, accumulators, iterator_a, iterator_b, accumulators);
  const cutlass::MatrixCoord offset_d{tile_row * G::kTM, tile_column * G::kTN};
  typename G::Visitor visitor(threadIdx.x, offset_d, p);
  typename G::Epilogue epilogue(shared.epilogue, threadIdx.x, warp_index, lane_index);
  epilogue(visitor, accumulators);
}

template <class G>
int launch_cfg(Params const& p, cudaStream_t stream) {
  static int configured_device = -1;
  const int smem = static_cast<int>(sizeof(typename G::SharedStorage));
  int dev = 0;
  cudaGetDevice(&dev);
  if (configured_device != dev) {
    cudaError_t e = cudaFuncSetAttribute(glu_gemm_kernel<G>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (e != cudaSuccess) return static_cast<int>(e);
    configured_device = dev;
  }
  if (p.N % G::kTN != 0) return static_cast<int>(cudaErrorInvalidValue);
  const dim3 grid((p.M + G::kTM - 1) / G::kTM, p.N / G::kTN, 1);
  glu_gemm_kernel<G><<<grid, G::kThreads, smem, stream>>>(p);
  return static_cast<int>(cudaGetLastError());
}

// Tile table (index = the BT6_I8GLU_TILE override; the default choice by M is below).
using T0 = GluGemm<64, 64, 64, 32, 32, 64, 4>;
using T1 = GluGemm<64, 128, 64, 32, 64, 64, 3>;
using T2 = GluGemm<128, 64, 64, 64, 32, 64, 3>;
using T3 = GluGemm<128, 128, 64, 64, 64, 64, 3>;
using T4 = GluGemm<128, 128, 64, 64, 64, 64, 4>;
using T5 = GluGemm<256, 128, 64, 64, 64, 64, 3>;
using T6 = GluGemm<128, 128, 64, 32, 64, 64, 3>;
using T7 = GluGemm<128, 256, 64, 64, 64, 64, 3>;
using T8 = GluGemm<64, 64, 128, 32, 32, 128, 3>;
using T9 = GluGemm<128, 128, 64, 64, 32, 64, 3>;
using T10 = GluGemm<128, 128, 64, 64, 64, 64, 2>;
using T11 = GluGemm<64, 256, 64, 64, 64, 64, 3>;
using T12 = GluGemm<256, 64, 64, 64, 64, 64, 3>;
constexpr int kTiles = 13;

int launch_tile(int tile, Params const& p, cudaStream_t s) {
  switch (tile) {
    case 0: return launch_cfg<T0>(p, s);
    case 1: return launch_cfg<T1>(p, s);
    case 2: return launch_cfg<T2>(p, s);
    case 3: return launch_cfg<T3>(p, s);
    case 4: return launch_cfg<T4>(p, s);
    case 5: return launch_cfg<T5>(p, s);
    case 6: return launch_cfg<T6>(p, s);
    case 7: return launch_cfg<T7>(p, s);
    case 8: return launch_cfg<T8>(p, s);
    case 9: return launch_cfg<T9>(p, s);
    case 10: return launch_cfg<T10>(p, s);
    case 11: return launch_cfg<T11>(p, s);
    case 12: return launch_cfg<T12>(p, s);
  }
  return static_cast<int>(cudaErrorInvalidValue);
}

int default_tile(int M) {
#ifdef BT6_I8GLU_DEFAULT_TABLE
  BT6_I8GLU_DEFAULT_TABLE
#endif
  // RTX 5090 sweep (k4/bench_tiles_1.txt, N 2048, K 1024): tile 3 fastest at every M from 1024 to 16384
  (void)M;
  return 3;
}

}  // namespace bt6i8

namespace bt6 {

int i8gemm_glu_tiles() { return bt6i8::kTiles; }

// tile < 0: env BT6_I8GLU_TILE if set, else the default by M.
int launch_i8gemm_glu(int8_t* out, const int8_t* a, const int8_t* w, const float* tab, int M, int N, int K, int tile,
                      cudaStream_t stream) {
  if (M <= 0) return 0;
  if (N % 128 != 0 || K % 64 != 0) return static_cast<int>(cudaErrorInvalidValue);
  if (tile < 0) {
    static int env_tile = -2;
    if (env_tile == -2) {
      const char* e = std::getenv("BT6_I8GLU_TILE");
      env_tile = e && *e ? std::atoi(e) : -1;
    }
#ifdef BT6_WITH_SM90GLU
    // Hopper (compute capability 9.0): the TMA + wgmma kernel (bt6_i8gemm_sm90.cu), bit-identical codes; env BT6_I8GLU_SM90=0
    // or an explicit tile (attribute or BT6_I8GLU_TILE) keeps this file's Sm80 kernel; on any launch error it is the fallback.
    static const bool sm90_off = [] { const char* e = std::getenv("BT6_I8GLU_SM90"); return e && e[0] == '0'; }();
    if (!sm90_off && env_tile < 0 && i8gemm_glu_sm90_supported() &&
        launch_i8gemm_glu_sm90(out, a, w, tab, M, N, K, -1, stream) == 0)
      return 0;
#endif
    tile = env_tile >= 0 ? env_tile : bt6i8::default_tile(M);
  }
  bt6i8::Params p{out, a, w, tab, tab + N, tab + 2 * N, M, N, K};
  return bt6i8::launch_tile(tile, p, stream);
}

}  // namespace bt6

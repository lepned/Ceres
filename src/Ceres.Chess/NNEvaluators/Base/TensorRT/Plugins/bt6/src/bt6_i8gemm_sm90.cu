// BT6I8GemmGlu on Hopper (sm_90a): the same INT8 x INT8 -> INT32 GEMM with the sigmoid-GLU epilogue writing INT8 codes as
// bt6_i8gemm_kernels.cu, on Hopper's warpgroup tensor cores (wgmma) fed by the Tensor Memory Accelerator (TMA): CUTLASS 3.x
// warp-specialized persistent kernels (a producer warp issues the TMA loads into a multi-stage shared-memory pipeline,
// consumer warpgroups run wgmma) with a register-level epilogue (GluEpilogue below) in place of CUTLASS's own.
//
//   acc[m, n] = sum_k a[m, k] * w[n, k]                  a INT8 [M, K] row-major (codes), w INT8 [N, K] row-major
//   v[m, n]   = fmaf(float(acc), scale[n], bias[n])      FP32
//   columns are interleaved gate / up pairs (n = 2 j gate, 2 j + 1 up):
//   h[m, j]   = sigmoid(v[m, 2j]) * v[m, 2j + 1]         sigmoid = __fdividef(1, 1 + __expf(-g))
//   out[m, j] = clamp(floor(fmaf(h, r[j], 0.5)), -127, 127)  INT8 [M, N / 2]
// The epilogue uses the Sm80 visitor's expressions verbatim; the accumulation is exact integer arithmetic. So the codes are
// bit-identical to bt6_i8gemm_kernels.cu's (checked by test/test_glu90.cu).
//
// launch_i8gemm_glu (bt6_i8gemm_kernels.cu) calls launch_i8gemm_glu_sm90 on compute capability 9.0 when built with
// BT6_WITH_SM90GLU; env BT6_I8GLU_SM90=0 keeps the Sm80 kernel, BT6_I8GLU_SM90_CFG=<index> forces a configuration.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

#include "bt6_kernels.h"

namespace bt6glu90 {
using namespace cute;

// ---- the epilogue: CUTLASS's DefaultEpilogue interface (wrapped by Sm90TmaWarpSpecializedAdapter for the warp-specialized
//      kernels), operating on the accumulator registers: each thread owns column pairs (n, n + 1), n even, of its rows.
template <class EpilogueSchedule_>
class GluEpilogue {
 public:
  using EpilogueSchedule = EpilogueSchedule_;
  using DispatchPolicy = EpilogueSchedule_;
  using ElementOutput = int8_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = float;
  using ElementScalar = float;
  using ElementC = void;
  using ElementD = int8_t;
  using StrideC = cute::Stride<int64_t, cute::Int<1>, int64_t>;
  using StrideD = cute::Stride<int64_t, cute::Int<1>, int64_t>;
  using GmemTiledCopyC = void;
  using GmemTiledCopyD = void;
  struct ThreadEpilogueOp {  // only named by GemmUniversalAdapter (EpilogueOutputOp); the math is in operator() below
    using ElementOutput = int8_t;
    using ElementAccumulator = int32_t;
    using ElementCompute = float;
  };
  struct SharedStorage {};
  using TensorStorage = SharedStorage;
  struct Arguments {
    int8_t* out = nullptr;            // INT8 [M, N / 2]
    const float* scale = nullptr;     // [N]
    const float* bias = nullptr;      // [N]
    const float* prescale = nullptr;  // r [N / 2]
  };
  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(ProblemShape const&, Arguments const& args, void*) { return args; }
  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const&, Arguments const&) { return 0; }
  template <class ProblemShape>
  static cutlass::Status initialize_workspace(ProblemShape const&, Arguments const&, void*, cudaStream_t,
                                              cutlass::CudaHostAdapter* = nullptr) {
    return cutlass::Status::kSuccess;
  }
  template <class ProblemShape>
  static bool can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    return args.out && args.scale && args.bias && args.prescale && get<1>(problem_shape) % 2 == 0;
  }

  CUTLASS_HOST_DEVICE GluEpilogue(Params const& params_, SharedStorage const& = SharedStorage()) : params(params_) {}
  CUTLASS_DEVICE bool is_source_needed() { return false; }

  template <class ProblemShapeMNKL, class BlockShapeMNK, class BlockCoordMNKL, class FrgEngine, class FrgLayout, class TiledMma,
            class ResidueMNK>
  CUTLASS_DEVICE void operator()(ProblemShapeMNKL problem_shape_mnkl, BlockShapeMNK blk_shape_MNK, BlockCoordMNKL blk_coord_mnkl,
                                 cute::Tensor<FrgEngine, FrgLayout> const& accumulators, TiledMma tiled_mma, ResidueMNK,
                                 int thread_idx, char*) {
    static_assert(is_static<FrgLayout>::value, "accumulator layout must be static");
    const int M = get<0>(problem_shape_mnkl);
    const int N = get<1>(problem_shape_mnkl);
    const int half_n = N / 2;
    auto [m_coord, n_coord, k_coord, l_coord] = blk_coord_mnkl;
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor cD = local_tile(make_identity_tensor(make_shape(M, N)), take<0, 2>(blk_shape_MNK), make_coord(m_coord, n_coord));
    Tensor tCcD = thr_mma.partition_C(cD);  // (m, n) of every accumulator element of this thread
    CUTE_STATIC_ASSERT_V(size(tCcD) == size(accumulators));
    static_assert(decltype(size(accumulators))::value % 2 == 0, "pairs");
    // wgmma's C fragment, per thread: index i = e + 2 h + 4 c + 4 C rr -- e the column of a pair (n0 + 8 c + e), h the row half
    // (+8 rows), c the 8-column chunk (C = BLK_N / 8 of them), rr the 64-row atom (rows from tCcD). The quad (lane & 3 = t) holds
    // one row: thread t owns columns 8 c + 2 t + e, i.e. outputs j = 4 c + t of the tile's BLK_N / 2.
    constexpr int F = decltype(size(accumulators))::value;
    constexpr int BN = decltype(size<1>(blk_shape_MNK))::value;
    constexpr int C = BN / 8, R = F / (4 * C), Q = C / 4, U = Q / 4;  // Q words of 4 codes per row; U 16-byte stores per lane
    static_assert(F == 4 * C * R && Q % 4 == 0, "fragment layout");
    const int n0 = get<1>(tCcD(0));         // this thread's first column: the quad's base + 2 t
    const int t = threadIdx.x & 3;
    const int nq = n0 - 2 * t;              // the quad's (and the tile row's) first column
    uint32_t words[R][2][Q];
    // The code's floor and integer conversion at full rate: clamp(floor(x), +-127) == floor(clamp(x, +-127)) (NaN -> -127
    // both ways), and floor(y) for |y| <= 127 is the low byte of as_uint(fadd.rm(y, 1.5 * 2^23)) -- no FRND / F2I on the
    // quarter-rate conversion pipe. (int -> float is I2FP, already cheap on sm_90.)
    constexpr float kMagic = 12582912.0f;  // 1.5 * 2^23
    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < C; ++c) {
      if ((c & 7) == 0) asm volatile("" ::: "memory");  // load the constants of 8 chunks at a time
      const int n = n0 + 8 * c;  // the pair (n, n + 1); output j = n / 2
      const float2 sc = __ldg(reinterpret_cast<const float2*>(params.scale + n));
      const float2 bi = __ldg(reinterpret_cast<const float2*>(params.bias + n));
      const float r = __ldg(params.prescale + n / 2);
      CUTLASS_PRAGMA_UNROLL
      for (int rr = 0; rr < R; ++rr)
        CUTLASS_PRAGMA_UNROLL
        for (int h = 0; h < 2; ++h) {
          const int i = 2 * h + 4 * c + 4 * C * rr;
#ifdef BT6_GLU90_CHECK_PAIRS
          if (get<1>(tCcD(i)) != n || get<1>(tCcD(i + 1)) != n + 1 || get<0>(tCcD(i + 1)) != get<0>(tCcD(i)) ||
              get<0>(tCcD(i)) != get<0>(tCcD(4 * C * rr)) + 8 * h || (n & 1) || ((n - n0) & 7))
            __trap();
#endif
          const float v0 = fmaf(static_cast<float>(accumulators(i)), sc.x, bi.x);
          const float v1 = fmaf(static_cast<float>(accumulators(i + 1)), sc.y, bi.y);
#ifdef BT6_I8GLU_CONTROL_NO_SIGMOID
          const float gate = v0;
#else
          const float gate = __fdividef(1.0f, 1.0f + __expf(-v0));
#endif
          const float hv = gate * v1;
          const float y = fminf(fmaxf(fmaf(hv, r, 0.5f), -127.0f), 127.0f);
          const uint32_t code = __float_as_uint(__fadd_rd(y, kMagic)) & 0xffu;
          words[rr][h][c >> 2] = (c & 3) == 0 ? code : words[rr][h][c >> 2] | (code << (8 * (c & 3)));
        }
    }
    // Store: word q of quad lane s holds outputs j = 16 q + 4 i + s (i = byte). Lane t writes 16 contiguous outputs of word
    // q = U t + u: four shuffle rounds bring it word q of every quad lane (round k from lane (t + k) & 3), a 4 x 4 byte transpose
    // orders them by (i, k) and a byte rotation by t puts lane s's byte at position s.
    const int lane_base = threadIdx.x & 31 & ~3;
    CUTLASS_PRAGMA_UNROLL
    for (int rr = 0; rr < R; ++rr)
      CUTLASS_PRAGMA_UNROLL
      for (int h = 0; h < 2; ++h) {
        const int m = get<0>(tCcD(4 * C * rr)) + 8 * h;
        CUTLASS_PRAGMA_UNROLL
        for (int u = 0; u < U; ++u) {
          uint32_t rec[4];
          CUTLASS_PRAGMA_UNROLL
          for (int k = 0; k < 4; ++k) {
            const int dst = (t - k) & 3;  // the quad lane that reads this lane's word in round k
            const uint32_t send = dst == 0 ? words[rr][h][u] : dst == 1 ? words[rr][h][U + u]
                                : dst == 2 ? words[rr][h][2 * U + u] : words[rr][h][3 * U + u];
            rec[k] = __shfl_sync(0xffffffffu, send, lane_base | ((t + k) & 3));
          }
          const uint32_t t0 = __byte_perm(rec[0], rec[1], 0x5140), t1 = __byte_perm(rec[0], rec[1], 0x7362);
          const uint32_t t2 = __byte_perm(rec[2], rec[3], 0x5140), t3 = __byte_perm(rec[2], rec[3], 0x7362);
          const uint32_t y0 = __byte_perm(t0, t2, 0x5410), y1 = __byte_perm(t0, t2, 0x7632);
          const uint32_t y2 = __byte_perm(t1, t3, 0x5410), y3 = __byte_perm(t1, t3, 0x7632);
          const int sh = 8 * t;
          const uint4 v = make_uint4(__funnelshift_l(y0, y0, sh), __funnelshift_l(y1, y1, sh), __funnelshift_l(y2, y2, sh),
                                     __funnelshift_l(y3, y3, sh));
          if (m < M)
            *reinterpret_cast<uint4*>(params.out + static_cast<int64_t>(m) * half_n + nq / 2 + 16 * (U * t + u)) = v;
        }
      }
  }

 private:
  Params params;
};

// ---- configurations: tile (M, N, K), cluster, kernel schedule
template <class TileShape, class ClusterShape, class KernelSchedule>
struct Cfg {
  using Epilogue = cutlass::epilogue::collective::detail::Sm90TmaWarpSpecializedAdapter<GluEpilogue<cutlass::epilogue::NoSmemWarpSpecialized>>;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, int8_t, cutlass::layout::RowMajor, 16, int8_t,
      cutlass::layout::ColumnMajor, 16, int32_t, TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
      KernelSchedule>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int, int>, Mainloop, Epilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

using Coop = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using Ping = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
// Pingpong: two consumer warpgroups on alternate tiles, one's epilogue under the other's mainloop. The sweep also tried the
// cooperative 128 x 128 (clusters 1 x 2, 2 x 1), 128 x 256 (2 x 1), pingpong 64 x 256 (1 x 1, 2 x 1) and 128 x 128 (1 x 1):
// none was best at any Ceres size.
using C0 = Cfg<Shape<_64, _128, _128>, Shape<_1, _2, _1>, Ping>;    // M <= 1536
using C1 = Cfg<Shape<_128, _128, _128>, Shape<_2, _1, _1>, Ping>;   // the default from M = 2048
using C2 = Cfg<Shape<_256, _128, _128>, Shape<_1, _2, _1>, Coop>;   // within 1.5 % of C1 for M >= 8448
using C3 = Cfg<Shape<_128, _256, _128>, Shape<_1, _1, _1>, Coop>;
constexpr int kCfgs = 4;

struct Args {
  int8_t* out;
  const int8_t* a;
  const int8_t* w;
  const float* tab;
  int M, N, K;
};

int sm_count_of_device(int dev) {
  static int counts[64] = {0};
  if (dev < 0 || dev >= 64) return 0;
  if (!counts[dev]) cudaDeviceGetAttribute(&counts[dev], cudaDevAttrMultiProcessorCount, dev);
  return counts[dev];
}

template <class C>
int run(Args const& p, cudaStream_t stream) {
  using Gemm = typename C::Gemm;
  int dev = 0;
  cudaGetDevice(&dev);
  cutlass::KernelHardwareInfo hw{dev, sm_count_of_device(dev)};
  typename C::Kernel::StrideA stride_a = make_stride(static_cast<int64_t>(p.K), Int<1>{}, static_cast<int64_t>(p.M) * p.K);
  typename C::Kernel::StrideB stride_b = make_stride(static_cast<int64_t>(p.K), Int<1>{}, static_cast<int64_t>(p.N) * p.K);
  typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
                                {p.M, p.N, p.K, 1},
                                {p.a, stride_a, p.w, stride_b},
                                {p.out, p.tab, p.tab + p.N, p.tab + 2 * p.N},
                                hw};
  Gemm gemm;
  if (gemm.can_implement(args) != cutlass::Status::kSuccess) return static_cast<int>(cudaErrorInvalidValue);
  if (Gemm::get_workspace_size(args) != 0) return static_cast<int>(cudaErrorInvalidValue);  // the plugin has no workspace
  if (gemm.initialize(args, nullptr, stream) != cutlass::Status::kSuccess) return static_cast<int>(cudaErrorInvalidValue);
  if (gemm.run(stream) != cutlass::Status::kSuccess) {
    const cudaError_t e = cudaGetLastError();
    return static_cast<int>(e != cudaSuccess ? e : cudaErrorLaunchFailure);
  }
  return static_cast<int>(cudaGetLastError());
}

int run_cfg(int cfg, Args const& p, cudaStream_t s) {
  switch (cfg) {
    case 0: return run<C0>(p, s);
    case 1: return run<C1>(p, s);
    case 2: return run<C2>(p, s);
    case 3: return run<C3>(p, s);
  }
  return static_cast<int>(cudaErrorInvalidValue);
}

// H100 NVL sweep (test/test_glu90.cu, N 2048, K 1024, Ceres engine sizes M = 64 x 8 .. 224): the pingpong 64 x 128 kernel
// at M = 512 (11.7 us vs 14.7 for the Sm80 kernel), the pingpong 128 x 128 kernel with 2 x 1 clusters from M = 2048 on (its
// epilogue overlaps the other consumer warpgroup's mainloop; best or within 1.5 % of the best configuration at every size,
// 1.33x the Sm80 kernel at M = 2048, 1.53-1.69x from M = 4096).
int default_cfg(int M) { return M <= 1536 ? 0 : 1; }

}  // namespace bt6glu90

namespace bt6 {

int i8gemm_glu_sm90_cfgs() { return bt6glu90::kCfgs; }

bool i8gemm_glu_sm90_supported() {
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) return false;
  static int state[64] = {0};  // 0 unknown, 1 sm_90, 2 other
  if (!state[dev]) {
    int major = 0, minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
    state[dev] = major == 9 && minor == 0 ? 1 : 2;
  }
  return state[dev] == 1;
}

// cfg < 0: env BT6_I8GLU_SM90_CFG if set, else the default by M.
int launch_i8gemm_glu_sm90(int8_t* out, const int8_t* a, const int8_t* w, const float* tab, int M, int N, int K, int cfg,
                           cudaStream_t stream) {
  if (M <= 0) return 0;
  if (N % 256 != 0 || K % 128 != 0) return static_cast<int>(cudaErrorInvalidValue);
  if (cfg < 0) {
    static const int env_cfg = [] { const char* e = std::getenv("BT6_I8GLU_SM90_CFG"); return e && *e ? std::atoi(e) : -1; }();
    cfg = env_cfg >= 0 ? env_cfg : bt6glu90::default_cfg(M);
  }
  return bt6glu90::run_cfg(cfg, bt6glu90::Args{out, a, w, tab, M, N, K}, stream);
}

}  // namespace bt6

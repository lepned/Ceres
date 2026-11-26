// t5a.h -- raw tcgen05 helpers for the attention kernel spikes (sm_103a, cta_group::1). Syntax per the PTX ISA
// (ref/ptx_tcgen05.txt) and CUTLASS 4.4.1 (cute/arch/mma_sm100_umma.hpp, cutlass/arch/barrier.h).
#pragma once
#include <cuda_fp16.h>
#include <cstdint>
#include "tmem_ldst.h"

namespace t5a {
enum Swz : uint64_t { kNone = 0, k128 = 2, k64 = 4, k32 = 6 };
__device__ __forceinline__ uint32_t saddr(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
// Shared-memory matrix descriptor (Table 49): start >> 4 | LBO >> 4 << 16 | SBO >> 4 << 32 | 0b001 << 46 | swizzle << 61.
__device__ __forceinline__ uint64_t smem_desc(uint32_t start, uint32_t lbo_field, uint32_t sbo_bytes, Swz swz) {
  return static_cast<uint64_t>((start >> 4) & 0x7fffu) | (static_cast<uint64_t>(lbo_field & 0x7fffu) << 16)
         | (static_cast<uint64_t>((sbo_bytes >> 4) & 0x3fffu) << 32) | (1ull << 46) | (static_cast<uint64_t>(swz) << 61);
}
// Instruction descriptor for .kind::f16 (Table 51): dtype (0 F16, 1 F32) << 4, transpose A << 15, transpose B << 16, N >> 3 << 17,
// M >> 4 << 24; A and B element types F16 (0).
__host__ __device__ constexpr uint32_t idesc(int m, int n, int d_f32, int a_mn, int b_mn) {
  return (static_cast<uint32_t>(d_f32) << 4) | (static_cast<uint32_t>(a_mn) << 15) | (static_cast<uint32_t>(b_mn) << 16)
         | (static_cast<uint32_t>(n >> 3) << 17) | (static_cast<uint32_t>(m >> 4) << 24);
}
// D[+]= A (smem) * B (smem). One thread issues it. `mask` bit l = 1: lane l of D is NOT written.
__device__ __forceinline__ void mma_ss(uint32_t d_taddr, uint64_t a_desc, uint64_t b_desc, uint32_t id, uint32_t accumulate,
                                       uint32_t m0 = 0, uint32_t m1 = 0, uint32_t m2 = 0, uint32_t m3 = 0) {
  asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
               "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%5, %6, %7, %8}, p;\n\t}\n"
               :: "r"(d_taddr), "l"(a_desc), "l"(b_desc), "r"(id), "r"(accumulate), "r"(m0), "r"(m1), "r"(m2), "r"(m3));
}
// D[+]= A (Tensor Memory, one FP16 per column) * B (smem).
__device__ __forceinline__ void mma_ts(uint32_t d_taddr, uint32_t a_taddr, uint64_t b_desc, uint32_t id, uint32_t accumulate,
                                       uint32_t m0 = 0, uint32_t m1 = 0, uint32_t m2 = 0, uint32_t m3 = 0) {
  asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
               "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, {%5, %6, %7, %8}, p;\n\t}\n"
               :: "r"(d_taddr), "r"(a_taddr), "l"(b_desc), "r"(id), "r"(accumulate), "r"(m0), "r"(m1), "r"(m2), "r"(m3));
}
__device__ __forceinline__ void commit(uint32_t mbar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(mbar) : "memory");
}
__device__ __forceinline__ void fence_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void fence_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wait_st() { asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void fence_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ void mbar_init(uint32_t mbar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mbar), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_init_fence() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
__device__ __forceinline__ void mbar_arrive(uint32_t mbar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(mbar) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t mbar, uint32_t phase) {
  asm volatile("{\n\t.reg .pred P1;\n\tLAB_WAIT:\n\tmbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n\t"
               "@P1 bra DONE;\n\tbra LAB_WAIT;\n\tDONE:\n\t}" :: "r"(mbar), "r"(phase) : "memory");
}
// Tensor Memory allocation: one full warp; writes the base address to *dst (shared memory).
__device__ __forceinline__ void tmem_alloc(uint32_t* dst, uint32_t columns) {
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(saddr(dst)), "r"(columns) : "memory");
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void tmem_free(uint32_t taddr, uint32_t columns) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(taddr), "r"(columns) : "memory");
}
// Byte offset of 16-byte chunk `chunk` of row `row` (row_bytes = 32, 64 or 128) inside a swizzled operand block whose start is
// aligned to 8 rows' bytes: Swizzle<B,4,3> XORs address bits [7, 7 + B) into bits [4, 4 + B).
__device__ __forceinline__ uint32_t swz_off(uint32_t row, uint32_t chunk, uint32_t row_bytes) {
  const uint32_t off = row * row_bytes + chunk * 16u, mask = (row_bytes >> 4) - 1u;
  return off ^ (((off >> 7) & mask) << 4);
}
}  // namespace t5a

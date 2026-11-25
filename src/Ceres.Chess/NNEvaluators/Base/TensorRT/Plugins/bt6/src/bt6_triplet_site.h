// BT6TripletSite host entry points (bt6_triplet_site.cu). Namespace bt6t so the files drop into the r4-based library
// next to bt6_kernels.h without name clashes.
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstddef>

namespace bt6t {

// table: R [16][heads] | Wv [32][16] | Wg [16][16] | bg [16] | Wo [16][32] | W1 [64][16] | Wr [64][16] | b1 [64] | W2 [16][64]
constexpr int kTripletTabFixed = 32 * 16 + 16 * 16 + 16 + 16 * 32 + 64 * 16 + 64 * 16 + 64 + 16 * 64;  // 4432
constexpr int triplet_tab_size(int heads) { return 16 * heads + kTripletTabFixed; }                    // 4944 at 32 heads

// workspace bytes for `batch` positions: e_hat FP32 [B, 4096, 16] | e_hat FP16 [B, 4096, 16] | va FP16 [B, 32, 4096]
size_t triplet_site_workspace(int batch);
// opt the triplet kernel into its 79,360 bytes of dynamic shared memory (once per process / device; cheap to repeat)
int prepare_triplet_site();
// state FP16 [B, 64, 64, 16] cell-major, H FP16 [B, heads, 64, 64], tab FP32 [triplet_tab_size(heads)],
// heads 24 or 32, contraction 0 = path, 1 = ag -> out FP16 [B, 64, 64, 16] cell-major. Returns a cudaError_t.
int launch_triplet_site(__half* out, const __half* state, const __half* H, const float* tab, int heads, int contraction,
                        void* workspace, int batch, cudaStream_t stream);

}  // namespace bt6t

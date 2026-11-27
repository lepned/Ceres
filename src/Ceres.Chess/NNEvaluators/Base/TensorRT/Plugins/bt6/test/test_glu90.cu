// test_glu90: BT6I8GemmGlu -- the Sm80 kernel (launch_i8gemm_glu with an explicit tile) against every Hopper configuration
// (launch_i8gemm_glu_sm90) on the real weights and table of one layer (glu<k>_w.bin INT8 [2048, 1024], glu<k>_tab.bin FP32
// [5120], dumped from the INT8 plugin ONNX) and random activation codes (N(0, sigma) rounded, clamped to +-127).
//   test_glu90 <w.bin> <tab.bin> [sigma] [tile_sm80] [M ...]
// Per M: mismatching codes against the Sm80 kernel (must be 0), times (median of 40 launches, CUDA events) and TOPS.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include "bt6_kernels.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(1); } } while (0)

static std::vector<char> read_file(const char* fn) {
  FILE* f = fopen(fn, "rb");
  if (!f) { fprintf(stderr, "cannot open %s\n", fn); exit(1); }
  fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  std::vector<char> v(n);
  if (fread(v.data(), 1, n, f) != (size_t)n) { fprintf(stderr, "short read %s\n", fn); exit(1); }
  fclose(f);
  return v;
}

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: test_glu90 <w.bin> <tab.bin> [sigma] [tile_sm80] [M ...]\n"); return 2; }
  std::vector<char> wraw = read_file(argv[1]);
  const std::vector<char> traw = read_file(argv[2]);
  if (argc > 3 && atof(argv[3]) < -100) {  // synthetic weights +-127 (the fallback path: |acc| up to 127 * 127 * K)
    std::mt19937 wr(3);
    for (auto& x : wraw) x = (char)((wr() & 1) ? 127 : -127);
    printf("synthetic +-127 weights\n");
  }
  const float sigma = argc > 3 ? atof(argv[3]) : 30.f;
  const int tile80 = argc > 4 ? atoi(argv[4]) : 3;
  std::vector<int> Ms;
  for (int i = 5; i < argc; ++i) Ms.push_back(atoi(argv[i]));
  if (Ms.empty()) Ms = {512, 2048, 4096, 6144, 8448, 10752, 13312, 14336, 32768};
  const int N = 2048, K = 1024;
  if ((int)wraw.size() != N * K || (int)traw.size() != (2 * N + N / 2) * 4) { fprintf(stderr, "sizes\n"); return 1; }
  if (!bt6::i8gemm_glu_sm90_supported()) { fprintf(stderr, "not a compute capability 9.0 device\n"); return 1; }
  const int Mmax = *std::max_element(Ms.begin(), Ms.end());
  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0.f, std::fabs(sigma));
  std::vector<int8_t> a((size_t)Mmax * K);
  for (auto& x : a) x = (int8_t)std::max(-127.f, std::min(127.f, std::nearbyint(nd(rng))));
  if (sigma < 0) {  // adversarial: every 7th row is 127 * sign(w[n, k]) for a pseudo-random n -> |acc| ~ 127 * sum |w| > 2^22
    const int8_t* w8 = reinterpret_cast<const int8_t*>(wraw.data());
    long big = 0;
    for (int m = 0; m < Mmax; m += 7) {
      const int n = (m * 37) % N;
      long acc = 0;
      for (int k = 0; k < K; ++k) { a[(size_t)m * K + k] = w8[(size_t)n * K + k] >= 0 ? 127 : -127; acc += 127L * std::abs((int)w8[(size_t)n * K + k]); }
      big += acc >= (1L << 22);
    }
    printf("adversarial rows: %ld of %d with |acc| >= 2^22 in their target column\n", big, (Mmax + 6) / 7);
  }
  int8_t *d_a, *d_w, *d_o80, *d_o90;
  float* d_tab;
  CK(cudaMalloc(&d_a, a.size())); CK(cudaMalloc(&d_w, wraw.size())); CK(cudaMalloc(&d_tab, traw.size()));
  CK(cudaMalloc(&d_o80, (size_t)Mmax * N / 2)); CK(cudaMalloc(&d_o90, (size_t)Mmax * N / 2));
  CK(cudaMemcpy(d_a, a.data(), a.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_w, wraw.data(), wraw.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_tab, traw.data(), traw.size(), cudaMemcpyHostToDevice));
  cudaStream_t st;
  CK(cudaStreamCreate(&st));
  const int ncfg = bt6::i8gemm_glu_sm90_cfgs();
  const int only_cfg = getenv("CFG") ? atoi(getenv("CFG")) : -1;
  auto time_it = [&](auto launch) {
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    std::vector<float> ms;
    for (int it = 0; it < 45; ++it) {
      CK(cudaEventRecord(e0, st));
      if (launch() != 0) return -1.f;
      CK(cudaEventRecord(e1, st));
      CK(cudaEventSynchronize(e1));
      float t; CK(cudaEventElapsedTime(&t, e0, e1));
      if (it >= 5) ms.push_back(t);
    }
    std::sort(ms.begin(), ms.end());
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return 1000.f * ms[ms.size() / 2];
  };
  printf("sigma %.1f, Sm80 tile %d; per configuration: mismatching codes / us / TOPS\n", sigma, tile80);
  printf("%6s | %-18s", "M", "sm80 us (TOPS)");
  for (int c = 0; c < ncfg; ++c) printf(" | cfg%-2d mis      us   TOPS", c);
  printf("\n");
  for (int M : Ms) {
    CK(cudaMemset(d_o80, 0x55, (size_t)M * N / 2));
    if (bt6::launch_i8gemm_glu(d_o80, d_a, d_w, d_tab, M, N, K, tile80, st) != 0) { fprintf(stderr, "sm80 launch\n"); return 1; }
    CK(cudaStreamSynchronize(st));
    std::vector<int8_t> o80((size_t)M * N / 2), o90(o80.size());
    CK(cudaMemcpy(o80.data(), d_o80, o80.size(), cudaMemcpyDeviceToHost));
    const double ops = 2.0 * M * N * K;
    const float t80 = time_it([&] { return bt6::launch_i8gemm_glu(d_o80, d_a, d_w, d_tab, M, N, K, tile80, st); });
    printf("%6d | %8.1f (%5.0f)  ", M, t80, ops / t80 * 1e-6);
    for (int c = 0; c < ncfg; ++c) {
      if (only_cfg >= 0 && c != only_cfg) continue;
      CK(cudaMemset(d_o90, 0x55, (size_t)M * N / 2));
      const int rc = bt6::launch_i8gemm_glu_sm90(d_o90, d_a, d_w, d_tab, M, N, K, c, st);
      cudaError_t e = cudaStreamSynchronize(st);
      if (rc != 0 || e != cudaSuccess) { printf(" | cfg%-2d FAILED rc %d %s", c, rc, cudaGetErrorString(e)); if (e != cudaSuccess) return 1; continue; }
      CK(cudaMemcpy(o90.data(), d_o90, o90.size(), cudaMemcpyDeviceToHost));
      long mis = 0;
      for (size_t i = 0; i < o80.size(); ++i) mis += o80[i] != o90[i];
      const float t90 = time_it([&] { return bt6::launch_i8gemm_glu_sm90(d_o90, d_a, d_w, d_tab, M, N, K, c, st); });
      printf(" | cfg%-2d %4ld %7.1f %6.0f", c, mis, t90, ops / t90 * 1e-6);
    }
    printf("\n");
    fflush(stdout);
  }
  return 0;
}

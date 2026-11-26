// test_t5: BT6EgtAttention on the GB300 -- the plugin's v9 kernel (launch_egt_attention with BT6_EGT_T5=0) against the t5 port
// (launch_egt_t5) and a float64 CPU reference of bt6_export_onnx.BT6Served.attn_core, on identical inputs: the real tables of one
// layer (k1_tab16_<i>.bin / k1_tab32_<i>.bin dumped from the FP16 plugin ONNX) and synthetic activations (qkvg, state ~ N(0, 1),
// S in [0.3, 3], relation bits with the real positions' statistics: ~214 of 4096 cells set per position, 1-7 bits each).
//   test_t5 <tab16.bin> <tab32.bin> <export_h 0|1> [ref_samples] [batches...]
// Prints per batch: max / rms |error| of out and H for v9 and t5 against the reference (first ref_samples samples), t5 vs v9 over
// the whole batch, and the kernel times (CUDA events, median of 50 launches).
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <algorithm>
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
static float h2f(__half h) { return __half2float(h); }

int main(int argc, char** argv) {
  if (argc < 4) { fprintf(stderr, "usage: test_t5 <tab16.bin> <tab32.bin> <export_h> [ref_samples] [batches...]\n"); return 2; }
  setenv("BT6_EGT_T5", "0", 1);  // launch_egt_attention -> v9
  const std::vector<char> t16raw = read_file(argv[1]), t32raw = read_file(argv[2]);
  const bool export_h = atoi(argv[3]) != 0;
  const int ref_samples = argc > 4 ? atoi(argv[4]) : 2;
  std::vector<int> batches;
  for (int a = 5; a < argc; ++a) batches.push_back(atoi(argv[a]));
  if (batches.empty()) batches = {8, 32, 64, 96, 132, 168, 208, 224, 512};
  const int H = 32, HD = 32, W = 1024, S4 = 4096;
  if ((int)t16raw.size() != bt6::kEgtTab16 * 2 || (int)t32raw.size() != bt6::kEgtTab32 * 4) { fprintf(stderr, "table sizes\n"); return 1; }
  const __half* t16 = reinterpret_cast<const __half*>(t16raw.data());
  const float* t32 = reinterpret_cast<const float*>(t32raw.data());
  const __half *tq = t16, *tk = tq + H * 40 * HD, *trdg = tk + H * 40 * HD, *tcb = trdg + H * 48;
  const float *qk_scale = t32, *attack = qk_scale + H, *scaled = attack + H * 34, *gate_b = scaled + H * 34;

  int dev = 0, sms = 0;
  CK(cudaGetDevice(&dev));
  CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  if (bt6::prepare_egt_attention() != 0) { fprintf(stderr, "prepare_egt_attention failed\n"); return 1; }
  if (!bt6::egt_t5_supported()) { fprintf(stderr, "t5 not supported on this device\n"); return 1; }
  printf("device %d, %d SMs, export_h %d, reference on %d samples\n", dev, sms, (int)export_h, ref_samples);

  const int Bmax = *std::max_element(batches.begin(), batches.end());
  std::mt19937 rng(12345);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::uniform_real_distribution<float> ud(0.3f, 3.f);
  std::vector<__half> qkvg((size_t)Bmax * 64 * S4), state((size_t)Bmax * 4096 * 16), norm((size_t)Bmax * 4096);
  std::vector<int> edges((size_t)Bmax * 4096 * 2, 0);
  for (auto& x : qkvg) x = __float2half(nd(rng));
  for (auto& x : state) x = __float2half(nd(rng));
  for (auto& x : norm) x = __float2half(ud(rng));
  std::vector<unsigned long long> mask((size_t)Bmax * 4096, 0ull);
  for (int b = 0; b < Bmax; ++b) {
    std::vector<int> cells(4096);
    for (int i = 0; i < 4096; ++i) cells[i] = i;
    std::shuffle(cells.begin(), cells.end(), rng);
    for (int n = 0; n < 214; ++n) {
      const int c = cells[n];
      const int bits = 1 + (int)(std::min(6.0, -std::log(std::max(1e-9, (double)(rng() % 1000000) / 1e6)) * 0.5));  // mostly 1, up to 7
      unsigned long long m = 0;
      while (__builtin_popcountll(m) < bits) m |= 1ull << (rng() % 34);
      mask[(size_t)b * 4096 + c] = m;
      edges[((size_t)b * 4096 + c) * 2] = (int)(m & 0x1ffffull);
      edges[((size_t)b * 4096 + c) * 2 + 1] = (int)(m >> 17);
    }
  }
  __half *d_qkvg, *d_state, *d_norm, *d_t16, *d_out_v9, *d_out_t5, *d_h_v9 = nullptr, *d_h_t5 = nullptr;
  int* d_edges;
  float* d_t32;
  CK(cudaMalloc(&d_qkvg, qkvg.size() * 2)); CK(cudaMalloc(&d_state, state.size() * 2)); CK(cudaMalloc(&d_norm, norm.size() * 2));
  CK(cudaMalloc(&d_edges, edges.size() * 4)); CK(cudaMalloc(&d_t16, t16raw.size())); CK(cudaMalloc(&d_t32, t32raw.size()));
  CK(cudaMalloc(&d_out_v9, (size_t)Bmax * 64 * W * 2)); CK(cudaMalloc(&d_out_t5, (size_t)Bmax * 64 * W * 2));
  if (export_h) { CK(cudaMalloc(&d_h_v9, (size_t)Bmax * H * 4096 * 2)); CK(cudaMalloc(&d_h_t5, (size_t)Bmax * H * 4096 * 2)); }
  CK(cudaMemcpy(d_qkvg, qkvg.data(), qkvg.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_state, state.data(), state.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_norm, norm.data(), norm.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_edges, edges.data(), edges.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_t16, t16raw.data(), t16raw.size(), cudaMemcpyHostToDevice));
  __half* d_t16_t5 = d_t16;
#ifdef T5_CB_PACK  // the t5 launches read a copy whose constant-bias section is permuted to [8 groups][64][64][4 heads]
  {
    std::vector<__half> p(t16, t16 + bt6::kEgtTab16);
    __half* cbp = p.data() + (tcb - t16);
    for (int h = 0; h < H; ++h) for (int x = 0; x < 4096; ++x) cbp[((h / 4) * 4096 + x) * 4 + (h % 4)] = tcb[h * 4096 + x];
    CK(cudaMalloc(&d_t16_t5, t16raw.size()));
    CK(cudaMemcpy(d_t16_t5, p.data(), t16raw.size(), cudaMemcpyHostToDevice));
  }
#endif
  CK(cudaMemcpy(d_t32, t32raw.data(), t32raw.size(), cudaMemcpyHostToDevice));
  cudaStream_t st;
  CK(cudaStreamCreate(&st));

  // float64 reference of attn_core for sample b -> out [64][1024], H [32][64][64]
  auto reference = [&](int b, std::vector<double>& out, std::vector<double>& Href) {
    out.assign(64 * W, 0.0); Href.assign(H * 4096, 0.0);
    const __half* L = qkvg.data() + (size_t)b * 64 * S4;
    for (int h = 0; h < H; ++h) {
      std::vector<double> row(64 * 34), col(64 * 34);
      for (int i = 0; i < 64; ++i)
        for (int c = 0; c < 34; ++c) {
          double r = attack[h * 34 + c], k = 0;
          for (int d = 0; d < HD; ++d) {
            r += h2f(L[i * S4 + 0 * W + h * HD + d]) * h2f(tq[(h * 40 + c) * HD + d]);
            k += h2f(L[i * S4 + 1 * W + h * HD + d]) * h2f(tk[(h * 40 + c) * HD + d]);
          }
          row[i * 34 + c] = r; col[i * 34 + c] = k;
        }
      for (int i = 0; i < 64; ++i) {
        std::vector<double> hv(64), gt(64);
        for (int j = 0; j < 64; ++j) {
          const size_t cell = (size_t)b * 4096 + i * 64 + j;
          double qk = 0;
          for (int d = 0; d < HD; ++d) qk += h2f(L[i * S4 + h * HD + d]) * h2f(L[j * S4 + W + h * HD + d]);
          double e = 0, pair = h2f(tcb[h * 4096 + i * 64 + j]);
          const unsigned long long m = mask[cell];
          for (int c = 0; c < 34; ++c)
            if (m >> c & 1ull) { e += row[i * 34 + c] + col[j * 34 + c]; pair += scaled[h * 34 + c]; }
          double rd = 0, dr = 0, gg = 0;
          for (int ch = 0; ch < 16; ++ch) {
            const double s = h2f(state[cell * 16 + ch]);
            rd += h2f(trdg[h * 48 + ch]) * s; dr += h2f(trdg[h * 48 + 16 + ch]) * s; gg += h2f(trdg[h * 48 + 32 + ch]) * s;
          }
          const double S = h2f(norm[cell]);
          hv[j] = (qk_scale[h] * qk + e + S * pair + rd) * (1.0 + dr);
          gt[j] = 2.0 / (1.0 + std::exp(-(gg + gate_b[h])));
          Href[(h * 64 + i) * 64 + j] = hv[j];
        }
        const double mx = *std::max_element(hv.begin(), hv.end());
        double z = 0;
        for (int j = 0; j < 64; ++j) { hv[j] = std::exp(hv[j] - mx); z += hv[j]; }
        for (int d = 0; d < HD; ++d) {
          double acc = 0;
          for (int j = 0; j < 64; ++j) acc += hv[j] / z * gt[j] * h2f(L[j * S4 + 2 * W + h * HD + d]);
          const double og = h2f(L[i * S4 + 3 * W + h * HD + d]);
          out[i * W + h * HD + d] = acc * 2.0 / (1.0 + std::exp(-og));
        }
      }
    }
  };
  std::vector<std::vector<double>> ref_out(ref_samples), ref_h(ref_samples);
  for (int b = 0; b < ref_samples; ++b) reference(b, ref_out[b], ref_h[b]);

  printf("%5s | %-23s | %-23s | %-23s | %-23s | %-17s | %8s %8s %6s\n", "batch", "v9 out max / rms", "t5 out max / rms",
         "v9 H max / rms", "t5 H max / rms", "t5-v9 out max/rms", "v9 us", "t5 us", "ratio");
  for (int B : batches) {
    CK(cudaMemset(d_out_v9, 0, (size_t)B * 64 * W * 2)); CK(cudaMemset(d_out_t5, 0, (size_t)B * 64 * W * 2));
    if (export_h) { CK(cudaMemset(d_h_v9, 0, (size_t)B * H * 4096 * 2)); CK(cudaMemset(d_h_t5, 0, (size_t)B * H * 4096 * 2)); }
    if (bt6::launch_egt_attention(d_out_v9, d_qkvg, d_edges, d_norm, d_state, d_t16, d_t32, B, sms, st, d_h_v9, 0.f) != 0) { fprintf(stderr, "v9 launch\n"); return 1; }
    if (bt6::launch_egt_t5(d_out_t5, d_qkvg, d_edges, d_norm, d_state, d_t16_t5, d_t32, B, sms, st, d_h_t5) != 0) { fprintf(stderr, "t5 launch\n"); return 1; }
    CK(cudaStreamSynchronize(st));
    std::vector<__half> o9((size_t)B * 64 * W), o5((size_t)B * 64 * W), h9, h5;
    CK(cudaMemcpy(o9.data(), d_out_v9, o9.size() * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(o5.data(), d_out_t5, o5.size() * 2, cudaMemcpyDeviceToHost));
    if (export_h) {
      h9.resize((size_t)B * H * 4096); h5.resize(h9.size());
      CK(cudaMemcpy(h9.data(), d_h_v9, h9.size() * 2, cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(h5.data(), d_h_t5, h5.size() * 2, cudaMemcpyDeviceToHost));
    }
    double e9 = 0, r9 = 0, e5 = 0, r5 = 0, eh9 = 0, rh9 = 0, eh5 = 0, rh5 = 0, d95 = 0, rd95 = 0;
    long n = 0, nh = 0;
    bool finite = true;
    for (int b = 0; b < std::min(B, ref_samples); ++b) {
      for (int x = 0; x < 64 * W; ++x) {
        const double rv = ref_out[b][x], a9 = h2f(o9[(size_t)b * 64 * W + x]), a5 = h2f(o5[(size_t)b * 64 * W + x]);
        e9 = std::max(e9, std::fabs(a9 - rv)); r9 += (a9 - rv) * (a9 - rv);
        e5 = std::max(e5, std::fabs(a5 - rv)); r5 += (a5 - rv) * (a5 - rv); ++n;
      }
      if (export_h)
        for (int x = 0; x < H * 4096; ++x) {
          const double rv = ref_h[b][x], a9 = h2f(h9[(size_t)b * H * 4096 + x]), a5 = h2f(h5[(size_t)b * H * 4096 + x]);
          eh9 = std::max(eh9, std::fabs(a9 - rv)); rh9 += (a9 - rv) * (a9 - rv);
          eh5 = std::max(eh5, std::fabs(a5 - rv)); rh5 += (a5 - rv) * (a5 - rv); ++nh;
        }
    }
    for (size_t x = 0; x < o9.size(); ++x) {
      const double a9 = h2f(o9[x]), a5 = h2f(o5[x]);
      if (!std::isfinite(a5) || !std::isfinite(a9)) finite = false;
      d95 = std::max(d95, std::fabs(a9 - a5)); rd95 += (a9 - a5) * (a9 - a5);
    }
    if (export_h)
      for (size_t x = 0; x < h5.size(); ++x) if (!std::isfinite(h2f(h5[x]))) finite = false;
    // timing: median of 50 launches each
    auto time_it = [&](bool t5) {
      cudaEvent_t a, z;
      CK(cudaEventCreate(&a)); CK(cudaEventCreate(&z));
      std::vector<float> ms;
      for (int it = 0; it < 55; ++it) {
        CK(cudaEventRecord(a, st));
        if (t5) bt6::launch_egt_t5(d_out_t5, d_qkvg, d_edges, d_norm, d_state, d_t16_t5, d_t32, B, sms, st, d_h_t5);
        else bt6::launch_egt_attention(d_out_v9, d_qkvg, d_edges, d_norm, d_state, d_t16, d_t32, B, sms, st, d_h_v9, 0.f);
        CK(cudaEventRecord(z, st));
        CK(cudaEventSynchronize(z));
        float t; CK(cudaEventElapsedTime(&t, a, z));
        if (it >= 5) ms.push_back(t);
      }
      std::sort(ms.begin(), ms.end());
      cudaEventDestroy(a); cudaEventDestroy(z);
      return 1000.f * ms[ms.size() / 2];
    };
    const float tv9 = time_it(false), tt5 = time_it(true);
    const double nn = std::max<long>(n, 1), nnh = std::max<long>(nh, 1);
    printf("%5d | %10.3e / %10.3e | %10.3e / %10.3e | %10.3e / %10.3e | %10.3e / %10.3e | %7.1e/%7.1e | %8.1f %8.1f %5.2fx%s\n", B,
           e9, std::sqrt(r9 / nn), e5, std::sqrt(r5 / nn), eh9, std::sqrt(rh9 / nnh), eh5, std::sqrt(rh5 / nnh), d95,
           std::sqrt(rd95 / o9.size()), tv9, tt5, tv9 / tt5, finite ? "" : "  NON-FINITE!");
  }
  return 0;
}

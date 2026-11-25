// trt_bt6_plugin k1 (library bt6_trt_plugin_k1, 2026-11-25) = round 4 + two plugins for Kovax's BT6 flagship
// (1024 x 24, 32 heads, lc0 planes input); every round-1..4 plugin name / version below is unchanged:
// BT6EgtAttention version "1": inputs 0 qkvg FP16 [B, 64, 4096] ([Q | K | V | G], heads side by side), 1 edges INT32
//   [B, 64, 64, 2], 2 norm FP16 [B, 64, 64], 3 state FP16 [B, 64, 64, 16] cell-major, 4 tab16 FP16 [kEgtTab16],
//   5 tab32 FP32 [kEgtTab32]; attributes heads (32), head_dim (32), export_h (0 / 1), out_int8 (0 / 1), out_scale;
//   outputs 0 out FP16 [B, 64, 1024] (output-gated A @ V; INT8 codes with out_int8), 1 (export_h) H FP16
//   [B, 32, 64, 64] = the door-scaled logits (bt6_egt_attn.cu).
// BT6TripletSite version "1": fused edge site incl. the triplet branch (src_k1_triplet/NOTES.md).
// BT6PlanesPrologue version "1": planes FP32 [B, 112, 8, 8] (or [B, 112, 64]) + tab FP32 [kPlanesPrologueTab32],
//   attribute eps (default 1e-6) -> edges INT32 [B, 64, 64, 2], S FP16 [B, 64, 64], e0 FP16 [B, 64, 64, 16],
//   pol FP32 [B, 64, 64] (the policy head's S * pair term) (bt6_prologue_kernels.cu, kPlanes); attribute e0_fp32
//   (0 / 1, default 0): a 5th output e0f FP32 [B, 16, 64, 64] (e0 channels-first, unrounded, for the edge sites).
//
// trt_bt6_plugin round 4 (library bt6_trt_plugin_r4; every round-1, round-2 and round-3 plugin name / version is kept): TensorRT IPluginV3 plugins for the flat 768x16 BT6 trunk.
//
// BT6TermsAttention version "1": unchanged from round 1 (one output; kept so round-1 ONNX files still build).
// BT6TermsAttention version "2": the same attention plus attribute export_h (0 / 1). With export_h = 1 a second
//   output carries the RAW logits the edge sites read, Hraw = q.k / sqrt(hd) + template bias, FP16 [B, heads, 64, 64]
//   (QK-normed q, k): the exported graph's own QK-norm / q.k / template-bias nodes of those layers can then go.
//   Attribute fused_coef (0 / 1): input 0 is [B, 64, 3 * H * hd + H * T], the qkv GEMM with the template mixer's
//   columns appended (one GEMM), and input 1 is the mixer bias FP16 [H * T].
// BT6TermsAttention version "3" (round 3): version 2 plus attributes out_int8 (0 / 1) and out_scale (FP32). With
//   out_int8 = 1 output 0 is INT8 codes round(out / out_scale), clamped to +-127: the W_h input of an INT8 graph
//   (the graph's DequantizeLinear with the same scale reads it; the separate quantize kernel goes).
// BT6EdgeSite version "1": one edge-state site (readback, rms, edge FFN with the reverse edge, rms) on the
//   cell-major state and the layer's raw logits (bt6_site_kernels.cu).
// BT6GluQuant version "1" (round 3): gate INT8 [..., 1536] + up INT8 [..., 1536] (the FFN gate / up GEMM outputs,
//   quantized with gate_scale / up_scale) + gate bias FP16 [1536] -> codes INT8 [..., 1536] = round(127 m / r) and
//   rmax FP16 [..., 1] = r, with m = SiLU(g + b) * u and r = the row's max |m| (bt6_glu_kernels.cu).
// BT6DecAttn version "1" (round 4): one attention of a move-token decoder block (self / cross / post-move), attributes
//   mode (0 / 1 / 2), k_off, v_off (column offsets of k and v), scale (FP32, 1/sqrt(32)) (bt6_dec_kernels.cu):
//   mode 0: qkv FP16 [B, 80, W] (q at 0, k at k_off, v at v_off) + key bias FP16 [B, 80]
//   mode 1: q FP16 [B, 80, 256] + kv FP16 [B, 64, W] (the shared square K|V GEMM)
//   mode 2: q + kv + idx INT32 [B, 80, 3] (from, to, mover) + tab FP16 [2, 7, 256] (pm_dk | pm_dv)
//   -> o FP16 [B, 80, 256], the input of the block's projection GEMM.
// BT6DecHead version "1" (round 4): x FP16 [B, 80, 256] (after the last block) + key bias FP16 [B, 80] + sel INT32
//   [B, 80] + tab FP32 [3138] (out_ln scale | pol W [4, 256] | per-move bias [1858]) + slot INT32 [1858], attribute eps
//   -> policy FP16 [B, 1858], pooled FP16 [B, 512] (mean | max of out_ln(x) over valid tokens).
// BT6Prologue version "1": squares FP16 [B, 64, 137] + tab FP32 [8288] -> edges INT32 [B, 64, 64, 2],
//   S FP16 [B, 64, 64], e0 FP16 [B, 64, 64, 16] cell-major (bt6_prologue_kernels.cu).
//
// BT6TermsAttention inputs (all linear, batch dimension dynamic):
//   0 qkv    FP16 [B, 64, 3 * H * hd]  the layer's raw qkv GEMM output (per head [q | k | v]; QK-norm is applied inside)
//   1 coef   FP16 [B, 64, H * T]       the template mixer output (tmix.mix, with bias)
//   2 edges  INT32 [B, 64, 64, 2]      relation bits per cell: E channels 0-16 | channels 17-33 (exact integers)
//   3 norm   FP16 [B, 64, 64]          S, BT6's pair normaliser (rsqrt(mean_d Z^2 + eps))
//   4 state  FP16 [B, 64, 64, 16]      the edge state e, cell-major (channel innermost)
//   5 tab16  FP16 [N16]                per-layer FP16 packs (see bt6_attn_kernels.cu)
//   6 tab32  FP32 [N32]                per-layer FP32 scalars and QK-norm scales
//   7 tpl    FP16 [T, 64, 64]          the shared template table
// Outputs: 0 out FP16 [B, 64, H * hd] (A @ V, heads side by side); v2 with export_h: 1 hraw FP16 [B, H, 64, 64].
#include <NvInferRuntime.h>
#include <NvInferPlugin.h>

#include <cstring>
#include <string>
#include <vector>

#include "bt6_kernels.h"

using namespace nvinfer1;

namespace {

constexpr char const* kAttnName = "BT6TermsAttention";
constexpr char const* kProName = "BT6Prologue";
constexpr int kAttnInputs = 8;

ILoggerFinder* gLoggerFinder = nullptr;

void log_error(char const* who, char const* msg) {
  std::string m = std::string("[") + who + "] " + msg;
  if (gLoggerFinder) {
    if (ILogger* l = gLoggerFinder->findLogger()) {
      l->log(ILogger::Severity::kERROR, m.c_str());
      return;
    }
  }
  fprintf(stderr, "%s\n", m.c_str());
}

int read_int(PluginField const& f) {
  if (f.type == PluginFieldType::kINT32) return *static_cast<int32_t const*>(f.data);
  if (f.type == PluginFieldType::kINT64) return static_cast<int>(*static_cast<int64_t const*>(f.data));
  if (f.type == PluginFieldType::kINT16) return *static_cast<int16_t const*>(f.data);
  if (f.type == PluginFieldType::kINT8) return *static_cast<int8_t const*>(f.data);
  return -1;
}

float read_float(PluginField const& f) {
  if (f.type == PluginFieldType::kFLOAT32) return *static_cast<float const*>(f.data);
  if (f.type == PluginFieldType::kFLOAT64) return static_cast<float>(*static_cast<double const*>(f.data));
  if (f.type == PluginFieldType::kFLOAT16) return __half2float(*static_cast<__half const*>(f.data));
  return -1.f;
}

int device_sms(int& sms) {
  if (sms > 0) return 0;
  int dev = 0, n = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) return 1;
  if (cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess || n <= 0) return 1;
  sms = n;
  return 0;
}

// ===================================================================================================== attention
class BT6Plugin : public IPluginV3, public IPluginV3OneCore, public IPluginV3OneBuild, public IPluginV3OneRuntime {
 public:
  BT6Plugin(int version, int heads, int head_dim, int templates, int export_h, int fused_coef, int out_int8 = 0,
            float out_scale = 0.f)
      : mVersion(version), mHeads(heads), mHeadDim(head_dim), mTemplates(templates), mExportH(export_h),
        mFusedCoef(fused_coef), mOutInt8(out_int8), mOutScale(out_scale) {
    mFields[0] = PluginField("heads", &mHeads, PluginFieldType::kINT32, 1);
    mFields[1] = PluginField("head_dim", &mHeadDim, PluginFieldType::kINT32, 1);
    mFields[2] = PluginField("templates", &mTemplates, PluginFieldType::kINT32, 1);
    mFields[3] = PluginField("export_h", &mExportH, PluginFieldType::kINT32, 1);
    mFields[4] = PluginField("fused_coef", &mFusedCoef, PluginFieldType::kINT32, 1);
    mFields[5] = PluginField("out_int8", &mOutInt8, PluginFieldType::kINT32, 1);
    mFields[6] = PluginField("out_scale", &mOutScale, PluginFieldType::kFLOAT32, 1);
    mFc.nbFields = version >= 3 ? 7 : version >= 2 ? 5 : 3;
    mFc.fields = mFields;
  }

  IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override {
    switch (type) {
      case PluginCapabilityType::kCORE: return static_cast<IPluginV3OneCore*>(this);
      case PluginCapabilityType::kBUILD: return static_cast<IPluginV3OneBuild*>(this);
      case PluginCapabilityType::kRUNTIME: return static_cast<IPluginV3OneRuntime*>(this);
    }
    return nullptr;
  }
  IPluginV3* clone() noexcept override {
    auto* p = new BT6Plugin(mVersion, mHeads, mHeadDim, mTemplates, mExportH, mFusedCoef, mOutInt8, mOutScale);
    p->mSms = mSms;
    p->mPrepared = mPrepared;
    return p;
  }

  char const* getPluginName() const noexcept override { return kAttnName; }
  char const* getPluginVersion() const noexcept override { return mVersion >= 3 ? "3" : mVersion >= 2 ? "2" : "1"; }
  char const* getPluginNamespace() const noexcept override { return ""; }

  int32_t configurePlugin(DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*, int32_t) noexcept override {
    return 0;
  }
  int32_t getNbOutputs() const noexcept override { return mExportH ? 2 : 1; }
  int32_t getOutputDataTypes(DataType* outputTypes, int32_t nbOutputs, DataType const*, int32_t) const noexcept override {
    if (nbOutputs != getNbOutputs()) return 1;
    for (int i = 0; i < nbOutputs; ++i) outputTypes[i] = DataType::kHALF;
    if (mOutInt8) outputTypes[0] = DataType::kINT8;
    return 0;
  }
  int32_t getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const*, int32_t, DimsExprs* outputs,
                          int32_t nbOutputs, IExprBuilder& eb) noexcept override {
    if (nbInputs != kAttnInputs || nbOutputs != getNbOutputs()) return 1;
    outputs[0].nbDims = 3;
    outputs[0].d[0] = inputs[0].d[0];
    outputs[0].d[1] = inputs[0].d[1];
    outputs[0].d[2] = eb.constant(mHeads * mHeadDim);
    if (mExportH) {
      outputs[1].nbDims = 4;
      outputs[1].d[0] = inputs[0].d[0];
      outputs[1].d[1] = eb.constant(mHeads);
      outputs[1].d[2] = eb.constant(64);
      outputs[1].d[3] = eb.constant(64);
    }
    return 0;
  }
  bool supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs,
                                 int32_t nbOutputs) noexcept override {
    if (nbInputs != kAttnInputs || nbOutputs != getNbOutputs()) return false;
    PluginTensorDesc const& d = inOut[pos].desc;
    if (d.format != TensorFormat::kLINEAR) return false;
    switch (pos) {
      case 2: return d.type == DataType::kINT32;
      case 6: return d.type == DataType::kFLOAT;
      case kAttnInputs: return d.type == (mOutInt8 ? DataType::kINT8 : DataType::kHALF);
      default: return d.type == DataType::kHALF;
    }
  }
  size_t getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t, DynamicPluginTensorDesc const*,
                          int32_t) const noexcept override {
    const int64_t maxB = inputs[0].max.d[0] > 0 ? inputs[0].max.d[0] : 1;
    return static_cast<size_t>(maxB) * mHeads * 4096 * sizeof(__half) + 256;
  }

  int32_t onShapeChange(PluginTensorDesc const*, int32_t, PluginTensorDesc const*, int32_t) noexcept override {
    return ensure_device();
  }
  int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const*, void const* const* inputs,
                  void* const* outputs, void* workspace, cudaStream_t stream) noexcept override {
    if (ensure_device() != 0) return 1;
    const int batch = static_cast<int>(inputDesc[0].dims.d[0]);
    auto* bias = reinterpret_cast<__half*>((reinterpret_cast<uintptr_t>(workspace) + 255) & ~uintptr_t(255));
    const int rc = bt6::launch_attention(
        outputs[0], static_cast<__half const*>(inputs[0]), static_cast<__half const*>(inputs[1]),
        static_cast<int const*>(inputs[2]), static_cast<__half const*>(inputs[3]), static_cast<__half const*>(inputs[4]),
        static_cast<__half const*>(inputs[5]), static_cast<float const*>(inputs[6]), static_cast<__half const*>(inputs[7]),
        mTemplates, bias, batch, mSms, stream, mExportH ? static_cast<__half*>(outputs[1]) : nullptr, mFusedCoef,
        mOutInt8 ? mOutScale : 0.f);
    if (rc != 0) {
      log_error(kAttnName, (std::string("kernel launch failed: ") + cudaGetErrorString(static_cast<cudaError_t>(rc))).c_str());
      return 1;
    }
    return 0;
  }
  // device setup at context creation, never inside a CUDA-graph capture of enqueue
  IPluginV3* attachToContext(IPluginResourceContext*) noexcept override {
    auto* p = static_cast<BT6Plugin*>(clone());
    p->ensure_device();
    return p;
  }
  PluginFieldCollection const* getFieldsToSerialize() noexcept override { return &mFc; }

  bool valid() const {
    return mHeads == bt6::kPluginHeads && mHeadDim == bt6::kPluginHeadDim && mTemplates == 8 &&
           (mExportH == 0 || mExportH == 1) && (mFusedCoef == 0 || mFusedCoef == 1) &&
           (mVersion >= 2 || (mExportH == 0 && mFusedCoef == 0)) && (mOutInt8 == 0 || mOutInt8 == 1) &&
           (mVersion >= 3 || mOutInt8 == 0) && (mOutInt8 == 0 || mOutScale > 0.f);
  }

 private:
  int ensure_device() {
    if (device_sms(mSms) != 0) return 1;
    if (!mPrepared) {
      if (bt6::prepare_attention() != 0) {
        log_error(kAttnName, "cudaFuncSetAttribute (dynamic shared memory) failed");
        return 1;
      }
      mPrepared = true;
    }
    return 0;
  }

  int mVersion;
  int32_t mHeads, mHeadDim, mTemplates, mExportH, mFusedCoef, mOutInt8;
  float mOutScale;
  int mSms = 0;
  bool mPrepared = false;
  PluginField mFields[7];
  PluginFieldCollection mFc{};
};

class BT6Creator : public IPluginCreatorV3One {
 public:
  explicit BT6Creator(int version) : mVersion(version) {
    mFields.emplace_back("heads", nullptr, PluginFieldType::kINT32, 1);
    mFields.emplace_back("head_dim", nullptr, PluginFieldType::kINT32, 1);
    mFields.emplace_back("templates", nullptr, PluginFieldType::kINT32, 1);
    if (version >= 2) {
      mFields.emplace_back("export_h", nullptr, PluginFieldType::kINT32, 1);
      mFields.emplace_back("fused_coef", nullptr, PluginFieldType::kINT32, 1);
    }
    if (version >= 3) {
      mFields.emplace_back("out_int8", nullptr, PluginFieldType::kINT32, 1);
      mFields.emplace_back("out_scale", nullptr, PluginFieldType::kFLOAT32, 1);
    }
    mFc.nbFields = static_cast<int32_t>(mFields.size());
    mFc.fields = mFields.data();
  }
  IPluginV3* createPlugin(char const*, PluginFieldCollection const* fc, TensorRTPhase) noexcept override {
    int heads = bt6::kPluginHeads, hd = bt6::kPluginHeadDim, templates = 8, export_h = 0, fused_coef = 0, out_int8 = 0;
    float out_scale = 0.f;
    for (int i = 0; fc && i < fc->nbFields; ++i) {
      PluginField const& f = fc->fields[i];
      if (!f.name || !f.data) continue;
      if (!strcmp(f.name, "heads")) heads = read_int(f);
      else if (!strcmp(f.name, "head_dim")) hd = read_int(f);
      else if (!strcmp(f.name, "templates")) templates = read_int(f);
      else if (!strcmp(f.name, "export_h") && mVersion >= 2) export_h = read_int(f);
      else if (!strcmp(f.name, "fused_coef") && mVersion >= 2) fused_coef = read_int(f);
      else if (!strcmp(f.name, "out_int8") && mVersion >= 3) out_int8 = read_int(f);
      else if (!strcmp(f.name, "out_scale") && mVersion >= 3) out_scale = read_float(f);
    }
    auto* p = new BT6Plugin(mVersion, heads, hd, templates, export_h, fused_coef, out_int8, out_scale);
    if (!p->valid()) {
      log_error(kAttnName, "unsupported attributes: this build serves heads 24, head_dim 32, templates 8, export_h 0/1 (v2), out_int8 0/1 with out_scale > 0 (v3)");
      delete p;
      return nullptr;
    }
    return p;
  }
  PluginFieldCollection const* getFieldNames() noexcept override { return &mFc; }
  char const* getPluginName() const noexcept override { return kAttnName; }
  char const* getPluginVersion() const noexcept override { return mVersion >= 3 ? "3" : mVersion >= 2 ? "2" : "1"; }
  char const* getPluginNamespace() const noexcept override { return ""; }

 private:
  int mVersion;
  std::vector<PluginField> mFields;
  PluginFieldCollection mFc{};
};

// ===================================================================================================== prologue
class ProPlugin : public IPluginV3, public IPluginV3OneCore, public IPluginV3OneBuild, public IPluginV3OneRuntime {
 public:
  ProPlugin() { mFc.nbFields = 0; mFc.fields = nullptr; }
  IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override {
    switch (type) {
      case PluginCapabilityType::kCORE: return static_cast<IPluginV3OneCore*>(this);
      case PluginCapabilityType::kBUILD: return static_cast<IPluginV3OneBuild*>(this);
      case PluginCapabilityType::kRUNTIME: return static_cast<IPluginV3OneRuntime*>(this);
    }
    return nullptr;
  }
  IPluginV3* clone() noexcept override {
    auto* p = new ProPlugin();
    p->mSms = mSms;
    return p;
  }
  char const* getPluginName() const noexcept override { return kProName; }
  char const* getPluginVersion() const noexcept override { return "1"; }
  char const* getPluginNamespace() const noexcept override { return ""; }
  int32_t configurePlugin(DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*, int32_t) noexcept override {
    return 0;
  }
  int32_t getNbOutputs() const noexcept override { return 3; }
  int32_t getOutputDataTypes(DataType* t, int32_t n, DataType const*, int32_t) const noexcept override {
    if (n != 3) return 1;
    t[0] = DataType::kINT32;
    t[1] = DataType::kHALF;
    t[2] = DataType::kHALF;
    return 0;
  }
  int32_t getOutputShapes(DimsExprs const* in, int32_t nbInputs, DimsExprs const*, int32_t, DimsExprs* out,
                          int32_t nbOutputs, IExprBuilder& eb) noexcept override {
    if (nbInputs != 2 || nbOutputs != 3) return 1;
    auto const* c64 = eb.constant(64);
    out[0].nbDims = 4; out[0].d[0] = in[0].d[0]; out[0].d[1] = c64; out[0].d[2] = c64; out[0].d[3] = eb.constant(2);
    out[1].nbDims = 3; out[1].d[0] = in[0].d[0]; out[1].d[1] = c64; out[1].d[2] = c64;
    out[2].nbDims = 4; out[2].d[0] = in[0].d[0]; out[2].d[1] = c64; out[2].d[2] = c64; out[2].d[3] = eb.constant(16);
    return 0;
  }
  bool supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs,
                                 int32_t nbOutputs) noexcept override {
    if (nbInputs != 2 || nbOutputs != 3) return false;
    PluginTensorDesc const& d = inOut[pos].desc;
    if (d.format != TensorFormat::kLINEAR) return false;
    switch (pos) {
      case 0: return d.type == DataType::kHALF && d.dims.nbDims == 3 && (d.dims.d[2] == 137 || d.dims.d[2] == -1);
      case 1: return d.type == DataType::kFLOAT;
      case 2: return d.type == DataType::kINT32;
      default: return d.type == DataType::kHALF;
    }
  }
  size_t getWorkspaceSize(DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*, int32_t) const noexcept override {
    return 0;
  }
  int32_t onShapeChange(PluginTensorDesc const* in, int32_t, PluginTensorDesc const*, int32_t) noexcept override {
    if (in[0].dims.d[1] != 64 || in[0].dims.d[2] != 137 || in[1].dims.d[0] != bt6::kPrologueTab32) {
      log_error(kProName, "expects squares [B, 64, 137] and a table of 8288 floats");
      return 1;
    }
    return device_sms(mSms);
  }
  int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const*, void const* const* inputs,
                  void* const* outputs, void*, cudaStream_t stream) noexcept override {
    if (device_sms(mSms) != 0) return 1;
    const int batch = static_cast<int>(inputDesc[0].dims.d[0]);
    const int rc = bt6::launch_prologue(static_cast<int*>(outputs[0]), static_cast<__half*>(outputs[1]),
                                        static_cast<__half*>(outputs[2]), static_cast<__half const*>(inputs[0]),
                                        static_cast<float const*>(inputs[1]), batch, mSms, stream);
    if (rc != 0) {
      log_error(kProName, (std::string("kernel launch failed: ") + cudaGetErrorString(static_cast<cudaError_t>(rc))).c_str());
      return 1;
    }
    return 0;
  }
  IPluginV3* attachToContext(IPluginResourceContext*) noexcept override {
    auto* p = static_cast<ProPlugin*>(clone());
    device_sms(p->mSms);
    return p;
  }
  PluginFieldCollection const* getFieldsToSerialize() noexcept override { return &mFc; }

 private:
  int mSms = 0;
  PluginFieldCollection mFc{};
};

class ProCreator : public IPluginCreatorV3One {
 public:
  ProCreator() { mFc.nbFields = 0; mFc.fields = nullptr; }
  IPluginV3* createPlugin(char const*, PluginFieldCollection const*, TensorRTPhase) noexcept override { return new ProPlugin(); }
  PluginFieldCollection const* getFieldNames() noexcept override { return &mFc; }
  char const* getPluginName() const noexcept override { return kProName; }
  char const* getPluginVersion() const noexcept override { return "1"; }
  char const* getPluginNamespace() const noexcept override { return ""; }

 private:
  PluginFieldCollection mFc{};
};


// ===================================================================================================== edge site
// BT6EdgeSite version "1": state FP16 [B, 64, 64, 16] cell-major + hraw FP16 [B, 24, 64, 64] + tab FP32 [3520]
//   -> state' FP16 [B, 64, 64, 16] cell-major (bt6_site_kernels.cu)
constexpr char const* kSiteName = "BT6EdgeSite";

class SitePlugin : public IPluginV3, public IPluginV3OneCore, public IPluginV3OneBuild, public IPluginV3OneRuntime {
 public:
  SitePlugin() { mFc.nbFields = 0; mFc.fields = nullptr; }
  IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override {
    switch (type) {
      case PluginCapabilityType::kCORE: return static_cast<IPluginV3OneCore*>(this);
      case PluginCapabilityType::kBUILD: return static_cast<IPluginV3OneBuild*>(this);
      case PluginCapabilityType::kRUNTIME: return static_cast<IPluginV3OneRuntime*>(this);
    }
    return nullptr;
  }
  IPluginV3* clone() noexcept override { return new SitePlugin(); }
  char const* getPluginName() const noexcept override { return kSiteName; }
  char const* getPluginVersion() const noexcept override { return "1"; }
  char const* getPluginNamespace() const noexcept override { return ""; }
  int32_t configurePlugin(DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*, int32_t) noexcept override {
    return 0;
  }
  int32_t getNbOutputs() const noexcept override { return 1; }
  int32_t getOutputDataTypes(DataType* t, int32_t n, DataType const*, int32_t) const noexcept override {
    if (n != 1) return 1;
    t[0] = DataType::kHALF;
    return 0;
  }
  int32_t getOutputShapes(DimsExprs const* in, int32_t nbInputs, DimsExprs const*, int32_t, DimsExprs* out,
                          int32_t nbOutputs, IExprBuilder&) noexcept override {
    if (nbInputs != 3 || nbOutputs != 1) return 1;
    out[0] = in[0];
    return 0;
  }
  bool supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs,
                                 int32_t nbOutputs) noexcept override {
    if (nbInputs != 3 || nbOutputs != 1) return false;
    PluginTensorDesc const& d = inOut[pos].desc;
    if (d.format != TensorFormat::kLINEAR) return false;
    return pos == 2 ? d.type == DataType::kFLOAT : d.type == DataType::kHALF;
  }
  size_t getWorkspaceSize(DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*, int32_t) const noexcept override {
    return 0;
  }
  int32_t onShapeChange(PluginTensorDesc const* in, int32_t, PluginTensorDesc const*, int32_t) noexcept override {
    if (in[0].dims.nbDims != 4 || in[0].dims.d[1] != 64 || in[0].dims.d[2] != 64 || in[0].dims.d[3] != 16 ||
        in[1].dims.nbDims != 4 || in[1].dims.d[1] != 24 || in[2].dims.d[0] != bt6::kSiteTab32) {
      log_error(kSiteName, "expects state [B, 64, 64, 16], hraw [B, 24, 64, 64] and a table of 3520 floats");
      return 1;
    }
    return 0;
  }
  int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const*, void const* const* inputs,
                  void* const* outputs, void*, cudaStream_t stream) noexcept override {
    const int batch = static_cast<int>(inputDesc[0].dims.d[0]);
    const int rc = bt6::launch_site(static_cast<__half*>(outputs[0]), static_cast<__half const*>(inputs[0]),
                                    static_cast<__half const*>(inputs[1]), static_cast<float const*>(inputs[2]), batch, stream);
    if (rc != 0) {
      log_error(kSiteName, (std::string("kernel launch failed: ") + cudaGetErrorString(static_cast<cudaError_t>(rc))).c_str());
      return 1;
    }
    return 0;
  }
  IPluginV3* attachToContext(IPluginResourceContext*) noexcept override { return clone(); }
  PluginFieldCollection const* getFieldsToSerialize() noexcept override { return &mFc; }

 private:
  PluginFieldCollection mFc{};
};

class SiteCreator : public IPluginCreatorV3One {
 public:
  SiteCreator() { mFc.nbFields = 0; mFc.fields = nullptr; }
  IPluginV3* createPlugin(char const*, PluginFieldCollection const*, TensorRTPhase) noexcept override { return new SitePlugin(); }
  PluginFieldCollection const* getFieldNames() noexcept override { return &mFc; }
  char const* getPluginName() const noexcept override { return kSiteName; }
  char const* getPluginVersion() const noexcept override { return "1"; }
  char const* getPluginNamespace() const noexcept override { return ""; }

 private:
  PluginFieldCollection mFc{};
};

// BT6GluQuant version "1": see the header comment.
constexpr char const* kGluName = "BT6GluQuant";

class GluPlugin : public IPluginV3, public IPluginV3OneCore, public IPluginV3OneBuild, public IPluginV3OneRuntime {
 public:
  GluPlugin(float gate_scale, float up_scale) : mGateScale(gate_scale), mUpScale(up_scale) {
    mFields[0] = PluginField("gate_scale", &mGateScale, PluginFieldType::kFLOAT32, 1);
    mFields[1] = PluginField("up_scale", &mUpScale, PluginFieldType::kFLOAT32, 1);
    mFc.nbFields = 2;
    mFc.fields = mFields;
  }
  IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override {
    switch (type) {
      case PluginCapabilityType::kCORE: return static_cast<IPluginV3OneCore*>(this);
      case PluginCapabilityType::kBUILD: return static_cast<IPluginV3OneBuild*>(this);
      case PluginCapabilityType::kRUNTIME: return static_cast<IPluginV3OneRuntime*>(this);
    }
    return nullptr;
  }
  IPluginV3* clone() noexcept override { return new GluPlugin(mGateScale, mUpScale); }
  char const* getPluginName() const noexcept override { return kGluName; }
  char const* getPluginVersion() const noexcept override { return "1"; }
  char const* getPluginNamespace() const noexcept override { return ""; }
  int32_t configurePlugin(DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*, int32_t) noexcept override {
    return 0;
  }
  int32_t getNbOutputs() const noexcept override { return 2; }
  int32_t getOutputDataTypes(DataType* t, int32_t n, DataType const*, int32_t) const noexcept override {
    if (n != 2) return 1;
    t[0] = DataType::kINT8;
    t[1] = DataType::kHALF;
    return 0;
  }
  int32_t getOutputShapes(DimsExprs const* in, int32_t nbInputs, DimsExprs const*, int32_t, DimsExprs* out,
                          int32_t nbOutputs, IExprBuilder& eb) noexcept override {
    if (nbInputs != 3 || nbOutputs != 2 || in[0].nbDims < 2) return 1;
    out[0] = in[0];
    out[1] = in[0];
    out[1].d[in[0].nbDims - 1] = eb.constant(1);
    return 0;
  }
  bool supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs,
                                 int32_t nbOutputs) noexcept override {
    if (nbInputs != 3 || nbOutputs != 2) return false;
    PluginTensorDesc const& d = inOut[pos].desc;
    if (d.format != TensorFormat::kLINEAR) return false;
    switch (pos) {
      case 0: case 1: case 3: return d.type == DataType::kINT8;
      default: return d.type == DataType::kHALF;
    }
  }
  size_t getWorkspaceSize(DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*, int32_t) const noexcept override {
    return 0;
  }
  int32_t onShapeChange(PluginTensorDesc const* in, int32_t, PluginTensorDesc const*, int32_t) noexcept override {
    const int nd = in[0].dims.nbDims;
    bool ok = nd >= 2 && in[1].dims.nbDims == nd && in[0].dims.d[nd - 1] == bt6::kGluChannels &&
              in[2].dims.nbDims == 1 && in[2].dims.d[0] == bt6::kGluChannels;
    for (int i = 0; ok && i < nd; ++i) ok = in[0].dims.d[i] == in[1].dims.d[i];
    if (!ok) {
      log_error(kGluName, "expects gate and up INT8 [..., 1536] of the same shape and a bias of 1536");
      return 1;
    }
    return 0;
  }
  int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const*, void const* const* inputs,
                  void* const* outputs, void*, cudaStream_t stream) noexcept override {
    int64_t rows = 1;
    for (int i = 0; i + 1 < inputDesc[0].dims.nbDims; ++i) rows *= inputDesc[0].dims.d[i];
    const int rc = bt6::launch_glu_quant(static_cast<int8_t*>(outputs[0]), static_cast<__half*>(outputs[1]),
                                         static_cast<int8_t const*>(inputs[0]), static_cast<int8_t const*>(inputs[1]),
                                         static_cast<__half const*>(inputs[2]), mGateScale, mUpScale,
                                         static_cast<int>(rows), stream);
    if (rc != 0) {
      log_error(kGluName, (std::string("kernel launch failed: ") + cudaGetErrorString(static_cast<cudaError_t>(rc))).c_str());
      return 1;
    }
    return 0;
  }
  IPluginV3* attachToContext(IPluginResourceContext*) noexcept override { return clone(); }
  PluginFieldCollection const* getFieldsToSerialize() noexcept override { return &mFc; }

 private:
  float mGateScale, mUpScale;
  PluginField mFields[2];
  PluginFieldCollection mFc{};
};

class GluCreator : public IPluginCreatorV3One {
 public:
  GluCreator() {
    mFields[0] = PluginField("gate_scale", nullptr, PluginFieldType::kFLOAT32, 1);
    mFields[1] = PluginField("up_scale", nullptr, PluginFieldType::kFLOAT32, 1);
    mFc.nbFields = 2;
    mFc.fields = mFields;
  }
  IPluginV3* createPlugin(char const*, PluginFieldCollection const* fc, TensorRTPhase) noexcept override {
    float gs = -1.f, us = -1.f;
    for (int i = 0; fc && i < fc->nbFields; ++i) {
      PluginField const& f = fc->fields[i];
      if (!f.name || !f.data) continue;
      if (!strcmp(f.name, "gate_scale")) gs = read_float(f);
      else if (!strcmp(f.name, "up_scale")) us = read_float(f);
    }
    if (!(gs > 0.f && us > 0.f)) {
      log_error(kGluName, "needs gate_scale > 0 and up_scale > 0");
      return nullptr;
    }
    return new GluPlugin(gs, us);
  }
  PluginFieldCollection const* getFieldNames() noexcept override { return &mFc; }
  char const* getPluginName() const noexcept override { return kGluName; }
  char const* getPluginVersion() const noexcept override { return "1"; }
  char const* getPluginNamespace() const noexcept override { return ""; }

 private:
  PluginField mFields[2];
  PluginFieldCollection mFc{};
};


// ===================================================================================================== k2 (INT8 Kovax BT6)
// BT6RowQuant version "1": input 0 x FP16 [..., 1024]; output 0 codes INT8 [..., 1024], output 1 r FP16 [..., 1]:
//   r = max(max_j |x_j|, eps) (FP16), codes = round(127 x / r) clamped to +-127. Attribute eps (FP32, default 1e-4).
// BT6SigGluQuant version "1": input 0 p FP16 [..., 2048] = [gate | up] (the FFN dense1 GEMM output with its bias),
//   m = sigmoid(gate) * up, then codes / r of m as BT6RowQuant; outputs codes INT8 [..., 1024], r FP16 [..., 1].
constexpr char const* kRowQuantName = "BT6RowQuant";
constexpr char const* kSigGluName = "BT6SigGluQuant";

class RowQuantPlugin : public IPluginV3, public IPluginV3OneCore, public IPluginV3OneBuild, public IPluginV3OneRuntime {
 public:
  RowQuantPlugin(int glu, float eps) : mGlu(glu), mEps(eps) {
    mFields[0] = PluginField("eps", &mEps, PluginFieldType::kFLOAT32, 1);
    mFc.nbFields = 1;
    mFc.fields = mFields;
  }
  IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override {
    switch (type) {
      case PluginCapabilityType::kCORE: return static_cast<IPluginV3OneCore*>(this);
      case PluginCapabilityType::kBUILD: return static_cast<IPluginV3OneBuild*>(this);
      case PluginCapabilityType::kRUNTIME: return static_cast<IPluginV3OneRuntime*>(this);
    }
    return nullptr;
  }
  IPluginV3* clone() noexcept override { return new RowQuantPlugin(mGlu, mEps); }
  char const* getPluginName() const noexcept override { return mGlu ? kSigGluName : kRowQuantName; }
  char const* getPluginVersion() const noexcept override { return "1"; }
  char const* getPluginNamespace() const noexcept override { return ""; }
  int32_t configurePlugin(DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*, int32_t) noexcept override {
    return 0;
  }
  int32_t getNbOutputs() const noexcept override { return 2; }
  int32_t getOutputDataTypes(DataType* t, int32_t n, DataType const*, int32_t) const noexcept override {
    if (n != 2) return 1;
    t[0] = DataType::kINT8;
    t[1] = DataType::kHALF;
    return 0;
  }
  int32_t getOutputShapes(DimsExprs const* in, int32_t nbInputs, DimsExprs const*, int32_t, DimsExprs* out,
                          int32_t nbOutputs, IExprBuilder& eb) noexcept override {
    if (nbInputs != 1 || nbOutputs != 2 || in[0].nbDims < 2) return 1;
    const int nd = in[0].nbDims;
    out[0] = in[0];
    out[0].d[nd - 1] = eb.constant(bt6::kRowQuantChannels);
    out[1] = in[0];
    out[1].d[nd - 1] = eb.constant(1);
    return 0;
  }
  bool supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs,
                                 int32_t nbOutputs) noexcept override {
    if (nbInputs != 1 || nbOutputs != 2) return false;
    PluginTensorDesc const& d = inOut[pos].desc;
    if (d.format != TensorFormat::kLINEAR) return false;
    return pos == 1 ? d.type == DataType::kINT8 : d.type == DataType::kHALF;
  }
  size_t getWorkspaceSize(DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*, int32_t) const noexcept override {
    return 0;
  }
  int32_t onShapeChange(PluginTensorDesc const* in, int32_t, PluginTensorDesc const*, int32_t) noexcept override {
    const int nd = in[0].dims.nbDims;
    if (nd < 2 || in[0].dims.d[nd - 1] != (mGlu ? 2 : 1) * bt6::kRowQuantChannels) {
      log_error(getPluginName(), mGlu ? "expects FP16 [..., 2048] = [gate | up]" : "expects FP16 [..., 1024]");
      return 1;
    }
    return 0;
  }
  int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const*, void const* const* inputs,
                  void* const* outputs, void*, cudaStream_t stream) noexcept override {
    int64_t rows = 1;
    for (int i = 0; i + 1 < inputDesc[0].dims.nbDims; ++i) rows *= inputDesc[0].dims.d[i];
    const int rc = bt6::launch_row_quant(static_cast<int8_t*>(outputs[0]), static_cast<__half*>(outputs[1]),
                                         static_cast<__half const*>(inputs[0]), mGlu, mEps, static_cast<int>(rows), stream);
    if (rc != 0) {
      log_error(getPluginName(), (std::string("kernel launch failed: ") + cudaGetErrorString(static_cast<cudaError_t>(rc))).c_str());
      return 1;
    }
    return 0;
  }
  IPluginV3* attachToContext(IPluginResourceContext*) noexcept override { return clone(); }
  PluginFieldCollection const* getFieldsToSerialize() noexcept override { return &mFc; }

 private:
  int mGlu;
  float mEps;
  PluginField mFields[1];
  PluginFieldCollection mFc{};
};

class RowQuantCreator : public IPluginCreatorV3One {
 public:
  explicit RowQuantCreator(int glu) : mGlu(glu) {
    mFields[0] = PluginField("eps", nullptr, PluginFieldType::kFLOAT32, 1);
    mFc.nbFields = 1;
    mFc.fields = mFields;
  }
  IPluginV3* createPlugin(char const*, PluginFieldCollection const* fc, TensorRTPhase) noexcept override {
    float eps = 1e-4f;
    for (int i = 0; fc && i < fc->nbFields; ++i) {
      PluginField const& f = fc->fields[i];
      if (f.name && f.data && !strcmp(f.name, "eps")) eps = read_float(f);
    }
    if (!(eps > 0.f)) {
      log_error(getPluginName(), "needs eps > 0");
      return nullptr;
    }
    return new RowQuantPlugin(mGlu, eps);
  }
  PluginFieldCollection const* getFieldNames() noexcept override { return &mFc; }
  char const* getPluginName() const noexcept override { return mGlu ? kSigGluName : kRowQuantName; }
  char const* getPluginVersion() const noexcept override { return "1"; }
  char const* getPluginNamespace() const noexcept override { return ""; }

 private:
  int mGlu;
  PluginField mFields[1];
  PluginFieldCollection mFc{};
};

// ===================================================================================================== decoder (round 4)
constexpr char const* kDecAttnName = "BT6DecAttn";
constexpr char const* kDecHeadName = "BT6DecHead";

class SimplePluginBase : public IPluginV3, public IPluginV3OneCore, public IPluginV3OneBuild, public IPluginV3OneRuntime {
 public:
  IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override {
    switch (type) {
      case PluginCapabilityType::kCORE: return static_cast<IPluginV3OneCore*>(this);
      case PluginCapabilityType::kBUILD: return static_cast<IPluginV3OneBuild*>(this);
      case PluginCapabilityType::kRUNTIME: return static_cast<IPluginV3OneRuntime*>(this);
    }
    return nullptr;
  }
  char const* getPluginVersion() const noexcept override { return "1"; }
  char const* getPluginNamespace() const noexcept override { return ""; }
  int32_t configurePlugin(DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*, int32_t) noexcept override {
    return 0;
  }
  size_t getWorkspaceSize(DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*, int32_t) const noexcept override {
    return 0;
  }
  IPluginV3* attachToContext(IPluginResourceContext*) noexcept override { return clone(); }
};

class DecAttnPlugin : public SimplePluginBase {
 public:
  DecAttnPlugin(int mode, int k_off, int v_off, float scale) : mMode(mode), mKOff(k_off), mVOff(v_off), mScale(scale) {
    mFields[0] = PluginField("mode", &mMode, PluginFieldType::kINT32, 1);
    mFields[1] = PluginField("k_off", &mKOff, PluginFieldType::kINT32, 1);
    mFields[2] = PluginField("v_off", &mVOff, PluginFieldType::kINT32, 1);
    mFields[3] = PluginField("scale", &mScale, PluginFieldType::kFLOAT32, 1);
    mFc.nbFields = 4;
    mFc.fields = mFields;
  }
  IPluginV3* clone() noexcept override { return new DecAttnPlugin(mMode, mKOff, mVOff, mScale); }
  char const* getPluginName() const noexcept override { return kDecAttnName; }
  int nIn() const { return mMode == 2 ? 4 : 2; }
  int32_t getNbOutputs() const noexcept override { return 1; }
  int32_t getOutputDataTypes(DataType* t, int32_t n, DataType const*, int32_t) const noexcept override {
    if (n != 1) return 1;
    t[0] = DataType::kHALF;
    return 0;
  }
  int32_t getOutputShapes(DimsExprs const* in, int32_t nbInputs, DimsExprs const*, int32_t, DimsExprs* out,
                          int32_t nbOutputs, IExprBuilder& eb) noexcept override {
    if (nbInputs != nIn() || nbOutputs != 1 || in[0].nbDims != 3) return 1;
    out[0] = in[0];
    out[0].d[2] = eb.constant(256);
    return 0;
  }
  bool supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs,
                                 int32_t nbOutputs) noexcept override {
    if (nbInputs != nIn() || nbOutputs != 1) return false;
    PluginTensorDesc const& d = inOut[pos].desc;
    if (d.format != TensorFormat::kLINEAR) return false;
    if (mMode == 2 && pos == 2) return d.type == DataType::kINT32;
    return d.type == DataType::kHALF;
  }
  int32_t onShapeChange(PluginTensorDesc const* in, int32_t nb, PluginTensorDesc const*, int32_t) noexcept override {
    bool ok = nb == nIn() && in[0].dims.nbDims == 3 && in[0].dims.d[1] == 80 && in[0].dims.d[2] % 8 == 0;
    if (ok && mMode == 0) {
      ok = in[1].dims.nbDims == 2 && in[1].dims.d[1] == 80 && mKOff % 8 == 0 && mVOff % 8 == 0 &&
           mKOff + 256 <= in[0].dims.d[2] && mVOff + 256 <= in[0].dims.d[2];
    } else if (ok) {
      ok = in[1].dims.nbDims == 3 && in[1].dims.d[1] == 64 && in[1].dims.d[2] % 8 == 0 && mKOff % 8 == 0 &&
           mVOff % 8 == 0 && mKOff + 256 <= in[1].dims.d[2] && mVOff + 256 <= in[1].dims.d[2];
      if (ok && mMode == 2)
        ok = in[2].dims.nbDims == 3 && in[2].dims.d[1] == 80 && in[2].dims.d[2] == 3 && in[3].dims.nbDims == 3 &&
             in[3].dims.d[0] == 2 && in[3].dims.d[1] == 7 && in[3].dims.d[2] == 256;
    }
    if (!ok) {
      log_error(kDecAttnName, "unexpected input shapes (80 tokens, 8 heads x 32, 64 squares)");
      return 1;
    }
    return 0;
  }
  int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const*, void const* const* inputs,
                  void* const* outputs, void*, cudaStream_t stream) noexcept override {
    const int batch = static_cast<int>(inputDesc[0].dims.d[0]);
    const int qs = static_cast<int>(inputDesc[0].dims.d[2]);
    const __half* q = static_cast<__half const*>(inputs[0]);
    int rc;
    if (mMode == 0) {
      rc = bt6::launch_dec_attn(static_cast<__half*>(outputs[0]), q, qs, q, qs, mKOff, mVOff,
                                static_cast<__half const*>(inputs[1]), nullptr, nullptr, mScale, 0, batch, stream);
    } else {
      const int kvs = static_cast<int>(inputDesc[1].dims.d[2]);
      rc = bt6::launch_dec_attn(static_cast<__half*>(outputs[0]), q, qs, static_cast<__half const*>(inputs[1]), kvs, mKOff,
                                mVOff, nullptr, mMode == 2 ? static_cast<int const*>(inputs[2]) : nullptr,
                                mMode == 2 ? static_cast<__half const*>(inputs[3]) : nullptr, mScale, mMode, batch, stream);
    }
    if (rc != 0) {
      log_error(kDecAttnName, (std::string("kernel launch failed: ") + cudaGetErrorString(static_cast<cudaError_t>(rc))).c_str());
      return 1;
    }
    return 0;
  }
  PluginFieldCollection const* getFieldsToSerialize() noexcept override { return &mFc; }

 private:
  int32_t mMode, mKOff, mVOff;
  float mScale;
  PluginField mFields[4];
  PluginFieldCollection mFc{};
};

class DecAttnCreator : public IPluginCreatorV3One {
 public:
  DecAttnCreator() {
    mFields[0] = PluginField("mode", nullptr, PluginFieldType::kINT32, 1);
    mFields[1] = PluginField("k_off", nullptr, PluginFieldType::kINT32, 1);
    mFields[2] = PluginField("v_off", nullptr, PluginFieldType::kINT32, 1);
    mFields[3] = PluginField("scale", nullptr, PluginFieldType::kFLOAT32, 1);
    mFc.nbFields = 4;
    mFc.fields = mFields;
  }
  IPluginV3* createPlugin(char const*, PluginFieldCollection const* fc, TensorRTPhase) noexcept override {
    int mode = -1, ko = -1, vo = -1;
    float sc = -1.f;
    for (int i = 0; fc && i < fc->nbFields; ++i) {
      PluginField const& f = fc->fields[i];
      if (!f.name || !f.data) continue;
      if (!strcmp(f.name, "mode")) mode = read_int(f);
      else if (!strcmp(f.name, "k_off")) ko = read_int(f);
      else if (!strcmp(f.name, "v_off")) vo = read_int(f);
      else if (!strcmp(f.name, "scale")) sc = read_float(f);
    }
    if (mode < 0 || mode > 2 || ko < 0 || vo < 0 || !(sc > 0.f)) {
      log_error(kDecAttnName, "needs mode 0..2, k_off >= 0, v_off >= 0 and scale > 0");
      return nullptr;
    }
    return new DecAttnPlugin(mode, ko, vo, sc);
  }
  PluginFieldCollection const* getFieldNames() noexcept override { return &mFc; }
  char const* getPluginName() const noexcept override { return kDecAttnName; }
  char const* getPluginVersion() const noexcept override { return "1"; }
  char const* getPluginNamespace() const noexcept override { return ""; }

 private:
  PluginField mFields[4];
  PluginFieldCollection mFc{};
};

class DecHeadPlugin : public SimplePluginBase {
 public:
  explicit DecHeadPlugin(float eps) : mEps(eps) {
    mFields[0] = PluginField("eps", &mEps, PluginFieldType::kFLOAT32, 1);
    mFc.nbFields = 1;
    mFc.fields = mFields;
  }
  IPluginV3* clone() noexcept override { return new DecHeadPlugin(mEps); }
  char const* getPluginName() const noexcept override { return kDecHeadName; }
  int32_t getNbOutputs() const noexcept override { return 2; }
  int32_t getOutputDataTypes(DataType* t, int32_t n, DataType const*, int32_t) const noexcept override {
    if (n != 2) return 1;
    t[0] = DataType::kHALF;
    t[1] = DataType::kHALF;
    return 0;
  }
  int32_t getOutputShapes(DimsExprs const* in, int32_t nbInputs, DimsExprs const*, int32_t, DimsExprs* out,
                          int32_t nbOutputs, IExprBuilder& eb) noexcept override {
    if (nbInputs != 5 || nbOutputs != 2 || in[0].nbDims != 3) return 1;
    out[0].nbDims = 2;
    out[0].d[0] = in[0].d[0];
    out[0].d[1] = eb.constant(1858);
    out[1].nbDims = 2;
    out[1].d[0] = in[0].d[0];
    out[1].d[1] = eb.constant(512);
    return 0;
  }
  bool supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs,
                                 int32_t nbOutputs) noexcept override {
    if (nbInputs != 5 || nbOutputs != 2) return false;
    PluginTensorDesc const& d = inOut[pos].desc;
    if (d.format != TensorFormat::kLINEAR) return false;
    switch (pos) {
      case 2: case 4: return d.type == DataType::kINT32;
      case 3: return d.type == DataType::kFLOAT;
      default: return d.type == DataType::kHALF;
    }
  }
  int32_t onShapeChange(PluginTensorDesc const* in, int32_t nb, PluginTensorDesc const*, int32_t) noexcept override {
    const bool ok = nb == 5 && in[0].dims.nbDims == 3 && in[0].dims.d[1] == 80 && in[0].dims.d[2] == 256 &&
                    in[1].dims.nbDims == 2 && in[1].dims.d[1] == 80 && in[2].dims.nbDims == 2 && in[2].dims.d[1] == 80 &&
                    in[3].dims.nbDims == 1 && in[3].dims.d[0] == bt6::kDecHeadTab32 && in[4].dims.nbDims == 1 &&
                    in[4].dims.d[0] == 1858;
    if (!ok) {
      log_error(kDecHeadName, "expects x [B, 80, 256], key bias [B, 80], sel [B, 80], tab [3138], slot [1858]");
      return 1;
    }
    return 0;
  }
  int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const*, void const* const* inputs,
                  void* const* outputs, void*, cudaStream_t stream) noexcept override {
    const int batch = static_cast<int>(inputDesc[0].dims.d[0]);
    const int rc = bt6::launch_dec_head(static_cast<__half*>(outputs[0]), static_cast<__half*>(outputs[1]),
                                        static_cast<__half const*>(inputs[0]), static_cast<__half const*>(inputs[1]),
                                        static_cast<int const*>(inputs[2]), static_cast<float const*>(inputs[3]),
                                        static_cast<int const*>(inputs[4]), mEps, batch, stream);
    if (rc != 0) {
      log_error(kDecHeadName, (std::string("kernel launch failed: ") + cudaGetErrorString(static_cast<cudaError_t>(rc))).c_str());
      return 1;
    }
    return 0;
  }
  PluginFieldCollection const* getFieldsToSerialize() noexcept override { return &mFc; }

 private:
  float mEps;
  PluginField mFields[1];
  PluginFieldCollection mFc{};
};

class DecHeadCreator : public IPluginCreatorV3One {
 public:
  DecHeadCreator() {
    mFields[0] = PluginField("eps", nullptr, PluginFieldType::kFLOAT32, 1);
    mFc.nbFields = 1;
    mFc.fields = mFields;
  }
  IPluginV3* createPlugin(char const*, PluginFieldCollection const* fc, TensorRTPhase) noexcept override {
    float eps = -1.f;
    for (int i = 0; fc && i < fc->nbFields; ++i) {
      PluginField const& f = fc->fields[i];
      if (f.name && f.data && !strcmp(f.name, "eps")) eps = read_float(f);
    }
    if (!(eps > 0.f)) {
      log_error(kDecHeadName, "needs eps > 0");
      return nullptr;
    }
    return new DecHeadPlugin(eps);
  }
  PluginFieldCollection const* getFieldNames() noexcept override { return &mFc; }
  char const* getPluginName() const noexcept override { return kDecHeadName; }
  char const* getPluginVersion() const noexcept override { return "1"; }
  char const* getPluginNamespace() const noexcept override { return ""; }

 private:
  PluginField mFields[1];
  PluginFieldCollection mFc{};
};

// ===================================================================================================== k1: Kovax BT6
constexpr char const* kEgtName = "BT6EgtAttention";
constexpr char const* kPlanesProName = "BT6PlanesPrologue";
constexpr int kEgtInputs = 6;

class EgtPlugin : public SimplePluginBase {
 public:
  EgtPlugin(int heads, int head_dim, int export_h, int out_int8, float out_scale)
      : mHeads(heads), mHeadDim(head_dim), mExportH(export_h), mOutInt8(out_int8), mOutScale(out_scale) {
    mFields[0] = PluginField("heads", &mHeads, PluginFieldType::kINT32, 1);
    mFields[1] = PluginField("head_dim", &mHeadDim, PluginFieldType::kINT32, 1);
    mFields[2] = PluginField("export_h", &mExportH, PluginFieldType::kINT32, 1);
    mFields[3] = PluginField("out_int8", &mOutInt8, PluginFieldType::kINT32, 1);
    mFields[4] = PluginField("out_scale", &mOutScale, PluginFieldType::kFLOAT32, 1);
    mFc.nbFields = 5;
    mFc.fields = mFields;
  }
  IPluginV3* clone() noexcept override {
    auto* p = new EgtPlugin(mHeads, mHeadDim, mExportH, mOutInt8, mOutScale);
    p->mSms = mSms;
    p->mPrepared = mPrepared;
    return p;
  }
  char const* getPluginName() const noexcept override { return kEgtName; }
  int32_t getNbOutputs() const noexcept override { return mExportH ? 2 : 1; }
  int32_t getOutputDataTypes(DataType* t, int32_t n, DataType const*, int32_t) const noexcept override {
    if (n != getNbOutputs()) return 1;
    for (int i = 0; i < n; ++i) t[i] = DataType::kHALF;
    if (mOutInt8) t[0] = DataType::kINT8;
    return 0;
  }
  int32_t getOutputShapes(DimsExprs const* in, int32_t nbInputs, DimsExprs const*, int32_t, DimsExprs* out,
                          int32_t nbOutputs, IExprBuilder& eb) noexcept override {
    if (nbInputs != kEgtInputs || nbOutputs != getNbOutputs()) return 1;
    out[0].nbDims = 3;
    out[0].d[0] = in[0].d[0];
    out[0].d[1] = in[0].d[1];
    out[0].d[2] = eb.constant(mHeads * mHeadDim);
    if (mExportH) {
      out[1].nbDims = 4;
      out[1].d[0] = in[0].d[0];
      out[1].d[1] = eb.constant(mHeads);
      out[1].d[2] = eb.constant(64);
      out[1].d[3] = eb.constant(64);
    }
    return 0;
  }
  bool supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs,
                                 int32_t nbOutputs) noexcept override {
    if (nbInputs != kEgtInputs || nbOutputs != getNbOutputs()) return false;
    PluginTensorDesc const& d = inOut[pos].desc;
    if (d.format != TensorFormat::kLINEAR) return false;
    switch (pos) {
      case 1: return d.type == DataType::kINT32;
      case 5: return d.type == DataType::kFLOAT;
      case kEgtInputs: return d.type == (mOutInt8 ? DataType::kINT8 : DataType::kHALF);
      default: return d.type == DataType::kHALF;
    }
  }
  int32_t onShapeChange(PluginTensorDesc const* in, int32_t nb, PluginTensorDesc const*, int32_t) noexcept override {
    const bool ok = nb == kEgtInputs && in[0].dims.nbDims == 3 && in[0].dims.d[1] == 64 &&
                    in[0].dims.d[2] == bt6::kEgtQkvgWidth && in[1].dims.nbDims == 4 && in[1].dims.d[3] == 2 &&
                    in[3].dims.nbDims == 4 && in[3].dims.d[3] == 16 && in[4].dims.d[0] == bt6::kEgtTab16 &&
                    in[5].dims.d[0] == bt6::kEgtTab32;
    if (!ok) {
      log_error(kEgtName, "expects qkvg [B, 64, 4096], edges [B, 64, 64, 2], state [B, 64, 64, 16] and the k1 tables");
      return 1;
    }
    return ensure_device();
  }
  int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const*, void const* const* inputs,
                  void* const* outputs, void*, cudaStream_t stream) noexcept override {
    if (ensure_device() != 0) return 1;
    const int batch = static_cast<int>(inputDesc[0].dims.d[0]);
    const int rc = bt6::launch_egt_attention(
        outputs[0], static_cast<__half const*>(inputs[0]), static_cast<int const*>(inputs[1]),
        static_cast<__half const*>(inputs[2]), static_cast<__half const*>(inputs[3]), static_cast<__half const*>(inputs[4]),
        static_cast<float const*>(inputs[5]), batch, mSms, stream, mExportH ? static_cast<__half*>(outputs[1]) : nullptr,
        mOutInt8 ? mOutScale : 0.f);
    if (rc != 0) {
      log_error(kEgtName, (std::string("kernel launch failed: ") + cudaGetErrorString(static_cast<cudaError_t>(rc))).c_str());
      return 1;
    }
    return 0;
  }
  IPluginV3* attachToContext(IPluginResourceContext*) noexcept override {
    auto* p = static_cast<EgtPlugin*>(clone());
    p->ensure_device();
    return p;
  }
  PluginFieldCollection const* getFieldsToSerialize() noexcept override { return &mFc; }
  bool valid() const {
    return mHeads == bt6::kEgtHeads && mHeadDim == bt6::kEgtHeadDim && (mExportH == 0 || mExportH == 1) &&
           (mOutInt8 == 0 || mOutInt8 == 1) && (mOutInt8 == 0 || mOutScale > 0.f);
  }

 private:
  int ensure_device() {
    if (device_sms(mSms) != 0) return 1;
    if (!mPrepared) {
      if (bt6::prepare_egt_attention() != 0) {
        log_error(kEgtName, "cudaFuncSetAttribute (dynamic shared memory) failed");
        return 1;
      }
      mPrepared = true;
    }
    return 0;
  }
  int32_t mHeads, mHeadDim, mExportH, mOutInt8;
  float mOutScale;
  int mSms = 0;
  bool mPrepared = false;
  PluginField mFields[5];
  PluginFieldCollection mFc{};
};

class EgtCreator : public IPluginCreatorV3One {
 public:
  EgtCreator() {
    mFields[0] = PluginField("heads", nullptr, PluginFieldType::kINT32, 1);
    mFields[1] = PluginField("head_dim", nullptr, PluginFieldType::kINT32, 1);
    mFields[2] = PluginField("export_h", nullptr, PluginFieldType::kINT32, 1);
    mFields[3] = PluginField("out_int8", nullptr, PluginFieldType::kINT32, 1);
    mFields[4] = PluginField("out_scale", nullptr, PluginFieldType::kFLOAT32, 1);
    mFc.nbFields = 5;
    mFc.fields = mFields;
  }
  IPluginV3* createPlugin(char const*, PluginFieldCollection const* fc, TensorRTPhase) noexcept override {
    int heads = bt6::kEgtHeads, hd = bt6::kEgtHeadDim, export_h = 0, out_int8 = 0;
    float out_scale = 0.f;
    for (int i = 0; fc && i < fc->nbFields; ++i) {
      PluginField const& f = fc->fields[i];
      if (!f.name || !f.data) continue;
      if (!strcmp(f.name, "heads")) heads = read_int(f);
      else if (!strcmp(f.name, "head_dim")) hd = read_int(f);
      else if (!strcmp(f.name, "export_h")) export_h = read_int(f);
      else if (!strcmp(f.name, "out_int8")) out_int8 = read_int(f);
      else if (!strcmp(f.name, "out_scale")) out_scale = read_float(f);
    }
    auto* p = new EgtPlugin(heads, hd, export_h, out_int8, out_scale);
    if (!p->valid()) {
      log_error(kEgtName, "unsupported attributes: this build serves heads 32, head_dim 32, export_h 0/1, out_int8 0/1 with out_scale > 0");
      delete p;
      return nullptr;
    }
    return p;
  }
  PluginFieldCollection const* getFieldNames() noexcept override { return &mFc; }
  char const* getPluginName() const noexcept override { return kEgtName; }
  char const* getPluginVersion() const noexcept override { return "1"; }
  char const* getPluginNamespace() const noexcept override { return ""; }

 private:
  PluginField mFields[5];
  PluginFieldCollection mFc{};
};

class PlanesProPlugin : public SimplePluginBase {
 public:
  PlanesProPlugin(float eps, int e0_fp32) : mEps(eps), mE0F(e0_fp32) {
    mFields[0] = PluginField("eps", &mEps, PluginFieldType::kFLOAT32, 1);
    mFields[1] = PluginField("e0_fp32", &mE0F, PluginFieldType::kINT32, 1);
    mFc.nbFields = 2;
    mFc.fields = mFields;
  }
  IPluginV3* clone() noexcept override {
    auto* p = new PlanesProPlugin(mEps, mE0F);
    p->mSms = mSms;
    return p;
  }
  char const* getPluginName() const noexcept override { return kPlanesProName; }
  int32_t getNbOutputs() const noexcept override { return mE0F ? 5 : 4; }
  int32_t getOutputDataTypes(DataType* t, int32_t n, DataType const*, int32_t) const noexcept override {
    if (n != getNbOutputs()) return 1;
    if (mE0F) t[4] = DataType::kFLOAT;
    t[0] = DataType::kINT32;
    t[1] = DataType::kHALF;
    t[2] = DataType::kHALF;
    t[3] = DataType::kFLOAT;
    return 0;
  }
  int32_t getOutputShapes(DimsExprs const* in, int32_t nbInputs, DimsExprs const*, int32_t, DimsExprs* out,
                          int32_t nbOutputs, IExprBuilder& eb) noexcept override {
    if (nbInputs != 2 || nbOutputs != getNbOutputs()) return 1;
    auto const* c64 = eb.constant(64);
    if (mE0F) { out[4].nbDims = 4; out[4].d[0] = in[0].d[0]; out[4].d[1] = eb.constant(16); out[4].d[2] = c64; out[4].d[3] = c64; }
    out[0].nbDims = 4; out[0].d[0] = in[0].d[0]; out[0].d[1] = c64; out[0].d[2] = c64; out[0].d[3] = eb.constant(2);
    out[1].nbDims = 3; out[1].d[0] = in[0].d[0]; out[1].d[1] = c64; out[1].d[2] = c64;
    out[2].nbDims = 4; out[2].d[0] = in[0].d[0]; out[2].d[1] = c64; out[2].d[2] = c64; out[2].d[3] = eb.constant(16);
    out[3].nbDims = 3; out[3].d[0] = in[0].d[0]; out[3].d[1] = c64; out[3].d[2] = c64;
    return 0;
  }
  bool supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs,
                                 int32_t nbOutputs) noexcept override {
    if (nbInputs != 2 || nbOutputs != getNbOutputs()) return false;
    PluginTensorDesc const& d = inOut[pos].desc;
    if (d.format != TensorFormat::kLINEAR) return false;
    switch (pos) {
      case 0: case 1: case 5: case 6: return d.type == DataType::kFLOAT;
      case 2: return d.type == DataType::kINT32;
      default: return d.type == DataType::kHALF;
    }
  }
  int32_t onShapeChange(PluginTensorDesc const* in, int32_t, PluginTensorDesc const*, int32_t) noexcept override {
    int64_t per = 1;
    for (int i = 1; i < in[0].dims.nbDims; ++i) per *= in[0].dims.d[i];
    if (per != 112 * 64 || in[0].dims.d[1] != 112 || in[1].dims.nbDims != 1 || in[1].dims.d[0] != bt6::kPlanesPrologueTab32) {
      log_error(kPlanesProName, "expects planes [B, 112, 8, 8] or [B, 112, 64] and the k1 prologue table");
      return 1;
    }
    return device_sms(mSms);
  }
  int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const*, void const* const* inputs,
                  void* const* outputs, void*, cudaStream_t stream) noexcept override {
    if (device_sms(mSms) != 0) return 1;
    const int batch = static_cast<int>(inputDesc[0].dims.d[0]);
    const int rc = bt6::launch_prologue_planes(static_cast<int*>(outputs[0]), static_cast<__half*>(outputs[1]),
                                               static_cast<__half*>(outputs[2]), static_cast<float*>(outputs[3]),
                                               static_cast<float const*>(inputs[0]), static_cast<float const*>(inputs[1]),
                                               batch, mSms, mEps, stream, mE0F ? static_cast<float*>(outputs[4]) : nullptr);
    if (rc != 0) {
      log_error(kPlanesProName, (std::string("kernel launch failed: ") + cudaGetErrorString(static_cast<cudaError_t>(rc))).c_str());
      return 1;
    }
    return 0;
  }
  IPluginV3* attachToContext(IPluginResourceContext*) noexcept override {
    auto* p = static_cast<PlanesProPlugin*>(clone());
    device_sms(p->mSms);
    return p;
  }
  PluginFieldCollection const* getFieldsToSerialize() noexcept override { return &mFc; }

 private:
  float mEps;
  int32_t mE0F;
  int mSms = 0;
  PluginField mFields[2];
  PluginFieldCollection mFc{};
};

class PlanesProCreator : public IPluginCreatorV3One {
 public:
  PlanesProCreator() {
    mFields[0] = PluginField("eps", nullptr, PluginFieldType::kFLOAT32, 1);
    mFields[1] = PluginField("e0_fp32", nullptr, PluginFieldType::kINT32, 1);
    mFc.nbFields = 2;
    mFc.fields = mFields;
  }
  IPluginV3* createPlugin(char const*, PluginFieldCollection const* fc, TensorRTPhase) noexcept override {
    float eps = 1e-6f;
    int e0f = 0;
    for (int i = 0; fc && i < fc->nbFields; ++i) {
      PluginField const& f = fc->fields[i];
      if (!f.name || !f.data) continue;
      if (!strcmp(f.name, "eps")) eps = read_float(f);
      else if (!strcmp(f.name, "e0_fp32")) e0f = read_int(f);
    }
    if (!(eps > 0.f) || (e0f != 0 && e0f != 1)) {
      log_error(kPlanesProName, "needs eps > 0 and e0_fp32 0 / 1");
      return nullptr;
    }
    return new PlanesProPlugin(eps, e0f);
  }
  PluginFieldCollection const* getFieldNames() noexcept override { return &mFc; }
  char const* getPluginName() const noexcept override { return kPlanesProName; }
  char const* getPluginVersion() const noexcept override { return "1"; }
  char const* getPluginNamespace() const noexcept override { return ""; }

 private:
  PluginField mFields[2];
  PluginFieldCollection mFc{};
};

}  // namespace

#if defined(_WIN32)
#define BT6_EXPORT extern "C" __declspec(dllexport)
#else
#define BT6_EXPORT extern "C" __attribute__((visibility("default")))
#endif

// k1: BT6TripletSite v1 (bt6_triplet_site_plugin.cpp, namespace bt6t; the triplet agent's fused edge site)
IPluginCreatorInterface* bt6t_triplet_site_creator();
void bt6t_set_logger_finder(ILoggerFinder* finder);

BT6_EXPORT void setLoggerFinder(ILoggerFinder* finder) {
  gLoggerFinder = finder;
  bt6t_set_logger_finder(finder);
}

BT6_EXPORT IPluginCreatorInterface* const* getCreators(int32_t& nbCreators) {
  static BT6Creator v1(1), v2(2), v3(3);
  static ProCreator pro;
  static SiteCreator site;
  static GluCreator glu;
  static DecAttnCreator dec_attn;
  static DecHeadCreator dec_head;
  static EgtCreator egt;
  static PlanesProCreator planes_pro;
  static RowQuantCreator row_quant(0), sig_glu_quant(1);
  static IPluginCreatorInterface* const list[] = {&v1, &v2, &v3, &pro, &site, &glu, &dec_attn, &dec_head, &egt, &planes_pro,
                                                  bt6t_triplet_site_creator(), &row_quant, &sig_glu_quant};
  nbCreators = 13;
  return list;
}

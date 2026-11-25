// BT6TripletSite version "1": TensorRT 11.3 IPluginV3 for one edge site of the BT6 flagship WITH the triplet branch
// (bt6_triplet_site.cu). Self-contained: drop this file, bt6_triplet_site.cu and bt6_triplet_site.h into a plugin library
// and add bt6t_triplet_site_creator() to its getCreators() list (and forward its ILoggerFinder with
// bt6t_set_logger_finder()).
//
// Inputs (linear, batch dimension B dynamic; the kernels have no other shape freedom):
//   0 state  FP16  [B, 64, 64, 16]   the edge state e, cell-major (channel innermost) -- r4 BT6EdgeSite's layout
//   1 H      FP16  [B, heads, 64, 64] the layer's attention logits H = (scores + read) * (1 + door), BEFORE softmax
//   2 tab    FP32  [16 * heads + 4432] (4944 at 32 heads): see triplet_site_tables.py
// Output: 0 state' FP16 [B, 64, 64, 16] cell-major.
// Fields (all optional): heads INT32 (default 32; 24 also compiled), contraction INT32 (0 = path, the flagship; 1 = ag).
// Workspace: 160 bytes per cell (640 KB per position).
#include <NvInferRuntime.h>

#include <cstdio>
#include <cstring>
#include <string>

#include "bt6_triplet_site.h"

using namespace nvinfer1;

namespace {

constexpr char const* kName = "BT6TripletSite";
ILoggerFinder* gFinder = nullptr;

void log_err(char const* msg) {
  std::string m = std::string("[") + kName + "] " + msg;
  if (gFinder) {
    if (ILogger* l = gFinder->findLogger()) {
      l->log(ILogger::Severity::kERROR, m.c_str());
      return;
    }
  }
  fprintf(stderr, "%s\n", m.c_str());
}

int field_int(PluginField const& f, int fallback) {
  if (!f.data) return fallback;
  switch (f.type) {
    case PluginFieldType::kINT32: return *static_cast<int32_t const*>(f.data);
    case PluginFieldType::kINT64: return static_cast<int>(*static_cast<int64_t const*>(f.data));
    case PluginFieldType::kINT16: return *static_cast<int16_t const*>(f.data);
    case PluginFieldType::kINT8: return *static_cast<int8_t const*>(f.data);
    default: return fallback;
  }
}

class TripletSitePlugin : public IPluginV3, public IPluginV3OneCore, public IPluginV3OneBuild, public IPluginV3OneRuntime {
 public:
  TripletSitePlugin(int heads, int contraction) : mHeads(heads), mContraction(contraction) {
    mFields[0] = PluginField("heads", &mHeads, PluginFieldType::kINT32, 1);
    mFields[1] = PluginField("contraction", &mContraction, PluginFieldType::kINT32, 1);
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
  IPluginV3* clone() noexcept override { return new TripletSitePlugin(mHeads, mContraction); }
  char const* getPluginName() const noexcept override { return kName; }
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
  size_t getWorkspaceSize(DynamicPluginTensorDesc const* in, int32_t, DynamicPluginTensorDesc const*, int32_t) const noexcept override {
    return bt6t::triplet_site_workspace(static_cast<int>(in[0].max.d[0]));
  }
  int32_t onShapeChange(PluginTensorDesc const* in, int32_t nb, PluginTensorDesc const*, int32_t) noexcept override {
    if (nb != 3 || in[0].dims.nbDims != 4 || in[0].dims.d[1] != 64 || in[0].dims.d[2] != 64 || in[0].dims.d[3] != 16 ||
        in[1].dims.nbDims != 4 || in[1].dims.d[0] != in[0].dims.d[0] || in[1].dims.d[1] != mHeads ||
        in[1].dims.d[2] != 64 || in[1].dims.d[3] != 64 || in[2].dims.nbDims != 1 ||
        in[2].dims.d[0] != bt6t::triplet_tab_size(mHeads)) {
      log_err("expects state [B, 64, 64, 16], H [B, heads, 64, 64] and a table of 16 * heads + 4432 floats");
      return 1;
    }
    return 0;
  }
  int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const*, void const* const* inputs,
                  void* const* outputs, void* workspace, cudaStream_t stream) noexcept override {
    if (!mPrepared) {
      if (bt6t::prepare_triplet_site() != 0) {
        log_err("cudaFuncSetAttribute (dynamic shared memory) failed");
        return 1;
      }
      mPrepared = true;
    }
    const int batch = static_cast<int>(inputDesc[0].dims.d[0]);
    const int rc = bt6t::launch_triplet_site(static_cast<__half*>(outputs[0]), static_cast<__half const*>(inputs[0]),
                                             static_cast<__half const*>(inputs[1]), static_cast<float const*>(inputs[2]),
                                             mHeads, mContraction, workspace, batch, stream);
    if (rc != 0) {
      log_err((std::string("kernel launch failed: ") + cudaGetErrorString(static_cast<cudaError_t>(rc))).c_str());
      return 1;
    }
    return 0;
  }
  IPluginV3* attachToContext(IPluginResourceContext*) noexcept override { return clone(); }
  PluginFieldCollection const* getFieldsToSerialize() noexcept override { return &mFc; }

 private:
  int32_t mHeads, mContraction;
  bool mPrepared = false;
  PluginField mFields[2];
  PluginFieldCollection mFc{};
};

class TripletSiteCreator : public IPluginCreatorV3One {
 public:
  TripletSiteCreator() {
    mFields[0] = PluginField("heads", nullptr, PluginFieldType::kINT32, 1);
    mFields[1] = PluginField("contraction", nullptr, PluginFieldType::kINT32, 1);
    mFc.nbFields = 2;
    mFc.fields = mFields;
  }
  IPluginV3* createPlugin(char const*, PluginFieldCollection const* fc, TensorRTPhase) noexcept override {
    int heads = 32, contraction = 0;
    for (int i = 0; fc && i < fc->nbFields; ++i) {
      PluginField const& f = fc->fields[i];
      if (!f.name) continue;
      if (!strcmp(f.name, "heads")) heads = field_int(f, heads);
      else if (!strcmp(f.name, "contraction")) contraction = field_int(f, contraction);
    }
    if ((heads != 24 && heads != 32) || (contraction != 0 && contraction != 1)) {
      log_err("needs heads 24 or 32 and contraction 0 (path) or 1 (ag)");
      return nullptr;
    }
    return new TripletSitePlugin(heads, contraction);
  }
  PluginFieldCollection const* getFieldNames() noexcept override { return &mFc; }
  char const* getPluginName() const noexcept override { return kName; }
  char const* getPluginVersion() const noexcept override { return "1"; }
  char const* getPluginNamespace() const noexcept override { return ""; }

 private:
  PluginField mFields[2];
  PluginFieldCollection mFc{};
};

}  // namespace

void bt6t_set_logger_finder(ILoggerFinder* finder) { gFinder = finder; }

IPluginCreatorInterface* bt6t_triplet_site_creator() {
  static TripletSiteCreator creator;
  return &creator;
}

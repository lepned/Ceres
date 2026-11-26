// ONNX Runtime variant only (build_k4_trt10_ort): register the library's plugin creators with TensorRT's plugin registry
// when the DLL is loaded. ONNX Runtime's TensorRT EP loads trt_extra_plugin_lib_paths as a plain shared library, so the
// dynamic-plugin entry points (getCreators / setLoggerFinder, used by getPluginRegistry()->loadLibrary) are never
// called and TensorRT's ONNX parser would not find the plugins.
#include "NvInferRuntime.h"

using namespace nvinfer1;

extern "C" IPluginCreatorInterface* const* getCreators(int32_t& nbCreators);

namespace {
struct Bt6AutoRegister {
  Bt6AutoRegister() {
    int32_t n = 0;
    IPluginCreatorInterface* const* creators = getCreators(n);
    IPluginRegistry* registry = getPluginRegistry();
    if (registry == nullptr || creators == nullptr) return;
    for (int32_t i = 0; i < n; ++i) registry->registerCreator(*creators[i], "");
  }
} bt6AutoRegister;
}  // namespace

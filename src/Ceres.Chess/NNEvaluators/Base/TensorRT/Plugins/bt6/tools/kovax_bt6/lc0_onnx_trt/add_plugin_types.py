"""Add value_info (element type and shape) for every output of the trt.plugins nodes, taken from TensorRT's own parse of the
graph with the plugin library loaded, so ONNX Runtime's type inference can resolve the placeholder custom ops.
usage: add_plugin_types.py in.onnx out.onnx plugin.dll"""
import sys, ctypes, os
for _d in ("C:/Dev/TensorRT-11.3.0.99/bin", "C:/Dev/TensorRT-11.3.0.99/lib"):
  if os.path.isdir(_d): os.add_dll_directory(_d)
import onnx, tensorrt as trt
src, dst, dll = sys.argv[1:4]
ctypes.CDLL(dll)
log = trt.Logger(trt.Logger.ERROR)
trt.init_libnvinfer_plugins(log, "")
b = trt.Builder(log)
reg = b.get_plugin_registry() if hasattr(b, "get_plugin_registry") else trt.get_plugin_registry()
reg.load_library(dll)
net = b.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
p = trt.OnnxParser(net, log)
assert p.parse_from_file(src), [p.get_error(i).desc() for i in range(p.num_errors)]
types = {}
for i in range(net.num_layers):
  L = net.get_layer(i)
  for j in range(L.num_outputs):
    t = L.get_output(j); types[t.name] = (t.dtype, [d if d >= 0 else 'batch' for d in t.shape])
TRT2ONNX = {trt.float32: 1, trt.float16: 10, trt.int8: 3, trt.int32: 6, trt.int64: 7, trt.bool: 9, trt.bfloat16: 16}
m = onnx.load(src)
have = {v.name for v in m.graph.value_info} | {v.name for v in m.graph.output}
n = 0
for node in m.graph.node:
  if node.domain != "trt.plugins":
    continue
  for o in node.output:
    if o and o not in have:
      assert o in types, o
      dt, shp = types[o]; m.graph.value_info.append(onnx.helper.make_tensor_value_info(o, TRT2ONNX[dt], shp)); n += 1
onnx.save(m, dst)
print(f"{n} plugin outputs typed -> {dst}")

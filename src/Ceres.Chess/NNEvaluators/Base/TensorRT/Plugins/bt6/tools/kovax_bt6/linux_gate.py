"""Linux check of the plugin library: build a static-batch engine from a plugin ONNX with the TensorRT Python package
(strongly typed), run the real gate positions and compare with the PyTorch FP32 reference.
    python linux_gate.py <libbt6_trt_plugin.so> <plugin.onnx (planes fp32 in, policy/wdl/mlh/err out)> <batch>"""
import sys
import numpy as np
import torch
import tensorrt as trt

sys.path.insert(0, "/mnt/c/Dev/Chess/kovax_bt6_trt")
from metrics import LEGAL, gate

so, onnx_path, B = sys.argv[1], sys.argv[2], int(sys.argv[3])
log = trt.Logger(trt.Logger.WARNING)
assert trt.get_plugin_registry().load_library(so) is not None, "plugin library did not load"
b = trt.Builder(log)
net = b.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
p = trt.OnnxParser(net, log)
assert p.parse_from_file(onnx_path), [p.get_error(i).desc() for i in range(p.num_errors)]
cfg = b.create_builder_config()
prof = b.create_optimization_profile()
name_in = net.get_input(0).name
prof.set_shape(name_in, (B, 112, 8, 8), (B, 112, 8, 8), (B, 112, 8, 8))
cfg.add_optimization_profile(prof)
plan = b.build_serialized_network(net, cfg)
assert plan is not None, "engine build failed"
eng = trt.Runtime(log).deserialize_cuda_engine(plan)
ctx = eng.create_execution_context()
ctx.set_input_shape(name_in, (B, 112, 8, 8))
x = torch.from_numpy(np.fromfile(f"/mnt/c/Dev/Chess/kovax_bt6_trt/planes_b{B}.bin", np.float32).reshape(B, 112, 8, 8)).cuda()
outs = {}
for i in range(eng.num_io_tensors):
  n = eng.get_tensor_name(i)
  if eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
    ctx.set_tensor_address(n, x.data_ptr())
  else:
    dt = {trt.float32: torch.float32, trt.float16: torch.float16}[eng.get_tensor_dtype(n)]
    outs[n] = torch.empty(tuple(ctx.get_tensor_shape(n)), dtype=dt, device="cuda")
    ctx.set_tensor_address(n, outs[n].data_ptr())
assert ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
torch.cuda.synchronize()
out = {n: t.float().cpu().numpy() for n, t in outs.items()}
ref = dict(np.load("/mnt/c/Dev/Chess/kovax_bt6_trt/ref_fp32_256.npz"))
legal = np.load(LEGAL)["legal"]
print(f"TensorRT {trt.__version__} Linux, {so.split('/')[-1]}")
print(gate(out, {k: ref[k][:B] for k in ("policy", "wdl", "mlh", "err")}, legal, f"linux_b{B}"))

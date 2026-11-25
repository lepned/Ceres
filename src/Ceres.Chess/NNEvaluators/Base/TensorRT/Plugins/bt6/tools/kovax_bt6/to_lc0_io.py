"""Give a BT6 ONNX (input planes fp32 [B,112,8,8]; outputs policy, wdl logits, mlh, err fp32) the I/O convention of
Mycel's / Ceres' Lc0 net format (lc0 leela2onnx): FP16 input /input/planes, FP16 outputs /output/policy (1858
logits), /output/wdl (softmax: W/D/L probabilities), /output/mlh. err is dropped as an output. Graph otherwise
unchanged.   usage: to_lc0_io.py in.onnx out.onnx"""
import sys
import onnx
from onnx import TensorProto, helper

m = onnx.load(sys.argv[1])
g = m.graph
assert [i.name for i in g.input] == ["planes"], [i.name for i in g.input]
old_in = g.input[0]
dims = [d.dim_param or d.dim_value for d in old_in.type.tensor_type.shape.dim]
new_in = helper.make_tensor_value_info("/input/planes", TensorProto.FLOAT16, dims)
for n in g.node:
  for k, x in enumerate(n.input):
    if x == "planes":
      n.input[k] = "lc0io_planes32"
nodes = [helper.make_node("Cast", ["/input/planes"], ["lc0io_planes32"], to=TensorProto.FLOAT, name="lc0io_cast_in")]
outs = {o.name: o for o in g.output}
assert {"policy", "wdl", "mlh"} <= set(outs), list(outs)
nodes.append(helper.make_node("Cast", ["policy"], ["/output/policy"], to=TensorProto.FLOAT16, name="lc0io_cast_policy"))
nodes.append(helper.make_node("Softmax", ["wdl"], ["lc0io_wdl_p"], axis=-1, name="lc0io_wdl_softmax"))
nodes.append(helper.make_node("Cast", ["lc0io_wdl_p"], ["/output/wdl"], to=TensorProto.FLOAT16, name="lc0io_cast_wdl"))
nodes.append(helper.make_node("Cast", ["mlh"], ["/output/mlh"], to=TensorProto.FLOAT16, name="lc0io_cast_mlh"))
g.node.insert(0, nodes[0])
g.node.extend(nodes[1:])
del g.input[:]
g.input.append(new_in)
b = dims[0]
del g.output[:]
g.output.extend([helper.make_tensor_value_info("/output/policy", TensorProto.FLOAT16, [b, 1858]),
                 helper.make_tensor_value_info("/output/wdl", TensorProto.FLOAT16, [b, 3]),
                 helper.make_tensor_value_info("/output/mlh", TensorProto.FLOAT16, [b, 1])])
onnx.save(m, sys.argv[2])
print("wrote", sys.argv[2], "batch dim", b)

"""k4: rewrite of int8/k2_lc8.onnx (lc0ex Q1 static INT8 on all four trunk sites): per encoder layer the ffn_in chain
    codes(ffn_in, INT8) -> DQ(al) -> MatMul(DQ(W8, sc/al)) -> Add(b) -> Split -> Sigmoid(g) * u' -> Q(al_mid) -> codes(ffn_mid)
becomes ONE plugin node
    BT6I8GemmGlu(codes(ffn_in), W8 INT8 [2048, 1024] (the carrier's interleaved gate/up pairs), tab FP32 [5120])
      -> codes(ffn_mid) INT8 [.., 1024]     (tab = carrier dense1/i8 scale | dense1/i8 bias | ffn/mid/i8/r)
The output keeps the tensor name of the old QuantizeLinear (k2_L{i}_ffn_mid_q), so the DQ -> dense2 MatMul stays.

    python k4_onnx.py --out k4.onnx [--src ../int8/k2_lc8.onnx] [--layers 0-23] [--tile -1]
           [--control deinterleave]   (must fail: weights / tab in [gate | up] order, so the plugin pairs wrong columns)
"""
import argparse
import collections
import os

import numpy as np
import onnx
from onnx import helper, numpy_helper

DOMAIN = "trt.plugins"


def parse_layers(s):
  r = set()
  for part in s.split(","):
    if "-" in part:
      a, b = part.split("-"); r |= set(range(int(a), int(b) + 1))
    elif part:
      r.add(int(part))
  return r


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--src", default=r"C:\Dev\Chess\kovax_bt6_trt\int8\k2_lc8.onnx")
  ap.add_argument("--out", required=True)
  ap.add_argument("--layers", default="0-23")
  ap.add_argument("--tile", type=int, default=-1)
  ap.add_argument("--control", default="")
  ap.add_argument("--carrier", default=r"C:\Dev\Chess\kovax_bt6_trt\int8\carrier_all.npz")
  a = ap.parse_args()
  m = onnx.load(a.src)
  g = m.graph
  inits = {t.name: t for t in g.initializer}
  prod = {o: n for n in g.node for o in n.output}
  cons = collections.defaultdict(list)
  for n in g.node:
    for i in n.input:
      cons[i].append(n)
  z = np.load(a.carrier)
  car = lambda k: z[k.replace("/", "|")]
  arr = lambda name: numpy_helper.to_array(inits[name])
  layers = parse_layers(a.layers)
  new_nodes, new_inits, dead = [], [], set()
  worst = 0.0
  for i in sorted(layers):
    tag = f"k2_L{i}_ffn_in"
    mm = [n for n in g.node if n.op_type == "MatMul" and n.input[1] == f"{tag}_w"]
    assert len(mm) == 1, (i, len(mm))
    mm = mm[0]
    dq = prod[mm.input[0]]; assert dq.op_type == "DequantizeLinear" and dq.input[0] == f"{tag}_q"
    add = cons[mm.output[0]]; assert len(add) == 1 and add[0].op_type == "Add"; add = add[0]
    split = cons[add.output[0]]; assert len(split) == 1 and split[0].op_type == "Split"; split = split[0]
    sig = cons[split.output[0]]; assert len(sig) == 1 and sig[0].op_type == "Sigmoid"; sig = sig[0]
    mul = cons[sig.output[0]]; assert len(mul) == 1 and mul[0].op_type == "Mul"; mul = mul[0]
    assert split.output[1] in mul.input
    q = cons[mul.output[0]]; assert len(q) == 1 and q[0].op_type == "QuantizeLinear"; q = q[0]
    out = q.output[0]; assert out == f"k2_L{i}_ffn_mid_q", out
    al_mid = float(arr(q.input[1]))
    p_ = f"/encoder{i}"
    W8 = car(f"{p_}/ffn/dense1/i8/w").view(np.int8)            # [2048, 1024], rows g0 u0 g1 u1 ..
    sc = car(f"{p_}/ffn/dense1/i8/scale").astype(np.float32)
    bi = car(f"{p_}/ffn/dense1/i8/bias").astype(np.float32)
    r = car(f"{p_}/ffn/mid/i8/r").astype(np.float32)
    # cross-check against the k2lc8 graph: its weights are W8 de-interleaved, its up lane scaled by r * al_mid
    wg = arr(mm.input[1] if False else prod[mm.input[1]].input[0])   # k2_L{i}_ffn_in_w8 [1024, 2048] = W8[order].T
    order = np.concatenate([np.arange(0, 2048, 2), np.arange(1, 2048, 2)])
    assert np.array_equal(wg, np.ascontiguousarray(W8[order].T)), i
    bk = arr(add.input[1]).astype(np.float64)                   # (b * mult) FP16, [gate | up]
    mult = np.ones(2048); mult[1024:] = r.astype(np.float64) * al_mid
    bref = bi[order].astype(np.float64) * mult
    err = np.abs(bk - bref).max() / (np.abs(bref).max() + 1e-9)
    worst = max(worst, err)
    if a.control == "deinterleave":
      W8, sc, bi = W8[order], sc[order], bi[order]
    tab = np.concatenate([sc, bi, r]).astype(np.float32)
    wn = f"k4_L{i}_ffn_in_w8"
    tn = f"k4_L{i}_ffn_in_tab"
    new_inits += [numpy_helper.from_array(np.ascontiguousarray(W8), wn), numpy_helper.from_array(tab, tn)]
    new_nodes.append(helper.make_node("BT6I8GemmGlu", [dq.input[0], wn, tn], [out], name=f"k4_L{i}_BT6I8GemmGlu",
                                      domain=DOMAIN, tile=int(a.tile), plugin_version="1", plugin_namespace=""))
    dead |= {id(n) for n in (dq, mm, add, split, sig, mul, q)}
  keep = []
  for n in g.node:
    if id(n) in dead:
      continue
    keep.append(n)
  # insert each plugin node where its old QuantizeLinear was: before the first consumer of its output
  byname = {nn.output[0]: nn for nn in new_nodes}
  final = []
  placed = set()
  for n in keep:
    for x in n.input:
      if x in byname and x not in placed:
        final.append(byname[x]); placed.add(x)
    final.append(n)
  assert len(placed) == len(new_nodes)
  del g.node[:]; g.node.extend(final)
  used = {x for n in final for x in n.input}
  g.initializer.extend(new_inits)
  kept = [t for t in g.initializer if t.name in used]
  del g.initializer[:]; g.initializer.extend(kept)
  del g.value_info[:]
  print(f"{len(new_nodes)} layers -> BT6I8GemmGlu (tile {a.tile}){' CONTROL ' + a.control if a.control else ''}; "
        f"bias cross-check vs k2lc8 graph: max rel err {worst:.2e}; {len(final)} nodes")
  big = sum(t.ByteSize() for t in kept) > 1.9e9
  onnx.save(m, a.out, save_as_external_data=big, location=os.path.basename(a.out) + ".data" if big else None)
  print("saved", a.out, f"{os.path.getsize(a.out) / 1e6:.0f} MB")


if __name__ == "__main__":
  main()

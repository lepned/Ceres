"""k2: INT8 rewrite of the k1 FP16 plugin ONNX (plugin/k1_dyn_tri.onnx) of Kovax's BT6 flagship (1024x24x32).

Per encoder layer the four trunk GEMMs (sites, lc0ex names):
  attn_in  = qkvg  (ln1 out, 1024 -> 4096)      attn_out = out   (attention plugin out, 1024 -> 1024)
  ffn_in   = dense1 [gate|up] (ln2 out, 1024 -> 2048)   ffn_mid = dense2 (sigmoid(g) * (u + bu), 1024 -> 1024)
become INT8 GEMMs with symmetric per-output-channel INT8 weights (from the FP16 carrier weights) and DYNAMIC
PER-TOKEN activation scales, as plain ONNX ops (TensorRT/Myelin fuses them):
    r = max(ReduceMax(|x|, -1), eps);  codes = Q(x / r, 1/127);  y = (DQ(codes, 1/127) @ DQ(W8, s_n)) * r + b
Heads, embedding, prologue, LayerNorms, edge sites stay as in k1. (Emulator, emu.py: w8,a:all=token.)

    python k2_int8_onnx.py --src k1_dyn_tri.onnx --out X.onnx [--sites attn_in,attn_out,ffn_in,ffn_mid]
           [--layers 0-23] [--control noscale:<site>]
Plugin mode (--glu-plugin / --lnq-plugin ...) is added by later options (see NOTES_k2.md).
"""
from __future__ import annotations

import argparse
import collections
import os
import time

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

SITES = ("attn_in", "attn_out", "ffn_in", "ffn_mid")
DOMAIN = "trt.plugins"


def index(g):
  prod = {o: n for n in g.node for o in n.output}
  cons = collections.defaultdict(list)
  for n in g.node:
    for i in n.input:
      cons[i].append(n)
  return prod, cons


def weight_of(g, inits, prod, mm):
  """MatMul input 1 = Cast(FP32 initializer) -> (initializer name, Cast node)."""
  c = prod[mm.input[1]]
  assert c.op_type == "Cast" and c.input[0] in inits, (mm.name, c.op_type)
  return c.input[0], c


def trunk_sites(g, inits, prod, cons, layers=24):
  """[(layer, site, MatMul)] in graph order: matmul 4 + 4 i + {0..3} (the first four are the embedding's)."""
  mms = [n for n in g.node if n.op_type == "MatMul" and n.input[1] in prod and prod[n.input[1]].op_type == "Cast"
         and prod[n.input[1]].input[0] in inits]
  out = []
  for i in range(layers):
    for k, site in enumerate(SITES):
      mm = mms[4 + 4 * i + k]
      w, _ = weight_of(g, inits, prod, mm)
      K, N = inits[w].dims
      exp = {"attn_in": (1024, 4096), "attn_out": (1024, 1024), "ffn_in": (1024, 2048), "ffn_mid": (1024, 1024)}[site]
      assert (K, N) == exp, (i, site, mm.name, K, N)
      if site == "attn_out":
        assert mm.input[0] == f"att{i}", (i, mm.input[0])
      out.append((i, site, mm))
  return out


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
  ap.add_argument("--src", default="/mnt/c/Dev/Chess/kovax_bt6_trt/plugin/k1_dyn_tri.onnx")
  ap.add_argument("--out", required=True)
  ap.add_argument("--sites", default=",".join(SITES))
  ap.add_argument("--layers", default="0-23")
  ap.add_argument("--control", default="", help="noscale:<site> (must fail: no * r after that site's GEMM)")
  ap.add_argument("--eps", type=float, default=1e-4)
  ap.add_argument("--lc-sites", default="", help="sites served lc0ex-style (static, carrier int8 weights + r folded "
                  "into the LayerNorm): attn_in,ffn_in")
  ap.add_argument("--glu-plugin", action="store_true", help="ffn_mid per-token quant by BT6SigGluQuant")
  ap.add_argument("--rq-mid", action="store_true", help="ffn_mid: GLU left to TensorRT (FP16 m), BT6RowQuant on m")
  ap.add_argument("--rq-plugin", action="store_true", help="attn_out per-token quant by BT6RowQuant")
  ap.add_argument("--attn-i8", action="store_true", help="lc attn_out: the attention plugin writes the INT8 codes")
  ap.add_argument("--emb", action="store_true", help="embedding GLU FFN INT8 too (per-token)")
  ap.add_argument("--d1", action="store_true", help="lc0ex D1 value layer (value/dense2 w_d1, b_d1)")
  ap.add_argument("--carrier", default="/mnt/c/Dev/Chess/kovax_bt6_trt/int8/carrier_all.npz")
  a = ap.parse_args()
  t0 = time.time()
  m = onnx.load(a.src)
  g = m.graph
  inits = {t.name: t for t in g.initializer}
  prod, cons = index(g)
  sites = set(a.sites.split(","))
  layers = parse_layers(a.layers)
  ctl_site = a.control.split(":")[1] if a.control.startswith("noscale:") else ""
  new_inits, new_nodes = [], []
  dead_inits = set()

  def init(arr, name):
    new_inits.append(numpy_helper.from_array(arr, name)); return name

  inv127 = init(np.array(1.0 / 127.0, np.float16), "k2_inv127")
  zp = init(np.array(0, np.int8), "k2_zp")
  eps = init(np.array(a.eps, np.float16), "k2_eps")
  rq_cache = {}

  def per_token(x, tag):
    """x FP16 [.., K] -> (DQ'd codes tensor, r [.., 1])."""
    if x in rq_cache:
      return rq_cache[x]
    nodes = [helper.make_node("Abs", [x], [f"{tag}_abs"], name=f"{tag}_Abs"),
             helper.make_node("ReduceMax", [f"{tag}_abs", "k2_axes_last"], [f"{tag}_rm"], name=f"{tag}_ReduceMax", keepdims=1),
             helper.make_node("Max", [f"{tag}_rm", eps], [f"{tag}_r"], name=f"{tag}_RMax"),
             helper.make_node("Div", [x, f"{tag}_r"], [f"{tag}_xs"], name=f"{tag}_Div"),
             helper.make_node("QuantizeLinear", [f"{tag}_xs", inv127, zp], [f"{tag}_q"], name=f"{tag}_Q8"),
             helper.make_node("DequantizeLinear", [f"{tag}_q", inv127, zp], [f"{tag}_dq"], name=f"{tag}_DQ8")]
    new_nodes.extend(nodes)
    rq_cache[x] = (f"{tag}_dq", f"{tag}_r")
    return rq_cache[x]

  init(np.array([-1], np.int64), "k2_axes_last")
  lc_sites = set(x for x in a.lc_sites.split(",") if x)
  assert lc_sites <= set(SITES), lc_sites
  assert not (lc_sites & {"attn_out", "ffn_mid"}) or {"attn_in", "ffn_in"} <= lc_sites, "attn_out/ffn_mid lc need the producer lc"

  def pow2ceil(v):
    return float(2.0 ** np.ceil(np.log2(v)))

  def lc_conv(i, site):
    """attn_out / ffn_mid served lc0ex-style: (r, al) -- the producer's lane is scaled by r * al so that
    Q(lane, al) = round(x r) and the GEMM reads DQ(codes, al) @ DQ(W8, sc / al)."""
    p_ = f"/encoder{i}"
    stem, rv = {"attn_out": ("mha/out/i8", "mha/out/i8/r"), "ffn_mid": ("ffn/dense2/i8", "ffn/mid/i8/r")}[site]
    sc = car(f"{p_}/{stem}/scale").astype(np.float64)
    return car(f"{p_}/{rv}").astype(np.float64), pow2ceil(sc.max())

  def bias_add(mm):
    """the Add(MatMul, bias) after a GEMM -> (Add node, input index of the bias, bias FP32 array)."""
    add = cons[mm.output[0]]; assert len(add) == 1 and add[0].op_type == "Add", mm.name
    add = add[0]
    k = 1 if add.input[0] == mm.output[0] else 0
    t = add.input[k]
    if t in inits:
      return add, k, numpy_helper.to_array(inits[t]).astype(np.float64)
    c = prod[t]; assert c.op_type == "Cast" and c.input[0] in inits, c.op_type
    return add, k, numpy_helper.to_array(inits[c.input[0]]).astype(np.float64)
  car = None
  if lc_sites or a.d1:
    z = np.load(a.carrier)
    car = lambda k: z[k.replace("/", "|")]
  done = 0
  lc_done = 0
  for i, site, mm in trunk_sites(g, inits, prod, cons):
    if i not in layers or site not in sites:
      continue
    tag = f"k2_L{i}_{site}"
    wname, cast = weight_of(g, inits, prod, mm)
    if site in lc_sites and site in ("attn_in", "ffn_in"):
      # lc0ex Q1 (pre-norm): codes = round(LN(x) * r) from the LN with gamma*r*al, beta*r*al -> Q(al) -> DQ(al);
      # weights = the carrier's int8 W8 [n, k] with per-column scale sc (sc * codes.W8 = the GEMM), DQ scale sc / al.
      p_ = f"/encoder{i}"
      stem, rn = {"attn_in": ("mha/qkv/i8", "ln1"), "ffn_in": ("ffn/dense1/i8", "ln2")}[site]
      W8 = car(f"{p_}/{stem}/w").view(np.int8)
      sc = car(f"{p_}/{stem}/scale").astype(np.float64)
      r = car(f"{p_}/{rn}/quant/r").astype(np.float64)
      assert not car(f"{p_}/{rn}/quant/m").any()
      if site == "ffn_in":
        order = np.concatenate([np.arange(0, W8.shape[0], 2), np.arange(1, W8.shape[0], 2)])
        W8, sc = W8[order], sc[order]
      lane = {"attn_in": ("attn_out", slice(2048, 3072)), "ffn_in": ("ffn_mid", slice(1024, 2048))}[site]
      if lane[0] in lc_sites:   # scale the V lane (attn_out) / up lane (ffn_mid) by the consumer's r * al
        rr, alc = lc_conv(i, lane[0])
        mult = np.ones(sc.shape[0]); mult[lane[1]] = rr * alc
        if a.control == "lc_lane":   # must-fail control: the consumer's prescale left out of the lane
          mult[lane[1]] = alc * rr.mean()
        sc = sc * mult
        add, k, b = bias_add(mm)
        add.input[k] = init((b * mult).astype(np.float16), f"{tag}_bias")
      al = 2.0 ** np.ceil(np.log2(sc.max()))
      wsc = sc / al
      if wsc.min() < 6.2e-5:
        print(f"  note L{i} {site}: weight scale min {wsc.min():.3g} is FP16-subnormal")
      c16 = prod[mm.input[0]]; assert c16.op_type == "Cast", c16.op_type
      ln = prod[c16.input[0]]; assert ln.op_type == "LayerNormalization", ln.op_type
      assert len(cons[c16.input[0]]) == 1 and len(cons[c16.output[0]]) == 1, (i, site)
      if a.control == "lc_r1":   # must-fail control: the LN-site prescale r replaced by its mean (one step, no smoothing)
        r = np.full_like(r, r.mean())
      gam = numpy_helper.to_array(inits[ln.input[1]]).astype(np.float64)
      bet = numpy_helper.to_array(inits[ln.input[2]]).astype(np.float64)
      ln.input[1] = init((gam * r * al).astype(np.float32), f"{tag}_gamma")
      ln.input[2] = init((bet * r * al).astype(np.float32), f"{tag}_beta")
      new_nodes.append(helper.make_node("QuantizeLinear", [ln.output[0], init(np.array(al, np.float32), f"{tag}_al32"), zp],
                                        [f"{tag}_q"], name=f"{tag}_Q8"))
      new_nodes.append(helper.make_node("DequantizeLinear", [f"{tag}_q", init(np.array(al, np.float16), f"{tag}_al16"), zp],
                                        [f"{tag}_dq"], name=f"{tag}_DQ8"))
      new_nodes.append(helper.make_node("DequantizeLinear", [init(np.ascontiguousarray(W8.T), f"{tag}_w8"),
                                                             init(wsc.astype(np.float16), f"{tag}_ws"),
                                                             init(np.zeros(W8.shape[0], np.int8), f"{tag}_wz")],
                                        [f"{tag}_w"], name=f"{tag}_WDQ8", axis=1))
      mm.input[0], mm.input[1] = f"{tag}_dq", f"{tag}_w"
      lc_done += 1
      done += 1
      continue
    if site in lc_sites:   # attn_out / ffn_mid: Q(lane, al) of the pre-scaled producer lane, carrier int8 weights
      p_ = f"/encoder{i}"
      stem = {"attn_out": "mha/out/i8", "ffn_mid": "ffn/dense2/i8"}[site]
      W8 = car(f"{p_}/{stem}/w").view(np.int8)
      sc = car(f"{p_}/{stem}/scale").astype(np.float64)
      _, al = lc_conv(i, site)
      new_nodes.append(helper.make_node("DequantizeLinear", [init(np.ascontiguousarray(W8.T), f"{tag}_w8"),
                                                             init((sc / al).astype(np.float16), f"{tag}_ws"),
                                                             init(np.zeros(W8.shape[0], np.int8), f"{tag}_wz")],
                                        [f"{tag}_w"], name=f"{tag}_WDQ8", axis=1))
      al16 = init(np.array(al, np.float16), f"{tag}_al16")
      x = mm.input[0]
      if site == "attn_out" and a.attn_i8:
        att = prod[x]; assert att.op_type == "BT6EgtAttention" and att.output[0] == x
        for at in list(att.attribute):
          if at.name in ("out_int8", "out_scale"):
            att.attribute.remove(at)
        att.attribute.extend([helper.make_attribute("out_int8", 1), helper.make_attribute("out_scale", float(np.float32(al)))])
        qn = x
      else:
        new_nodes.append(helper.make_node("QuantizeLinear", [x, al16, zp], [f"{tag}_q"], name=f"{tag}_Q8"))
        qn = f"{tag}_q"
      new_nodes.append(helper.make_node("DequantizeLinear", [qn, al16, zp], [f"{tag}_dq"], name=f"{tag}_DQ8"))
      mm.input[0], mm.input[1] = f"{tag}_dq", f"{tag}_w"
      lc_done += 1
      done += 1
      continue
    W = numpy_helper.to_array(inits[wname]).astype(np.float32)          # [K, N]
    ws = (np.maximum(np.abs(W).max(axis=0), 1e-8) / 127.0).astype(np.float16)
    w8 = np.clip(np.round(W / ws.astype(np.float32)), -127, 127).astype(np.int8)
    new_nodes.append(helper.make_node("DequantizeLinear", [init(w8, f"{tag}_w8"), init(ws, f"{tag}_ws"),
                                                           init(np.zeros(W.shape[1], np.int8), f"{tag}_wz")],
                                      [f"{tag}_w"], name=f"{tag}_WDQ8", axis=1))
    if (site == "ffn_mid" and (a.glu_plugin or a.rq_mid)) or (site == "attn_out" and a.rq_plugin):
      x = mm.input[0]
      if site == "ffn_mid" and not a.rq_mid:   # Mul(Sigmoid(Split0), Split1) <- Split <- Add(dense1 + b): the plugin reads the Add output
        mul = prod[x]; assert mul.op_type == "Mul", mul.op_type
        sig = prod[mul.input[0]] if prod[mul.input[0]].op_type == "Sigmoid" else prod[mul.input[1]]
        assert sig.op_type == "Sigmoid"
        split = prod[sig.input[0]]; assert split.op_type == "Split" and split.output[0] == sig.input[0]
        x = split.input[0]
      op = "BT6SigGluQuant" if site == "ffn_mid" and not a.rq_mid else "BT6RowQuant"
      new_nodes.append(helper.make_node(op, [x], [f"{tag}_codes", f"{tag}_r"], name=f"{tag}_{op}", domain=DOMAIN,
                                        eps=float(a.eps), plugin_version="1", plugin_namespace=""))
      new_nodes.append(helper.make_node("DequantizeLinear", [f"{tag}_codes", inv127, zp], [f"{tag}_dq"], name=f"{tag}_DQ8"))
      xdq, r = f"{tag}_dq", f"{tag}_r"
      plugins_used = True
    else:
      xdq, r = per_token(mm.input[0], tag)
    mm.input[0], mm.input[1] = xdq, f"{tag}_w"
    if site != ctl_site:
      y = mm.output[0]
      mm.output[0] = f"{tag}_yn"
      new_nodes.append(helper.make_node("Mul", [f"{tag}_yn", r], [y], name=f"{tag}_RowScale"))
    done += 1
  if a.emb:   # the embedding's GLU FFN (dense1 1024 -> 4096, dense2 2048 -> 1024): w8 + per-token ONNX ops
    mms = [n for n in g.node if n.op_type == "MatMul" and n.input[1] in prod and prod[n.input[1]].op_type == "Cast"
           and prod[n.input[1]].input[0] in inits]
    for k, mm in enumerate(mms[2:4]):
      wname = prod[mm.input[1]].input[0]
      W = numpy_helper.to_array(inits[wname]).astype(np.float32)
      assert W.shape == ((1024, 4096), (2048, 1024))[k], W.shape
      tag = f"k2_emb{k}"
      ws = (np.maximum(np.abs(W).max(axis=0), 1e-8) / 127.0).astype(np.float16)
      w8 = np.clip(np.round(W / ws.astype(np.float32)), -127, 127).astype(np.int8)
      new_nodes.append(helper.make_node("DequantizeLinear", [init(w8, f"{tag}_w8"), init(ws, f"{tag}_ws"),
                                                             init(np.zeros(W.shape[1], np.int8), f"{tag}_wz")],
                                        [f"{tag}_w"], name=f"{tag}_WDQ8", axis=1))
      xdq, r = per_token(mm.input[0], tag)
      mm.input[0], mm.input[1] = xdq, f"{tag}_w"
      y = mm.output[0]
      mm.output[0] = f"{tag}_yn"
      new_nodes.append(helper.make_node("Mul", [f"{tag}_yn", r], [y], name=f"{tag}_RowScale"))
    print("embedding FFN INT8 (per-token)")
  if a.d1:   # lc0ex's D1 refit of the value head's WDL layer (the INT8 program serves it)
    hits = [n for n in g.node if n.op_type == "MatMul" and n.input[1] in inits and list(inits[n.input[1]].dims) == [128, 3]]
    assert len(hits) == 1, len(hits)
    mm = hits[0]
    wd = numpy_helper.to_array(inits[mm.input[1]])
    assert np.abs(wd.astype(np.float32) - car("/value/dense2/w").astype(np.float32)).max() < 1e-3
    mm.input[1] = init(car("/value/dense2/w_d1").astype(wd.dtype), "k2_value_w_d1")
    add, k, b = bias_add(mm)
    bd1 = car("/value/dense2/b_d1").reshape(b.shape)
    if add.input[k] in inits:
      add.input[k] = init(bd1.astype(numpy_helper.to_array(inits[add.input[k]]).dtype), "k2_value_b_d1")
    else:
      prod[add.input[k]].input[0] = init(bd1.astype(np.float32), "k2_value_b_d1")
    print("D1 value layer applied")
  # prune: dead Cast nodes of the replaced weights and their FP32 initializers
  nodes = list(g.node) + new_nodes
  g.initializer.extend(new_inits)
  # topological order: insert each new node right before its first consumer
  produced = {o for n in nodes for o in n.output}
  order, seen = [], set()
  avail = {t.name for t in g.initializer} | {v.name for v in g.input} | {""}
  byout = {o: n for n in nodes for o in n.output}
  visiting = set()

  def visit(n):
    if n.name in seen:
      return
    stack = [(n, 0)]
    while stack:
      node, k = stack.pop()
      if node.name in seen:
        continue
      if k < len(node.input):
        stack.append((node, k + 1))
        x = node.input[k]
        if x not in avail and x in byout and byout[x].name not in seen:
          stack.append((byout[x], 0))
        continue
      seen.add(node.name)
      order.append(node)
      for o in node.output:
        avail.add(o)

  # liveness from graph outputs
  live, work = set(), [o.name for o in g.output]
  while work:
    t = work.pop()
    n = byout.get(t)
    if n is None or n.name in live:
      continue
    live.add(n.name)
    work.extend(n.input)
  for n in nodes:
    if n.name in live:
      visit(n)
  used = {x for n in order for x in n.input}
  keep_inits = [t for t in g.initializer if t.name in used]
  del g.node[:]; g.node.extend(order)
  del g.initializer[:]; g.initializer.extend(keep_inits)
  del g.value_info[:]
  print(f"{done} GEMM sites INT8 ({lc_done} lc0ex-static, rest per-token){' CONTROL ' + a.control if a.control else ''}; {len(order)} nodes, "
        f"{len(keep_inits)} initializers, {time.time() - t0:.0f}s", flush=True)
  big = sum(t.ByteSize() for t in keep_inits) > 1.9e9
  onnx.save(m, a.out, save_as_external_data=big, location=os.path.basename(a.out) + ".data" if big else None)
  print("saved", a.out, f"{os.path.getsize(a.out) / 1e6:.0f} MB")


if __name__ == "__main__":
  main()

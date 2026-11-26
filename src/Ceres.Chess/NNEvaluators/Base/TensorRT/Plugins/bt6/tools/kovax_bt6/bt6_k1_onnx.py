"""k1: TensorRT plugin ONNX of Kovax's BT6 flagship (1024x24x32) for the bt6_trt_plugin_k1 library (WSL, torch).

Two steps in one run:
 1. Torch export (dynamo) of BT6K1 = bt6_export_onnx.BT6Served with the attention cores and the prologue CUT OUT of the
    graph: per layer i the qkvg GEMM output becomes a graph output `lanes{i}` FP16 [B, 64, 4096] and the input of the
    out GEMM a graph input `att{i}` FP16 [B, 64, 1024]; at the edge-site layers the site's H a graph input `H{i}`
    FP16 [B, 32, 64, 64]; the prologue's results are graph inputs `e0` FP16 [B, 64, 64, 16] (cell-major) and `polpair`
    FP32 [B, 64, 64] (= S * (policy pair term)); each edge state after a site is a graph output `state{k}` FP16
    [B, 64, 64, 16] cell-major. Everything else (embedding, LayerNorms, GEMMs, GLUs, edge sites incl. the triplet,
    policy / value heads) is bt6_export_onnx's FP16 graph unchanged. Then trt_gate_fix (the triplet's
    softmax * sigmoid -> MatMul pattern).
 2. ONNX rewrite: one BT6PlanesPrologue node (planes -> edges, S, e0, polpair) and per layer one BT6EgtAttention node
    (lanes{i}, edges, S, state of that layer, tab16_i, tab32_i -> att{i} [, H{i} at site layers]); the cut graph
    inputs / outputs go. Tables from the carrier's FP32 tensors (layout: trt_bt6_plugin/src_k1/bt6_egt_attn.cu,
    bt6_prologue_kernels.cu).

    python bt6_k1_onnx.py --out X.onnx [--batch B | --dynamic] [--control drop_attack|drop_prev|drop_pairoff] [--e0f] [--triplet-plugin]

--control makes a must-fail variant: drop_attack zeroes the attention plugins' attack table (a term of every layer),
drop_prev feeds the prologue planes 13-24 = planes 0-11 (wrong history: the gained / lost channels vanish; on the
256 gate positions this changes only 4 E bits, so it does NOT fail the gate), drop_pairoff zeroes the prologue's T.

Site interface for a fused (triplet) edge-site plugin: H{i} is FP16 [B, 32, 64, 64] (head-major, row i, key j
innermost) as the attention plugin writes it; the edge state the attention reads is FP16 [B, 64, 64, 16] cell-major
(channel innermost); the site in this graph takes e FP32 [B, 16, 4096] (channels first) and returns the same, and
state{k} = Transpose(Cast(e, FP16)). A fused site can take (state{k-1} or e, H{i}) and produce state{k} directly.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import onnx
import torch
from onnx import helper, numpy_helper

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, __import__("os").environ.get("CERESTRAIN_PY", "CeresTrain/src/CeresTrainPy"))
from bt6_export_onnx import BT6Served, load_carrier, mish, real_planes  # noqa: E402
from trt_gate_fix import apply_gate_fix  # noqa: E402

DOMAIN = "trt.plugins"
H_, HD, CH, PAD = 32, 32, 34, 40


class BT6K1(BT6Served):
  def forward(self, planes, e0, polpair, atts, Hs):
    return self.body(planes, e0.float().permute(0, 3, 1, 2), polpair, atts, Hs)

  fused_sites = False

  def body(self, planes, e0cf, polpair, atts, Hs):
    cfg, cd = self.cfg, self.cd
    B, D = planes.shape[0], cfg.d_model
    planes = planes.float().reshape(B, 112, 64)
    e = None if self.fused_sites else e0cf.reshape(B, cfg.edge_channels, 4096)                                 # FP32 channels-first for the sites
    per_square = planes.permute(0, 2, 1).to(cd)
    position = self.dense(per_square[:, :, :12].reshape(B, 768), "/attn_body/preproc").view(B, 64, -1)
    x = self.dense(torch.cat((per_square, position), dim=-1), "/attn_body/matmul")
    x = self.ln(mish(x), "/attn_body/ln0")
    x = x * self.T("/attn_body/mult_gate/w") + self.T("/attn_body/position/w").view(64, D)
    x = x + self.T("/attn_body/ffn/alpha/w") * self.glu(x, "/attn_body/ffn")
    x = self.ln(x, "/attn_body/ln1")
    lanes_out, states = [], []
    k = 0
    for i in range(cfg.layers):
      p = f"/encoder{i}"
      y = self.ln(x, f"{p}/ln1")
      lanes_out.append(self.dense(y, f"{p}/mha/qkvg"))
      x = x + self.T(f"{p}/mha/alpha/w") * self.dense(atts[i], f"{p}/mha/out")
      y = self.ln(x, f"{p}/ln2")
      x = x + self.T(f"{p}/ffn/alpha/w") * self.glu(y, f"{p}/ffn")
      if i in cfg.edge_sites and not self.fused_sites:
        e = self.site(i, e, Hs[k])
        k += 1
        states.append(e.to(cd).view(B, cfg.edge_channels, 64, 64).permute(0, 2, 3, 1).contiguous())
    x = self.ln(x, "/encoder/final_norm")
    emb = mish(self.dense(x, "/policy/embedding"))
    Q, K = self.dense(emb, "/policy/Q"), self.dense(emb, "/policy/K")
    logits = (Q @ K.transpose(-1, -2)) * self.T("/policy/qk_scale")
    logits = logits + polpair.to(cd)
    off = K[:, 56:64] @ self.T("/policy/promotion/w")
    promo = logits[:, 48:56, 56:64].unsqueeze(-1) + off[:, None, :, 0:3] + off[:, None, :, 3:4]
    record = torch.cat((logits.reshape(B, 4096), promo.reshape(B, 192)), dim=1)
    policy = torch.index_select(record, 1, self.gather)

    def head(prefix):
      sq = mish(self.dense(x, f"{prefix}/embedding")).reshape(B, -1)
      return self.dense(mish(self.dense(sq, f"{prefix}/dense1")), f"{prefix}/dense2")

    wdl = head("/value")
    mlh = torch.relu(head("/moves"))
    err = torch.sigmoid(head("/value_error")) if "/value_error/dense2/w" in self._names else torch.full_like(mlh, 0.5)
    return (policy.float(), wdl.float(), mlh.float(), err.float(), *lanes_out, *states)


class BT6K1F(BT6K1):
  """--e0f: the sites' e0 is the prologue plugin's FP32 channels-first output e0f [B, 16, 64, 64] (unrounded)."""
  def forward(self, planes, e0f, polpair, atts, Hs):
    return self.body(planes, e0f, polpair, atts, Hs)


class BT6K1T(BT6K1):
  """--triplet-plugin: the edge sites are BT6TripletSite nodes (added by the rewrite); the torch graph has no edge
  state at all (inputs planes, polpair, att*; outputs policy, wdl, mlh, err, lanes*)."""
  fused_sites = True

  def forward(self, planes, polpair, atts):
    return self.body(planes, None, polpair, atts, None)


def relidx():
  i = np.arange(64)
  return (15 * (i[:, None] // 8 - i[None] // 8 + 7) + (i[:, None] % 8 - i[None] % 8 + 7)).reshape(-1)


def prologue_table(t):
  """BT6PlanesPrologue blob: D [34][16] | Pin [34][16] | T [225][16] | Toff [225][16] | psc [34] | pcb [4096]."""
  f = lambda n: np.asarray(t[n], np.float64)
  D, Pin = f("/prologue/channel_mix"), f("/prologue/edge_mix")
  off, eoff = f("/prologue/offsets").reshape(16, 4096), f("/prologue/edge_offsets").reshape(16, 4096)
  rel = relidx()
  T = np.zeros((225, 16)); Toff = np.zeros((225, 16))
  T[rel] = off.T; Toff[rel] = eoff.T
  assert np.abs(T[rel].T - off).max() == 0 and np.abs(Toff[rel].T - eoff).max() == 0, "offsets are not relidx-only"
  psc = f("/policy/pair/scaled_coefficients").reshape(34)
  pcb = f("/policy/pair/constant_bias").reshape(4096)
  tab = np.concatenate([D.ravel(), Pin.ravel(), T.ravel(), Toff.ravel(), psc, pcb]).astype(np.float32)
  assert tab.size == 2 * 34 * 16 + 2 * 225 * 16 + 34 + 4096
  return tab


def attention_tables(t, i, control=""):
  g = f"/encoder{i}/mha/egt"
  f = lambda n: np.asarray(t[f"{g}/{n}"], np.float64)
  q, k = np.zeros((H_, PAD, HD)), np.zeros((H_, PAD, HD))
  q[:, :CH], k[:, :CH] = f("query"), f("key")
  rdg = np.concatenate([f("edge_read/w"), f("door/w"), f("gate/w")], 1)          # [32, 48]
  cb = f("constant_bias").reshape(H_, 4096)
  tab16 = np.concatenate([q.ravel(), k.ravel(), rdg.ravel(), cb.ravel()]).astype(np.float16)
  attack = f("attack") * (0.0 if control == "drop_attack" else 1.0)
  tab32 = np.concatenate([f("qk_scale").ravel(), attack.ravel(), f("scaled_coefficients").ravel(),
                          f("gate/b").ravel()]).astype(np.float32)
  assert tab16.size == 2 * 32 * 40 * 32 + 32 * 48 + 32 * 4096 and tab32.size == 32 + 2 * 32 * 34 + 32
  return tab16, tab32


def rewrite(m, t, cfg, control="", e0f=False, tsite=False):
  g = m.graph
  sites = list(cfg.edge_sites)
  tab_pro = prologue_table(t)
  if control == "drop_pairoff":   # must-fail control: the pair stream's relidx offsets T = 0 (S wrong everywhere)
    tab_pro[2 * 34 * 16:2 * 34 * 16 + 225 * 16] = 0
  inits = [numpy_helper.from_array(tab_pro, "k1_tab_pro")]
  nodes = []
  planes_in = "planes"
  if control == "drop_prev":   # must-fail control: history planes 13-24 replaced by 0-11
    idx = np.arange(112); idx[13:25] = np.arange(12)
    inits.append(numpy_helper.from_array(idx.astype(np.int64), "k1_prev_idx"))
    nodes.append(helper.make_node("Gather", ["planes", "k1_prev_idx"], ["k1_planes_ctl"], axis=1, name="k1_ctl_prev"))
    planes_in = "k1_planes_ctl"
  nodes.append(helper.make_node("BT6PlanesPrologue", [planes_in, "k1_tab_pro"],
                                ["k1_edges", "k1_S", "e0", "polpair"] + (["e0f"] if e0f else []),
                                name="k1_prologue", domain=DOMAIN, eps=float(cfg.rms_eps), e0_fp32=int(e0f),
                                plugin_version="1", plugin_namespace=""))
  k = 0
  for i in range(cfg.layers):
    tab16, tab32 = attention_tables(t, i, control)
    inits += [numpy_helper.from_array(tab16, f"k1_tab16_{i}"), numpy_helper.from_array(tab32, f"k1_tab32_{i}")]
    state = "e0" if k == 0 else f"state{k}"
    outs = [f"att{i}"] + ([f"H{i}"] if i in sites else [])
    nodes.append(helper.make_node("BT6EgtAttention", [f"lanes{i}", "k1_edges", "k1_S", state, f"k1_tab16_{i}",
                                                       f"k1_tab32_{i}"], outs, name=f"k1_attention_{i}", domain=DOMAIN,
                                  heads=H_, head_dim=HD, export_h=int(i in sites), out_int8=0, out_scale=0.0,
                                  plugin_version="1", plugin_namespace=""))
    if i in sites:
      k += 1
      if tsite:   # BT6TripletSite: (state of layer i, H{i}) -> state{k}, FP16 cell-major
        from triplet_site_tables import triplet_site_table
        inits.append(numpy_helper.from_array(triplet_site_table(t, i).astype(np.float32), f"k1_tabsite_{i}"))
        nodes.append(helper.make_node("BT6TripletSite", [state, f"H{i}", f"k1_tabsite_{i}"], [f"state{k}"],
                                      name=f"k1_site_{i}", domain=DOMAIN, heads=H_, contraction=0, plugin_version="1",
                                      plugin_namespace=""))
  cut_in = {"e0", "e0f", "polpair"} | {f"att{i}" for i in range(cfg.layers)} | {f"H{i}" for i in sites}
  for n in g.node:   # dynamic export: the batch size read off a cut input (Shape start 0 end 1) -> off planes
    if n.op_type == "Shape" and n.input[0] in cut_in:
      at = {a.name: a.i for a in n.attribute}
      assert at.get("start") == 0 and at.get("end") == 1, (n.name, at)
      n.input[0] = "planes"
  keep_in = [v for v in g.input if v.name not in cut_in]
  assert [v.name for v in keep_in] == ["planes"], [v.name for v in keep_in]
  keep_out = [v for v in g.output if v.name in ("policy", "wdl", "mlh", "err")]
  assert len(keep_out) == 4 and len(g.output) == 4 + cfg.layers + (0 if tsite else len(sites)), [v.name for v in g.output]
  del g.input[:]; g.input.extend(keep_in)
  del g.output[:]; g.output.extend(keep_out)
  # place the new nodes: prologue first, each attention right after its lanes / state producer
  old = list(g.node)
  produced_by = {o: j for j, n in enumerate(old) for o in n.output}
  after, slot = {}, {}
  for n in nodes:   # each new node right after its last producer (old node, or the slot of a new node before it)
    deps = [produced_by[x] for x in n.input if x in produced_by] + [slot[x] for x in n.input if x in slot]
    at = max(deps) if deps else -1
    after.setdefault(at, []).append(n)
    for o in n.output:
      slot[o] = at
  ordered = after.get(-1, [])
  for j, n in enumerate(old):
    ordered.append(n)
    ordered += after.get(j, [])
  del g.node[:]; g.node.extend(ordered)
  g.initializer.extend(inits)
  del g.value_info[:]
  if not any(o.domain == DOMAIN for o in m.opset_import):
    m.opset_import.append(helper.make_opsetid(DOMAIN, 1))
  return len(nodes)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", required=True)
  ap.add_argument("--batch", type=int, default=64)
  ap.add_argument("--dynamic", action="store_true")
  ap.add_argument("--control", default="")
  ap.add_argument("--e0f", action="store_true", help="sites read the prologue's FP32 e0 (BT6PlanesPrologue e0_fp32=1)")
  ap.add_argument("--triplet-plugin", action="store_true", help="edge sites as BT6TripletSite plugin nodes")
  ap.add_argument("--torch-onnx", default="", help="reuse this step-1 file (cut torch export, gate-fixed)")
  a = ap.parse_args()
  torch.set_grad_enabled(False)
  t0 = time.time()
  tensors, cfg = load_carrier()
  print("config", cfg, f"read {time.time() - t0:.0f}s", flush=True)
  if a.torch_onnx and os.path.exists(a.torch_onnx):
    m = onnx.load(a.torch_onnx)
  else:
    model = (BT6K1T if a.triplet_plugin else BT6K1F if a.e0f else BT6K1)(tensors, cfg, torch.float16).eval()
    B = a.batch
    planes = torch.from_numpy(real_planes(0, B))
    e0 = torch.zeros(B, 16, 64, 64) if a.e0f else torch.zeros(B, 64, 64, 16, dtype=torch.float16)
    pol = torch.zeros(B, 64, 64)
    atts = [torch.zeros(B, 64, 1024, dtype=torch.float16) for _ in range(cfg.layers)]
    Hs = [torch.zeros(B, 32, 64, 64, dtype=torch.float16) for _ in cfg.edge_sites]
    names_in = ["planes", "e0f" if a.e0f else "e0", "polpair"] + [f"att{i}" for i in range(cfg.layers)] + [f"H{i}" for i in cfg.edge_sites]
    names_out = ["policy", "wdl", "mlh", "err"] + [f"lanes{i}" for i in range(cfg.layers)] + \
                [f"state{k + 1}" for k in range(len(cfg.edge_sites))]
    kw = {}
    if a.dynamic:
      bd = torch.export.Dim("batch", min=1, max=1024)
      kw["dynamic_shapes"] = ({0: bd}, {0: bd}, {0: bd}, [{0: bd}] * cfg.layers, [{0: bd}] * len(cfg.edge_sites))
    t1 = time.time()
    if a.triplet_plugin:
      names_in = ["planes", "polpair"] + [f"att{i}" for i in range(cfg.layers)]
      names_out = names_out[:4 + cfg.layers]
      args = (planes, pol, atts)
      if a.dynamic:
        kw["dynamic_shapes"] = ({0: bd}, {0: bd}, [{0: bd}] * cfg.layers)
    else:
      args = (planes, e0, pol, atts, Hs)
    prog = torch.onnx.export(model, args, None, dynamo=True, opset_version=18,
                             input_names=names_in, output_names=names_out, optimize=True, **kw)
    m = prog.model_proto
    print(f"torch export {time.time() - t1:.0f}s, {len(m.graph.node)} nodes", flush=True)
    n = apply_gate_fix(m)
    print(f"trt_gate_fix: {n} gated attentions rewritten", flush=True)
    if a.torch_onnx:
      onnx.save(m, a.torch_onnx)
  n = rewrite(m, tensors, cfg, a.control, a.e0f and not a.triplet_plugin, a.triplet_plugin)
  onnx.save(m, a.out)
  print(f"{n} plugin nodes; {len(m.graph.node)} nodes -> {a.out} ({os.path.getsize(a.out) / 1e6:.0f} MB)", flush=True)


if __name__ == "__main__":
  main()

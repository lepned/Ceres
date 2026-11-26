"""Plain-ONNX export of Kovax's BT6 flagship (1024x24x32) from the lc0ex carrier, for plain TensorRT (no plugins).

BT6Served re-implements bt6_net.forward_served (CeresTrain, read-only import) with ONNX / TensorRT friendly ops:
every einsum is a MatMul over a fixed layout, mish = x * tanh(softplus(x)), the prologue code lookup is one Gather.
Compute dtype `cd` (float32 or float16). FP32 islands (always): BT6Relations (exact 0/1 / small integer counts),
the prologue (S, e0), every LayerNorm (the un-normed residual stream reaches |x| > 20,000, x^2 overflows FP16) and
every rms of the edge sites. Everything else runs in `cd`; the served FP16 weights are used as they are.

Input  planes float32 [B, 112, 8, 8] (lc0 classical planes, expand_planes of the masks / values).
Output policy [B, 1858] logits, wdl [B, 3] logits, mlh [B, 1], err [B, 1] (sigmoid)  -- all float32.

    python bt6_export_onnx.py check  [--count N]                 module (FP32) vs forward_served on real positions
    python bt6_export_onnx.py export --dtype fp32|fp16 --batch B --out X.onnx

Environment:
    BT6_CARRIER       the lc0ex carrier of the net (required)
    CERESTRAIN_PY     CeresTrain's src/CeresTrainPy (bt6_net.py, bt6_carrier.py, bt6_relations.py, bt6_planes.py)
    BT6_POSITIONS     optional .npz with real positions ("masks", "values" as in lc0ex's reference files). The export
                      only traces the graph, so without it the start position is used; check / torchref need it.
    BT6_GATE_OUTPUTS  optional .npz with "served_policy" for an extra comparison in check
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, __import__("os").environ.get("CERESTRAIN_PY", "CeresTrain/src/CeresTrainPy"))
from bt6_carrier import read_carrier  # noqa: E402
from bt6_net import CODE_GROUPS, CODE_GROUP_BASES, BT6Config, policy_gather_index  # noqa: E402
from bt6_relations import BT6Relations  # noqa: E402

CARRIER = os.environ.get("BT6_CARRIER")
POSITIONS = os.environ.get("BT6_POSITIONS")


def keep(name: str) -> bool:
  return not ("/i8/" in name or "/quant/" in name or name.endswith("_d1"))


def load_carrier():
  if not CARRIER or not os.path.isfile(CARRIER):
    sys.exit(f"set BT6_CARRIER to the net's lc0ex carrier (*_lc0ex_carrier.pb.gz); got {CARRIER!r}")
  tensors, _ = read_carrier(CARRIER, keep=keep)
  return tensors, BT6Config.from_served(tensors)


def mish(x):
  return x * torch.tanh(F.softplus(x))


def rms32(x, dim, eps):
  x = x.float()
  return x * torch.rsqrt(x.pow(2).mean(dim, keepdim=True) + eps)


class BT6Served(nn.Module):
  def __init__(self, tensors: dict, cfg: BT6Config, cd=torch.float32):
    super().__init__()
    self.cfg, self.cd = cfg, cd
    self._names = {}
    for name, array in tensors.items():
      if array.dtype.kind != "f":
        continue
      ten = torch.from_numpy(np.ascontiguousarray(array).astype(np.float32))
      self._put(name, ten)
    h = cfg.heads
    for i in range(cfg.layers):
      g = f"/encoder{i}/mha/egt"
      # read / door / gate reads of the edge state as one [3h, d_e] map
      self._put(f"{g}/rdg", torch.cat([self._raw(f"{g}/edge_read/w"), self._raw(f"{g}/door/w"),
                                       self._raw(f"{g}/gate/w")], 0))
      self._put(f"{g}/queryT", self._raw(f"{g}/query").transpose(1, 2).contiguous())   # [h, hd, 34]
      self._put(f"{g}/keyT", self._raw(f"{g}/key").transpose(1, 2).contiguous())
    # prologue code weights: codes[b, group, ij] = sum_c W[group, c] E[b, c, ij] + base[group]
    w = torch.zeros(len(CODE_GROUPS), cfg.relation_channels)
    for gi, (first, bits) in enumerate(CODE_GROUPS):
      w[gi, first:first + bits] = 2.0 ** torch.arange(bits)
    self.register_buffer("code_w", w)
    self.register_buffer("code_base", torch.tensor(CODE_GROUP_BASES, dtype=torch.float32).view(1, -1, 1))
    self.register_buffer("gather", policy_gather_index().to(torch.int64))
    self.relations = BT6Relations()

  # buffers: FP32 islands keep float32 copies, the rest is stored in cd
  def _put(self, name, ten):
    key = "w%04d" % len(self._names)
    self._names[name] = key
    self.register_buffer(key, ten)

  def _raw(self, name):
    return getattr(self, self._names[name])

  def T(self, name, dtype=None):
    return self._raw(name).to(self.cd if dtype is None else dtype)

  def dense(self, x, prefix):
    return x @ self.T(f"{prefix}/w") + self.T(f"{prefix}/b")

  def ln(self, x, prefix):
    y = F.layer_norm(x.float(), x.shape[-1:], self.T(f"{prefix}/scale", torch.float32),
                     self.T(f"{prefix}/bias", torch.float32), self.cfg.norm_eps)
    return y.to(self.cd)

  def glu(self, x, prefix):
    packed = self.dense(x, f"{prefix}/dense1")
    gate, up = packed.chunk(2, dim=-1)
    return self.dense(torch.sigmoid(gate) * up, f"{prefix}/dense2")

  def attn_core(self, i, lanes, E, Ei, Ej, S, e):
    """qkvg lanes [B, 64, 4 D] -> (output-gated attention [B, 64, D] before the out GEMM, H)."""
    cfg, cd = self.cfg, self.cd
    B, h, hd, D = lanes.shape[0], cfg.heads, cfg.head_dim, cfg.d_model
    g = f"/encoder{i}/mha/egt"
    lanes = lanes.view(B, 64, 4, h, hd)
    q = lanes[:, :, 0].permute(0, 2, 1, 3)                                    # [B, h, 64, hd]
    k = lanes[:, :, 1].permute(0, 2, 1, 3)
    v = lanes[:, :, 2].permute(0, 2, 1, 3)
    og = lanes[:, :, 3]                                                       # [B, 64, h, hd]
    scores = (q * self.T(f"{g}/qk_scale").view(1, h, 1, 1)) @ k.transpose(-1, -2)
    row = q @ self.T(f"{g}/queryT") + self.T(f"{g}/attack").view(1, h, 1, -1)   # [B, h, 64(i), 34]
    col = k @ self.T(f"{g}/keyT")                                                # [B, h, 64(j), 34]
    rowterm = (row.permute(0, 2, 1, 3) @ Ei).permute(0, 2, 1, 3)               # [B, i, h, j] -> [B, h, i, j]
    colterm = (col.permute(0, 2, 1, 3) @ Ej).permute(0, 2, 3, 1)               # [B, j, h, i] -> [B, h, i, j]
    pair = (self.T(f"{g}/scaled_coefficients") @ E.view(B, -1, 4096)).view(B, h, 64, 64) + self.T(f"{g}/constant_bias")
    rdg = (self.T(f"{g}/rdg") @ e.to(cd).view(B, -1, 4096)).view(B, 3, h, 64, 64)
    gate = rdg[:, 2] + self.T(f"{g}/gate/b").view(1, h, 1, 1)
    H = (scores + rowterm + colterm + S[:, None] * pair + rdg[:, 0]) * (1.0 + rdg[:, 1])
    A = torch.softmax(H, dim=-1) * (2.0 * torch.sigmoid(gate))
    att = (A @ v).permute(0, 2, 1, 3)                                          # [B, 64, h, hd]
    att = att * (cfg.output_gate_scale * torch.sigmoid(og))
    return att.reshape(B, 64, D), H

  def site(self, i, e, H):
    cfg, cd = self.cfg, self.cd
    B, s = e.shape[0], f"/encoder{i}/edge_site"
    eps = cfg.rms_eps
    e_hat = e + (self.T(f"{s}/readback/w") @ H.reshape(B, cfg.heads, 4096)).float()       # [B, 16, 4096] fp32
    if cfg.triplet:
      th, td = cfg.triplet_heads, cfg.triplet_dots
      n = rms32(e_hat, 1, eps).to(cd)
      v = self.T(f"{s}/triplet/value/w") @ n                                               # [B, 32, 4096]
      eg = (self.T(f"{s}/triplet/gate/w") @ n + self.T(f"{s}/triplet/gate/b").view(1, -1, 1)).view(B, 4, th, 64, 64)
      a_in = torch.softmax(eg[:, 0], dim=-1) * torch.sigmoid(eg[:, 1])                     # [B, th, i, k]
      a_out = torch.softmax(eg[:, 2], dim=-2) * torch.sigmoid(eg[:, 3])                    # [B, th, k, i]
      v_in = v[:, :cfg.edge_channels].reshape(B, th, td, 64, 64)                           # [b, h, d, k, j]
      v_out = v[:, cfg.edge_channels:].reshape(B, th, td, 64, 64)                          # [b, h, d, j, k]
      if cfg.triplet_contraction == "path":
        va_in = a_in[:, :, None] @ v_in
        va_out = a_out.transpose(-1, -2)[:, :, None] @ v_out.transpose(-1, -2)
      else:
        va_in = a_in[:, :, None] @ v_in.transpose(-1, -2)
        va_out = a_out.transpose(-1, -2)[:, :, None] @ v_out
      va = torch.cat((va_in.reshape(B, -1, 4096), va_out.reshape(B, -1, 4096)), dim=1)
      e_hat = e_hat + (self.T(f"{s}/triplet/out/w") @ va).float()
    n = rms32(e_hat, 1, eps).to(cd)
    nT = n.view(B, -1, 64, 64).transpose(-1, -2).reshape(B, -1, 4096)
    hidden = self.T(f"{s}/ffn/dense1/w") @ n + self.T(f"{s}/ffn/dense1/b").view(1, -1, 1)
    if cfg.rev_edge:
      hidden = hidden + self.T(f"{s}/ffn/dense1_rev/w") @ nT
    return rms32(e_hat + (self.T(f"{s}/ffn/dense2/w") @ torch.relu(hidden)).float(), 1, eps)

  def forward(self, planes):
    cfg, cd = self.cfg, self.cd
    B, D = planes.shape[0], cfg.d_model
    planes = planes.float().reshape(B, 112, 64)
    # ---- prologue (FP32)
    Eb = self.relations(planes[:, 0:12] > 0.5, planes[:, 13:25] > 0.5)               # bool [B, 34, 64, 64]
    E32 = Eb.float()
    codes = (self.code_w @ E32.view(B, 34, 4096) + self.code_base).long()             # [B, 4, 4096]
    table = self.T("/prologue/mix_codes", torch.float32)
    mixed = torch.index_select(table, 0, codes.reshape(-1)).view(B, 4, 4096, -1).sum(1)  # [B, 4096, 32]
    mixed = mixed.permute(0, 2, 1)                                                     # [B, 32, 4096]
    pc = cfg.pair_channels
    pair = mixed[:, :pc] + self.T("/prologue/offsets", torch.float32).view(1, pc, 4096)
    S = torch.rsqrt(pair.pow(2).mean(1) + cfg.rms_eps).view(B, 64, 64).to(cd)
    e = rms32(mixed[:, pc:] + self.T("/prologue/edge_offsets", torch.float32).view(1, -1, 4096), 1, cfg.rms_eps)
    E = E32.to(cd)
    Ei = E.permute(0, 2, 1, 3)                                                         # [B, i, c, j]
    Ej = E.permute(0, 3, 1, 2)                                                         # [B, j, c, i]
    # ---- embedding
    per_square = planes.permute(0, 2, 1).to(cd)                                        # [B, 64, 112]
    position = self.dense(per_square[:, :, :12].reshape(B, 768), "/attn_body/preproc").view(B, 64, -1)
    x = self.dense(torch.cat((per_square, position), dim=-1), "/attn_body/matmul")
    x = self.ln(mish(x), "/attn_body/ln0")
    x = x * self.T("/attn_body/mult_gate/w") + self.T("/attn_body/position/w").view(64, D)
    x = x + self.T("/attn_body/ffn/alpha/w") * self.glu(x, "/attn_body/ffn")
    x = self.ln(x, "/attn_body/ln1")
    # ---- tower
    for i in range(cfg.layers):
      p = f"/encoder{i}"
      y = self.ln(x, f"{p}/ln1")                                                     # PROF norm
      lanes = self.dense(y, f"{p}/mha/qkvg")                                         # PROF qkvg_gemm
      att, H = self.attn_core(i, lanes, E, Ei, Ej, S, e)                             # PROF attn_core
      x = x + self.T(f"{p}/mha/alpha/w") * self.dense(att, f"{p}/mha/out")           # PROF out_gemm
      y = self.ln(x, f"{p}/ln2")                                                     # PROF norm
      x = x + self.T(f"{p}/ffn/alpha/w") * self.glu(y, f"{p}/ffn")                   # PROF ffn
      if i in cfg.edge_sites:
        e = self.site(i, e, H)
    x = self.ln(x, "/encoder/final_norm")
    # ---- policy
    emb = mish(self.dense(x, "/policy/embedding"))
    Q, K = self.dense(emb, "/policy/Q"), self.dense(emb, "/policy/K")
    logits = (Q @ K.transpose(-1, -2)) * self.T("/policy/qk_scale")
    pb = (self.T("/policy/pair/scaled_coefficients") @ E.view(B, 34, 4096)).view(B, 64, 64)
    logits = logits + S * (pb + self.T("/policy/pair/constant_bias").view(64, 64))
    off = K[:, 56:64] @ self.T("/policy/promotion/w")                                  # [B, 8, 4]
    promo = logits[:, 48:56, 56:64].unsqueeze(-1) + off[:, None, :, 0:3] + off[:, None, :, 3:4]
    record = torch.cat((logits.reshape(B, 4096), promo.reshape(B, 192)), dim=1)
    policy = torch.index_select(record, 1, self.gather)

    def head(prefix):
      sq = mish(self.dense(x, f"{prefix}/embedding")).reshape(B, -1)
      return self.dense(mish(self.dense(sq, f"{prefix}/dense1")), f"{prefix}/dense2")

    wdl = head("/value")
    mlh = torch.relu(head("/moves"))
    # a carrier without the error head (make_carrier.py from the plain export): err = 0.5 placeholder
    err = torch.sigmoid(head("/value_error")) if "/value_error/dense2/w" in self._names else torch.full_like(mlh, 0.5)
    return policy.float(), wdl.float(), mlh.float(), err.float()


def startpos_planes(count):
  """The start position as lc0 classical planes [count, 112, 8, 8] (current board only, all castling rights, plane 111
  ones): enough to trace the graph for an export when no file with real positions is given."""
  p = np.zeros((112, 8, 8), np.float32)
  back = (3, 1, 2, 4, 5, 2, 1, 3)   # R N B Q K B N R as plane offsets (P N B R Q K = 0..5)
  for f in range(8):
    p[0, 1, f] = 1
    p[6, 6, f] = 1
    p[back[f], 0, f] = 1
    p[6 + back[f], 7, f] = 1
  p[104:108] = 1
  p[111] = 1
  return np.repeat(p[None], count, 0)


def real_planes(start, count, need_real=False):
  if not POSITIONS or not os.path.isfile(POSITIONS):
    if need_real:
      sys.exit("set BT6_POSITIONS to an .npz with real positions (masks, values): this mode compares on real positions")
    print("BT6_POSITIONS not set: tracing with the start position", flush=True)
    return startpos_planes(count)
  from bt6_planes import expand_planes
  with np.load(POSITIONS) as a:
    return expand_planes(a["masks"][start:start + count], a["values"][start:start + count]).reshape(-1, 112, 8, 8)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("mode", choices=["check", "export", "torchref"])
  ap.add_argument("--dtype", default="fp32")
  ap.add_argument("--batch", type=int, default=64)
  ap.add_argument("--count", type=int, default=64)
  ap.add_argument("--out", default=None)
  args = ap.parse_args()
  torch.set_grad_enabled(False)
  t0 = time.time()
  tensors, cfg = load_carrier()
  print("config", cfg, f"read {time.time() - t0:.0f}s", flush=True)
  cd = torch.float16 if args.dtype == "fp16" else torch.float32
  model = BT6Served(tensors, cfg, cd).eval()

  if args.mode == "check":
    from bt6_net import forward_served, served_to_torch
    planes = torch.from_numpy(real_planes(0, args.count, need_real=True))
    t = served_to_torch(tensors)
    ref = forward_served(t, planes.reshape(-1, 112, 64), cfg)
    got = model(planes)
    for name, a, b in zip(("policy", "wdl", "mlh", "err"), got, (ref["policy"], ref["wdl"], ref["mlh"], ref["err"])):
      print(f"{name}: max abs diff {float((a - b).abs().max()):.3e}  ref max |x| {float(b.abs().max()):.3e}")
    if os.environ.get("BT6_GATE_OUTPUTS"):
      with np.load(os.environ["BT6_GATE_OUTPUTS"]) as z:
        print("vs saved gate_outputs served_policy (first N):",
              float(np.abs(z["served_policy"][:args.count] - got[0].numpy()).max()))
  elif args.mode == "torchref":
    # FP32 PyTorch reference outputs of this module on the first --count real positions
    planes = torch.from_numpy(real_planes(0, args.count, need_real=True))
    outs = [model(planes[s:s + 64]) for s in range(0, args.count, 64)]
    np.savez(args.out, planes=planes.numpy(), **{n: torch.cat([o[k] for o in outs]).numpy()
                                                 for k, n in enumerate(("policy", "wdl", "mlh", "err"))})
    print("saved", args.out)
  else:
    planes = torch.from_numpy(real_planes(0, args.batch))
    t1 = time.time()
    torch.onnx.export(model, (planes,), args.out, dynamo=True, opset_version=18, external_data=False,
                      input_names=["planes"], output_names=["policy", "wdl", "mlh", "err"], optimize=True)
    print(f"exported {args.out} in {time.time() - t1:.0f}s", flush=True)


if __name__ == "__main__":
  main()

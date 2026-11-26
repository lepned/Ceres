"""INT8 fake-quant emulator of Kovax's BT6 flagship on GPU (FP32 math + quantisation), for choosing the scheme
before any TensorRT build. WSL: python3 emu.py <cmd> ...

  calib  --out calib.npz [--start 2048 --count 2048]     activation statistics per layer / site on real positions
  run    --scheme S [--calib calib.npz] [--out out.npz]   fake-quant forward on the 256 gate positions + gate

Sites per encoder (as lc0ex names them): attn_in = qkvg GEMM input (ln1 out), attn_out = out GEMM input (attention
output), ffn_in = dense1 (gate|up) input (ln2 out), ffn_mid = dense2 input (sigmoid(g) * (u + bu)).
Schemes (comma separated tokens):
  lc0ex                    the carrier's own int8 weights / r vectors (static SmoothQuant), D1 not applied
  w8                       per-output-channel symmetric int8 weights on all four sites (from the FP16 weights)
  a:<site>=<mode>          activation mode per site: tensor (static per-tensor, needs --calib), token (dynamic per
                           row), none (FP);  site 'all' sets all four
  pct=<p>                  static scale from the |x| quantile p (default amax) e.g. pct=0.999999
  skip=<l1/l2..>           layers kept FP
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/mnt/c/Dev/Chess/kovax_bt6_trt")
sys.path.insert(0, __import__("os").environ.get("CERESTRAIN_PY", "CeresTrain/src/CeresTrainPy"))
from bt6_export_onnx import BT6Served, load_carrier, real_planes  # noqa: E402
from metrics import LEGAL, gate  # noqa: E402

SITES = ("attn_in", "attn_out", "ffn_in", "ffn_mid")
SUFFIX = {"mha/qkvg": "attn_in", "mha/out": "attn_out", "ffn/dense1": "ffn_in", "ffn/dense2": "ffn_mid"}
CARRIER_SITE = {"attn_in": ("mha/qkv/i8", "ln1/quant/r"), "attn_out": ("mha/out/i8", "mha/out/i8/r"),
                "ffn_in": ("ffn/dense1/i8", "ln2/quant/r"), "ffn_mid": ("ffn/dense2/i8", "ffn/mid/i8/r")}
TOPK = 256


def site_of(prefix):
  if not prefix.startswith("/encoder") or prefix.startswith("/encoder/"):
    return None, None
  head, rest = prefix[len("/encoder"):].split("/", 1)
  return int(head), SUFFIX.get(rest)


class Emu(BT6Served):
  def setup(self, scheme: str, calib=None, raw=None):
    self.mode = "run"
    self.act = {s: "none" for s in SITES}
    self.sq, self.ol, self.sp = {}, {}, {}
    self.lc = set()
    self.gq = None
    self.d1 = False
    self.xq = set()
    self.extra_k = {}
    self.wmode, self.pct, self.skip, self.wsites = "fp", None, set(), set(SITES)
    for tok in [t for t in scheme.split(",") if t]:
      if tok == "lc0ex":
        self.wmode = "lc0ex"
        self.lc = set(SITES)
      elif tok.startswith("lc:"):
        self.lc = set(tok[3:].split("/"))
      elif tok == "w8":
        self.wmode = "w8"
      elif tok.startswith("sp:"):
        s_, m = tok[3:].split("=")
        for x in (SITES if s_ == "all" else s_.split("/")):
          self.sp[x] = float(m)
      elif tok[:3] in ("a:", "sq:", "ol:") or tok.startswith("a:"):
        kind, rest = tok.split(":", 1)
        s_, m = rest.split("=")
        for x in (SITES if s_ == "all" else s_.split("/")):
          if kind == "a":
            self.act[x] = m
          elif kind == "sq":
            self.sq[x] = float(m)
          else:
            self.ol[x] = int(m)
      elif tok.startswith("xq="):
        self.xq = {p for k in tok[3:].split("/") for p in self.XQ[k]}
      elif tok == "d1":
        self.d1 = True
      elif tok.startswith("gq="):
        self.gq = float(tok[3:])
      elif tok.startswith("pct="):
        self.pct = float(tok[4:])
      elif tok.startswith("ws="):
        self.wsites = set(tok[3:].split("/"))
      elif tok.startswith("only="):
        keep_ = {int(x) for x in tok[5:].split("/")}
        self.skip = set(range(self.cfg.layers)) - keep_
      elif tok.startswith("skip="):
        self.skip = {int(x) for x in tok[5:].split("/")}
      else:
        raise ValueError(tok)
    self.calib = calib
    self.P = {}
    if not hasattr(self, "_d1_orig"):
      self._d1_orig = (self._raw("/value/dense2/w").clone(), self._raw("/value/dense2/b").clone())
    self._raw("/value/dense2/w").copy_(self._d1_orig[0]); self._raw("/value/dense2/b").copy_(self._d1_orig[1])
    if self.d1:
      self._raw("/value/dense2/w").copy_(torch.from_numpy(raw["/value/dense2/w_d1"].astype(np.float32)))
      self._raw("/value/dense2/b").copy_(torch.from_numpy(raw["/value/dense2/b_d1"].astype(np.float32)))
    dev = next(self.buffers()).device
    for i in range(self.cfg.layers):
      if i in self.skip:
        continue
      for suf, site in SUFFIX.items():
        prefix = f"/encoder{i}/{suf}"
        W = self.T(f"{prefix}/w")                                     # [k, n] fp32
        P = {"s": None, "out": None, "W": W, "Wout": None, "tens": None}
        if site in self.lc:
          st, rv = CARRIER_SITE[site]
          p = f"/encoder{i}"
          W8 = torch.from_numpy(raw[f"{p}/{st}/w"].view(np.int8).astype(np.float32)).to(dev)     # [n, k]
          sc = torch.from_numpy(raw[f"{p}/{st}/scale"].astype(np.float32)).to(dev)
          r = torch.from_numpy(raw[f"{p}/{rv}"].astype(np.float32)).to(dev)
          if site == "ffn_in":   # glu pairs [g0 u0 g1 u1 ..] -> [g | u]
            n2 = W8.shape[0] // 2
            order = torch.cat([torch.arange(0, 2 * n2, 2), torch.arange(1, 2 * n2, 2)]).to(dev)
            W8, sc = W8[order], sc[order]
          P["W"] = (W8 * sc[:, None]).T.contiguous() * r[:, None]
          P["r"] = r
          self.P[prefix] = P
          continue
        ac = None
        if calib is not None and f"{i}/{site}/amax_c" in calib:
          ac = torch.from_numpy(calib[f"{i}/{site}/amax_c"].astype(np.float32)).to(dev).clamp_min(1e-6)
        keep = torch.ones(W.shape[0], dtype=torch.bool, device=dev)
        if site in self.ol and self.ol[site] > 0:
          keep[ac.topk(self.ol[site]).indices] = False
          P["out"] = keep
          P["Wout"] = W[~keep]
        if site in self.sq:
          al = self.sq[site]
          wmax = W.abs().amax(1).clamp_min(1e-8)
          sv = ac ** al / wmax ** (1 - al)
          W = W * sv[:, None]   # SmoothQuant: x' = x / sv, W' = sv W
          P["sv"] = sv
        if site in self.sp:
          c = torch.ceil(ac / (self.sp[site] * ac.median())).clamp_min(1)
          P["c"] = c
          self.extra_k.setdefault(site, []).append(int((c - 1).sum()))
        Wk = W[keep]
        if self.wmode == "w8" and site in self.wsites:
          sc = Wk.abs().amax(0, keepdim=True).clamp_min(1e-12) / 127
          Wk = (Wk / sc).round().clamp(-127, 127) * sc
        P["W"] = Wk
        if ac is not None:
          acs = ac / (P["sv"] if "sv" in P else 1.0)
          P["tens"] = float(acs[keep].max())
        self.P[prefix] = P
    return self

  def calib_rec(self, x, i, site):
    a = x.detach().abs().float().reshape(-1, x.shape[-1])
    st = self.stats.setdefault((i, site), {"amax_c": torch.zeros(x.shape[-1], device=x.device), "top": None,
                                           "n": 0, "row": []})
    st["amax_c"] = torch.maximum(st["amax_c"], a.amax(0))
    t = a.reshape(-1).topk(TOPK).values
    st["top"] = t if st["top"] is None else torch.cat([st["top"], t]).topk(TOPK).values
    st["n"] += a.numel()
    st["row"].append(a.amax(1).cpu())

  def q(self, x, i, site, P):
    m = self.act[site]
    if m == "none":
      return x
    if m == "token":
      s = x.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 127
    elif m == "tensor":
      if self.pct is not None and "sv" not in P and P["out"] is None:
        s = float(self.calib[f"{i}/{site}/p%g" % self.pct]) / 127
      else:
        s = P["tens"] / 127
    else:
      raise ValueError(m)
    return (x / s).round().clamp(-127, 127) * s

  XQ = {"emb": ("/attn_body/ffn/dense1", "/attn_body/ffn/dense2"), "pol": ("/policy/embedding", "/policy/Q", "/policy/K"),
        "val": ("/value/embedding", "/moves/embedding", "/value_error/embedding")}

  def dense(self, x, prefix):
    if self.mode == "run" and prefix in getattr(self, "xq", ()):
      W = self.T(f"{prefix}/w")
      sc = W.abs().amax(0, keepdim=True).clamp_min(1e-12) / 127
      s_ = x.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 127
      return ((x / s_).round().clamp(-127, 127) * s_) @ ((W / sc).round().clamp(-127, 127) * sc) + self.T(f"{prefix}/b")
    y = self.dense_(x, prefix)
    i, site = site_of(prefix)
    if self.mode == "run" and site == "ffn_in" and getattr(self, "gq", None) is not None and i not in self.skip:
      gg, uu = y.chunk(2, dim=-1)
      key = "amax" if self.gq == 0 else "p%g" % self.gq
      sg = float(self.calib[f"{i}/ffn_out_g/{key}"]) / 127
      su = float(self.calib[f"{i}/ffn_out_u/{key}"]) / 127
      y = torch.cat([(gg / sg).round().clamp(-127, 127) * sg, (uu / su).round().clamp(-127, 127) * su], -1)
    return y

  def dense_(self, x, prefix):
    i, site = site_of(prefix)
    if site is None or i in self.skip:
      return super().dense(x, prefix)
    if self.mode == "calib":
      self.calib_rec(x, i, site)
      y = super().dense(x, prefix)
      if site == "ffn_in":
        gg, uu = y.chunk(2, dim=-1)
        self.calib_rec(gg, i, "ffn_out_g")
        self.calib_rec(uu, i, "ffn_out_u")
      return y
    P = self.P[prefix]
    b = self.T(f"{prefix}/b")
    if "r" in P:
      r = P["r"]
      return ((x * r).round().clamp(-127, 127) / r) @ P["W"] + b
    y = b
    if P["out"] is not None:
      y = y + x[..., ~P["out"]] @ P["Wout"]
      x = x[..., P["out"]]
    if "sv" in P:
      sv = P["sv"] if P["out"] is None else P["sv"][P["out"]]
      x = x / sv
    if "c" in P:
      return (self.q(x / P["c"], i, site, P) * P["c"]) @ P["W"] + y
    return self.q(x, i, site, P) @ P["W"] + y


def load(dev):
  tensors, cfg = load_carrier()
  model = Emu(tensors, cfg, torch.float32).eval().to(dev)
  return model, cfg


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("cmd", choices=["calib", "run"])
  ap.add_argument("--scheme", default="")
  ap.add_argument("--schemes", default="", help="several schemes separated by ';'")
  ap.add_argument("--calib", default="")
  ap.add_argument("--out", default="")
  ap.add_argument("--start", type=int, default=2048)
  ap.add_argument("--count", type=int, default=2048)
  ap.add_argument("--n", type=int, default=256)
  ap.add_argument("--brief", action="store_true")
  a = ap.parse_args()
  torch.set_grad_enabled(False)
  dev = "cuda"
  model, cfg = load(dev)
  raw = None
  if a.cmd == "calib":
    model.setup("")
    model.mode, model.stats = "calib", {}
    t0 = time.time()
    for s in range(a.start, a.start + a.count, 64):
      model(torch.from_numpy(real_planes(s, 64)).to(dev))
    out = {}
    for (i, site), st in model.stats.items():
      top = st["top"].cpu().numpy().astype(np.float64)   # descending
      out[f"{i}/{site}/amax"] = top[0]
      out[f"{i}/{site}/amax_c"] = st["amax_c"].cpu().numpy()
      out[f"{i}/{site}/row"] = torch.cat(st["row"]).numpy()
      for p in (0.99999, 0.999999, 0.9999999):
        k = int(round((1 - p) * st["n"]))
        out[f"{i}/{site}/p%g" % p] = top[min(k, TOPK - 1)]
    np.savez(a.out, **out)
    print(f"calibrated on positions {a.start}..{a.start + a.count - 1} in {time.time() - t0:.0f}s -> {a.out}")
    for site in SITES:
      print(site, " ".join("%d:%.1f/%.1f" % (i, out[f"{i}/{site}/amax"], out[f"{i}/{site}/p1e-06".replace('1e-06', '0.999999')])
                           for i in range(cfg.layers)))
    return
  calib = dict(np.load(a.calib)) if a.calib else None
  ref = dict(np.load("/mnt/c/Dev/Chess/kovax_bt6_trt/ref_fp32_256.npz"))
  legal = np.load(LEGAL)["legal"]
  planes = torch.from_numpy(ref["planes"][:a.n]).to(dev)
  for scheme in (a.schemes.split(";") if a.schemes else [a.scheme]):
    if ("lc" in scheme or "d1" in scheme) and raw is None:
      raw = {k.replace("|", "/"): v for k, v in np.load("/mnt/c/Dev/Chess/kovax_bt6_trt/int8/carrier_all.npz").items()
             if "/i8/" in k.replace("|", "/") or "/quant/" in k.replace("|", "/") or k.endswith("_d1")}
    model.setup(scheme, calib, raw)
    if model.extra_k:
      print("  extra K per site (mean over layers):", {k: float(np.mean(v)) for k, v in model.extra_k.items()},
            "max", {k: max(v) for k, v in model.extra_k.items()})
    outs = [model(planes[s:s + 64]) for s in range(0, a.n, 64)]
    got = {n: torch.cat([o[k] for o in outs]).cpu().numpy() for k, n in enumerate(("policy", "wdl", "mlh", "err"))}
    txt = gate(got, {k: ref[k][:a.n] for k in got}, legal, f"emu [{scheme}]")
    if a.brief:
      import re
      L = txt.split(chr(10))
      wl = re.findall(r"[0-9.e+-]+", L[1].split("err")[1]); t1 = re.search(r"\((\d+)/", L[2]).group(1)
      kl = re.findall(r"[0-9.]+e[+-]\d+", L[3])
      print(f"{scheme:60s} W-L {wl[0]} / {wl[1]}  top1 {t1}  KL {kl[0]} / {kl[1]}", flush=True)
    else:
      print(txt, flush=True)
    if a.out:
      np.savez(a.out, **got)


if __name__ == "__main__":
  main()

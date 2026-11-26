"""Our Q1 analyser (per-site SmoothQuant vectors for an lc0ex INT8 build of a BT6 net) + an emulator of the result.

  vectors:  s_j = A_j^a / wmax_j^(1-a),  D = max_j(Aclip_j / s_j) / 127,  r_j = 1 / (s_j D)        (lab/_quant.py)
            A = per-channel statistic of the site input over the calibration positions (an_calib.py files),
            wmax_j = max_k |W[k, j]| of the site's GEMM (qkvg / out / dense1 gate|up / dense2), FP16 carrier weights.
  rule string: "a=<alpha>,stat=<amax|topK>,clip=<amax|topK>,margin=<headroom factor on D>"   (topK = the (K+1)-th largest |x| of the channel)

  BT6_CARRIER=<fp16 carrier> python analyser.py make  RULE OUT.npz CAL.npz [CAL2.npz ...]
  BT6_CARRIER=<fp16 carrier> python analyser.py emu   REF.npz CAL.npz[,CAL2..] RULE[;RULE..] | file:VECTORS.npz | carrier:CARRIER_ALL.npz
  emu = FP32 PyTorch with, at the four sites of every encoder, codes = clamp(floor(x r + 0.5), +-127), int8 weights
  W8 = round(s W / step) per output column (carrier.py _int8_site), GEMM = (codes D) @ (W8 step) + b.
"""
import sys
import numpy as np, torch
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), ".."))
sys.path.insert(0, __import__("os").environ.get("CERESTRAIN_PY", "CeresTrain/src/CeresTrainPy"))
from bt6_export_onnx import BT6Served, load_carrier
from metrics import LEGAL, gate
SUFFIX = {"mha/qkvg": "attn_in", "mha/out": "attn_out", "ffn/dense1": "ffn_in", "ffn/dense2": "ffn_mid"}
WNAME = {v: k for k, v in SUFFIX.items()}
CARR = {"attn_in": "ln1/quant/r", "attn_out": "mha/out/i8/r", "ffn_in": "ln2/quant/r", "ffn_mid": "ffn/mid/i8/r"}


def parse_rule(rule):
  d = {"a": "1", "stat": "amax", "clip": "amax", "margin": "1"}
  d.update(dict(t.split("=") for t in rule.split(",") if t))
  return d


def pick(top, name):
  return top[0] if name == "amax" else top[int(name[3:])]


def site_rule(rule, site):
  """per-site override: a=0.5,a@ffn_mid=0.7"""
  d = dict(rule)
  for k, v in rule.items():
    if "@" in k and k.split("@")[1] == site:
      d[k.split("@")[0]] = v
  return d


def make_vectors(tensors, layers, cals, rule):
  rule = parse_rule(rule)
  out = {}
  for i in range(layers):
    for site, wn in WNAME.items():
      rr = site_rule(rule, site)
      a = float(rr["a"])
      top = -np.sort(-np.concatenate([c[f"{i}/{site}/top"] for c in cals]).astype(np.float64), 0)
      A, Ac = np.maximum(pick(top, rr["stat"]), 1e-12), np.maximum(pick(top, rr["clip"]), 1e-12)
      W = tensors[f"/encoder{i}/{wn}/w"].astype(np.float64)                        # [k, n]
      wmax = np.maximum(np.abs(W).max(1), 1e-12)
      s = A ** a / wmax ** (1.0 - a)
      D = (Ac / s).max() / 127.0 * float(rr["margin"])
      r = 1.0 / (s * D)
      p = f"encoder{i}/{site}/"
      out[p + "r"], out[p + "s"], out[p + "amax"] = r.astype(np.float32), s.astype(np.float32), top[0].astype(np.float32)
      # r * s must be one constant to 1e-4 after float32 rounding (the reader's check): true by construction
  return out


class Q1Emu(BT6Served):
  P = None
  def setq(self, vec):
    """vec: dict encoder<i>/<site>/{r,s} (numpy) or None (FP)."""
    self.P = {}
    if vec is None:
      return
    dev = next(self.buffers()).device
    for i in range(self.cfg.layers):
      for suf, site in SUFFIX.items():
        prefix = f"/encoder{i}/{suf}"
        W = self.T(f"{prefix}/w").double()
        r = torch.from_numpy(vec[f"encoder{i}/{site}/r"].astype(np.float64)).to(dev)
        s = torch.from_numpy(vec[f"encoder{i}/{site}/s"].astype(np.float64)).to(dev)
        D = float((1.0 / (r * s)).mean())
        Wf = W * s[:, None]
        step = Wf.abs().amax(0) / 127.0
        step = torch.where(step > 0, step, torch.ones_like(step))
        Wq = (torch.round(Wf / step).clamp(-127, 127) * step * D).float()
        self.P[prefix] = (r.float(), Wq)

  def dense(self, x, prefix):
    if self.P and prefix in self.P:
      r, Wq = self.P[prefix]
      return torch.floor(x * r + 0.5).clamp(-127, 127) @ Wq + self.T(f"{prefix}/b")
    return super().dense(x, prefix)


def carrier_vectors(path, layers):
  """the 5.4M carrier's own r as a vector set (s from r with D = 1: only r s = const matters to the emulator)."""
  C = np.load(path)
  out = {}
  for i in range(layers):
    for site, rn in CARR.items():
      r = C[f"/encoder{i}/{rn}".replace("/", "|")].astype(np.float64)
      out[f"encoder{i}/{site}/r"], out[f"encoder{i}/{site}/s"] = r, 1.0 / r
  return out


def brief(txt):
  import re
  L = txt.split(chr(10))
  wl = re.findall(r"[0-9.e+-]+", L[1].split("err")[1]); t1 = re.search(r"\((\d+)/", L[2]).group(1)
  kl = re.findall(r"[0-9.]+e[+-]\d+", L[3])
  return f"W-L {wl[0]} / {wl[1]}  top1 {t1}  KL {kl[0]} / {kl[1]}"


def main():
  cmd = sys.argv[1]
  torch.set_grad_enabled(False)
  tensors, cfg = load_carrier(); import bt6_export_onnx as _m; print("carrier", _m.CARRIER[-70:], flush=True)
  if cmd == "make":
    rule, out, cals = sys.argv[2], sys.argv[3], [np.load(c) for c in sys.argv[4:]]
    vec = make_vectors(tensors, cfg.layers, cals, rule)
    np.savez(out, **vec)
    print(f"wrote {out}: {len(vec)} arrays, rule {rule}, calibration {[ (int(c['start']), int(c['count'])) for c in cals]}")
    return
  ref = dict(np.load(sys.argv[2])); cals = [np.load(c) for c in sys.argv[3].split(",") if c]
  legal = np.load(LEGAL)["legal"]
  model = Q1Emu(tensors, cfg, torch.float32).eval().to("cuda")
  planes = torch.from_numpy(ref["planes"]).to("cuda")
  for spec in sys.argv[4].split(";"):
    if spec == "fp":
      vec = None
    elif spec.startswith("file:"):
      vec = dict(np.load(spec[5:]))
    elif spec.startswith("carrier:"):
      vec = carrier_vectors(spec[8:], cfg.layers)
    else:
      vec = make_vectors(tensors, cfg.layers, cals, spec)
    model.setq(vec)
    outs = [model(planes[s:s + 64]) for s in range(0, len(planes), 64)]
    got = {n: torch.cat([o[k] for o in outs]).cpu().numpy() for k, n in enumerate(("policy", "wdl", "mlh", "err"))}
    if "err" not in ref:
      ref["err"] = got["err"]
    print(f"{spec:44s} {brief(gate(got, {k: ref[k] for k in got}, legal, spec))}", flush=True)
    if len(sys.argv) > 5:
      np.savez(sys.argv[5], **got)


if __name__ == "__main__":
  main()

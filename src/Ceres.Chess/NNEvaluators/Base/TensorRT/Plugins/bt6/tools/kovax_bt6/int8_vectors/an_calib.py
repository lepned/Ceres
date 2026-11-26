"""Analyser, part 1: per-channel activation statistics of the four Q1 sites of every encoder (FP32 PyTorch, BT6Served).
   BT6_CARRIER=<carrier> python an_calib.py OUT.npz [START COUNT]
Sites = inputs of the four GEMMs: attn_in (ln1 output -> qkvg), attn_out (attention output -> out), ffn_in (ln2 output
-> dense1), ffn_mid (sigmoid(g) * (u + bu) -> dense2).
Per layer/site: top[K, C] (K largest |x| per channel over [START, START+COUNT)), absmean, rms, mean, chunk[nchunk, C]
(amax per 256 positions), n."""
import sys, time
import numpy as np, torch
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), ".."))
sys.path.insert(0, __import__("os").environ.get("CERESTRAIN_PY", "CeresTrain/src/CeresTrainPy"))
from bt6_export_onnx import BT6Served, load_carrier, real_planes
SUFFIX = {"mha/qkvg": "attn_in", "mha/out": "attn_out", "ffn/dense1": "ffn_in", "ffn/dense2": "ffn_mid"}
K = 64

class Cal(BT6Served):
  stats = None
  def dense(self, x, prefix):
    if self.stats is not None and prefix.startswith("/encoder") and not prefix.startswith("/encoder/"):
      head, rest = prefix[len("/encoder"):].split("/", 1)
      site = SUFFIX.get(rest)
      if site is not None:
        a = x.detach().float().reshape(-1, x.shape[-1])
        st = self.stats.setdefault((int(head), site), {"top": None, "sa": 0, "sq": 0, "sm": 0, "n": 0, "cur": None})
        ab = a.abs()
        t = ab.topk(K, dim=0).values
        st["top"] = t if st["top"] is None else torch.cat([st["top"], t]).topk(K, dim=0).values
        st["sa"] = st["sa"] + ab.double().sum(0); st["sq"] = st["sq"] + a.double().pow(2).sum(0)
        st["sm"] = st["sm"] + a.double().sum(0); st["n"] += a.shape[0]
        m = ab.amax(0)
        st["cur"] = m if st["cur"] is None else torch.maximum(st["cur"], m)
    return super().dense(x, prefix)

def main():
  out = sys.argv[1]; start = int(sys.argv[2]) if len(sys.argv) > 2 else 2048; count = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
  torch.set_grad_enabled(False)
  tensors, cfg = load_carrier()
  model = Cal(tensors, cfg, torch.float32).eval().to("cuda")
  model.stats = {}
  chunks = {}
  t0 = time.time()
  for s in range(start, start + count, 64):
    model(torch.from_numpy(real_planes(s, 64)).to("cuda"))
    if (s + 64 - start) % 256 == 0 or s + 64 >= start + count:
      for k, st in model.stats.items():
        chunks.setdefault(k, []).append(st["cur"].cpu().numpy()); st["cur"] = None
  res = {"start": start, "count": count}
  for (i, site), st in model.stats.items():
    p = f"{i}/{site}/"
    res[p + "top"] = st["top"].cpu().numpy(); res[p + "absmean"] = (st["sa"] / st["n"]).cpu().numpy()
    res[p + "rms"] = (st["sq"] / st["n"]).sqrt().cpu().numpy(); res[p + "mean"] = (st["sm"] / st["n"]).cpu().numpy()
    res[p + "chunk"] = np.stack(chunks[(i, site)]); res[p + "n"] = st["n"]
  np.savez(out, **res)
  print(f"calibrated positions {start}..{start + count - 1} in {time.time() - t0:.0f}s -> {out}; layers {cfg.layers}")

main()

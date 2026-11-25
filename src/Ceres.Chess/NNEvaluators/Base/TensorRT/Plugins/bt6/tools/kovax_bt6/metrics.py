"""Gate metrics (numpy only): got vs reference outputs on the same real positions, policy over legal moves."""
import numpy as np

LEGAL = "/mnt/c/Dev/Chess/lc0ex_work/bt6_torch/legal.npz"


def wl(logits):
  z = logits.astype(np.float64)
  p = np.exp(z - z.max(-1, keepdims=True))
  p /= p.sum(-1, keepdims=True)
  return p[:, 0] - p[:, 2]


def lsm(z, mask):
  z = np.where(mask, z.astype(np.float64), -np.inf)
  m = z.max(-1, keepdims=True)
  lse = m + np.log(np.exp(z - m).sum(-1, keepdims=True))
  return z - lse


def gate(got: dict, ref: dict, legal: np.ndarray, label: str) -> str:
  n = len(got["policy"])
  legal = legal[:n]
  dwl = np.abs(wl(got["wdl"]) - wl(ref["wdl"]))
  lr, lg = lsm(ref["policy"], legal), lsm(got["policy"], legal)
  pr = np.exp(lr)
  kl = (pr * (np.where(legal, lr, 0) - np.where(legal, lg, 0))).sum(-1)
  top_ref = np.where(legal, ref["policy"], -np.inf).argmax(-1)
  top_got = np.where(legal, got["policy"], -np.inf).argmax(-1)
  dpol = np.abs(np.where(legal, got["policy"] - ref["policy"], 0)).max()
  lines = [f"{label}: {n} positions",
           f"  W-L   mean abs err {dwl.mean():.5f}  max {dwl.max():.5f}",
           f"  policy top-1 (legal) agreement {(top_ref == top_got).mean() * 100:.2f} % ({(top_ref == top_got).sum()}/{n})",
           f"  policy KL(ref||got) legal  mean {kl.mean():.3e}  max {kl.max():.3e};  max |dlogit| legal {dpol:.4f}",
           f"  mlh   mean abs err {np.abs(got['mlh'] - ref['mlh']).mean():.4f}  max {np.abs(got['mlh'] - ref['mlh']).max():.4f}",
           f"  err   mean abs err {np.abs(got['err'] - ref['err']).mean():.5f}  max {np.abs(got['err'] - ref['err']).max():.5f}"]
  return "\n".join(lines)

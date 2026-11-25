"""Gate an output .npz (policy, wdl, mlh, err) vs PyTorch FP32 ref_fp32_256.npz.  python npz_gate.py out.npz [label]"""
import sys
sys.path.insert(0, "/mnt/c/Dev/Chess/kovax_bt6_trt")
import numpy as np
from metrics import LEGAL, gate
got = dict(np.load(sys.argv[1]))
n = len(got["policy"])
ref = dict(np.load("/mnt/c/Dev/Chess/kovax_bt6_trt/ref_fp32_256.npz"))
legal = np.load(LEGAL)["legal"]
print(gate({k: got[k][:n] for k in ("policy", "wdl", "mlh", "err")}, {k: ref[k][:n] for k in ("policy", "wdl", "mlh", "err")},
           legal, sys.argv[2] if len(sys.argv) > 2 else sys.argv[1]))

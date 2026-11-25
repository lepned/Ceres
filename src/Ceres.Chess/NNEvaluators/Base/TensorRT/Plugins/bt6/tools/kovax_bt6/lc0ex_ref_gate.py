"""The real lc0ex FP16 program (reference_7040.npz, recorded 11-13) vs PyTorch FP32 on the same first N positions."""
import sys
import numpy as np
from metrics import LEGAL, gate
n = int(sys.argv[1])
prog = dict(np.load("/mnt/c/Dev/Chess/lc0ex_work/bt6_torch/reference_7040.npz"))
ref = dict(np.load("ref_fp32_256.npz"))
legal = np.load(LEGAL)["legal"]
print(gate({k: prog[k][:n] for k in ("policy", "wdl", "mlh", "err")}, {k: ref[k][:n] for k in ("policy", "wdl", "mlh", "err")},
           legal, "real lc0ex FP16 program vs PyTorch FP32"))

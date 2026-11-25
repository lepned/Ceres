"""Gate a trtexec --exportOutput JSON (real positions planes_bB.bin = first B of ref_fp32_256.npz) vs PyTorch FP32.
    python trt_gate.py out.json [label]"""
import json
import sys

import numpy as np

from metrics import LEGAL, gate

out = {d["name"]: np.array(d["values"], dtype=np.float32).reshape([int(x) for x in d["dimensions"].split("x")])
       for d in json.load(open(sys.argv[1]))}
n = len(out["policy"])
ref = dict(np.load("ref_fp32_256.npz"))
legal = np.load(LEGAL)["legal"]
print(gate(out, {k: ref[k][:n] for k in ("policy", "wdl", "mlh", "err")}, legal,
           sys.argv[2] if len(sys.argv) > 2 else sys.argv[1]))

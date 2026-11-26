"""Cache every tensor of an lc0ex carrier as one .npz (names with "/" -> "|"), for the INT8 scripts.
usage: BT6_CARRIER=<carrier.pb.gz> python cache_carrier.py <out carrier_all.npz>"""
import os, sys, time, numpy as np
sys.path.insert(0, __import__("os").environ.get("CERESTRAIN_PY", "CeresTrain/src/CeresTrainPy"))
from bt6_carrier import read_carrier
C = os.environ.get("BT6_CARRIER")
if not C or len(sys.argv) < 2: sys.exit(__doc__)
t = time.time(); tens, _ = read_carrier(C); print(len(tens), time.time() - t)
np.savez(sys.argv[1], **{k.replace("/", "|"): v for k, v in tens.items()})

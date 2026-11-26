import sys, time, numpy as np
sys.path.insert(0, __import__("os").environ.get("CERESTRAIN_PY", "CeresTrain/src/CeresTrainPy"))
from bt6_carrier import read_carrier
C = ("/mnt/c/lc0_bt6/net/bt6_flagship_ffnmuon_1024x24x32_5400000_swa_blend_grill_0.4315_optimistic_st_0.5685_"
     "winner_q1i8_v2c62ccc4_d1e4b030c2_with_error_head_st_lc0ex_carrier.pb.gz")
t = time.time(); tens, _ = read_carrier(C); print(len(tens), time.time() - t)
np.savez("/mnt/c/Dev/Chess/kovax_bt6_trt/int8/carrier_all.npz", **{k.replace("/", "|"): v for k, v in tens.items()})

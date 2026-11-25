"""Run the real lc0ex INT8 program on the first N gate positions (bt6_torch/positions.npz) -> npz.
   ~/lc0ex/lczero-triton/.venv/bin/python run_lc0ex_int8.py N BATCH OUT.npz"""
import sys
sys.path.insert(0, "/mnt/c/Dev/Chess/lc0ex_work/bt6_torch")
sys.path.insert(0, "/mnt/c/Users/lepne/source/repos/CeresTrain/.claude/worktrees/agent-a28190fa09ace81ed/src/CeresTrainPy")
from pathlib import Path
import numpy as np
from bt6_carrier import NUMPY_TO_ONNX, read_carrier
from lc0ex_runner_orig import Runner, load_executable
N, B, OUT = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
PROG = Path("/mnt/c/lc0_bt6/net/bt6_4m_wide_err_int8_dense_rungs_aabe871_5090_warm_from_s6_1028_sm120.lc0ex")
CARR = "/mnt/c/lc0_bt6/net/bt6_flagship_ffnmuon_1024x24x32_5400000_swa_blend_grill_0.4315_optimistic_st_0.5685_winner_q1i8_v2c62ccc4_d1e4b030c2_with_error_head_st_lc0ex_carrier.pb.gz"
ex = load_executable(PROG)
wanted = {b.name for b in ex.buffers}
tensors, _ = read_carrier(CARR, keep=lambda n: n in wanted)
carrier = {n: (NUMPY_TO_ONNX[a.dtype], tuple(a.shape), a.tobytes()) for n, a in tensors.items()}
print("missing from carrier:", sorted(wanted - set(tensors))[:20])
pos = np.load("/mnt/c/Dev/Chess/lc0ex_work/bt6_torch/positions.npz")
masks, values = pos["masks"][:N], pos["values"][:N]
runner = Runner(ex, carrier, 0)
prog = runner.program(None, B)
OUTS = ("/output/policy", "/output/wdl", "/output/mlh", "/output/err")
col = {n: [] for n in OUTS if n in prog.buffers}
for s in range(0, N, B):
    m, v = masks[s:s + B], values[s:s + B]
    c = len(m)
    if c < B:
        m = np.concatenate([m, np.repeat(m[-1:], B - c, 0)]); v = np.concatenate([v, np.repeat(v[-1:], B - c, 0)])
    prog.write("/input/plane_masks", m); prog.write("/input/plane_values", v); prog.run()
    for n in col: col[n].append(prog.read(n)[:c])
out = {n.rsplit("/", 1)[1]: np.concatenate(p) for n, p in col.items()}
np.savez(OUT, **out)
print({k: v.shape for k, v in out.items()}, "used buffers with i8:", sum('i8' in n for n in wanted), "fp16 qkvg used:", '/encoder0/mha/qkvg/w' in wanted)

# Round 3 gate (extends gate.py): TensorRT 11.3 strongly typed with the plugin library loaded, against a reference.
#   gate_r3.py <onnx> <plugin.dll> <squares.bin|positions.npz> <N> <ref.npz> [--tile K] [--start S] [--save out.npz]
#              [--batch B]
# squares.bin: FP16 [N,64,137] (as gate.py). positions.npz: lc0ex squares_byte, positions S..S+N (signed / 100 unless
#   the npz says unsigned; the lc0ex reference ref_h768.npz carries its own 256 positions, so it can be both arguments).
# ref.npz: the PyTorch FP32 reference (policy/value/value2), or the --save output of another engine (for example the
#          FP16 plugin engine on a second position set, to compare INT8 variants without tuning on the gate set).
# Prints, besides gate.py's W-L mean/max and policy top-1, the policy KL(ref || trt) over legal moves (mean / max) and
# the mean |logit diff| over legal moves: these see small systematic errors (a missing small bias) that W-L and top-1
# do not.
import argparse, os, sys, time
import numpy as np
for d in (r'C:\Dev\TensorRT-11.3.0.99\bin', r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin'):
    if os.path.isdir(d):
        os.add_dll_directory(d)
import tensorrt as trt
from polygraphy.backend.trt import NetworkFromOnnxPath, CreateConfig, EngineFromNetwork, TrtRunner, Profile

ap = argparse.ArgumentParser()
ap.add_argument('onnx'); ap.add_argument('plugin'); ap.add_argument('squares'); ap.add_argument('n', type=int)
ap.add_argument('ref'); ap.add_argument('--tile', type=int, default=1); ap.add_argument('--save', default='')
ap.add_argument('--start', type=int, default=0); ap.add_argument('--batch', type=int, default=0)
a = ap.parse_args()

logger = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(logger, '')
handle = trt.get_plugin_registry().load_library(a.plugin)
assert handle is not None, 'plugin library did not load'
if a.squares.endswith('.npz'):
    P = np.load(a.squares)
    b8 = P['squares_byte'][a.start:a.start + a.n]
    uns = bool(P['unsigned']) if 'unsigned' in P else False
    x = ((b8 if uns else b8.view(np.int8)).astype(np.float32) / 100.0).astype(np.float16).reshape(a.n, 64, 137)
else:
    x = np.fromfile(a.squares, dtype=np.float16).reshape(a.n, 64, 137)
ref = dict(np.load(a.ref))
ref = {k: ref[k][:a.n] for k in ('policy', 'value', 'value2')}
if a.tile > 1:
    x = np.concatenate([x] * a.tile)
    ref = {k: np.concatenate([v] * a.tile) for k, v in ref.items()}
N = x.shape[0]
B = a.batch or N
t0 = time.time()
eng = EngineFromNetwork(NetworkFromOnnxPath(a.onnx, strongly_typed=True),
                        config=CreateConfig(profiles=[Profile().add('squares', (1, 64, 137), (B, 64, 137), (B, 64, 137))]))
with TrtRunner(eng) as r:
    parts = []
    for i in range(0, N, B):   # copy each batch: polygraphy reuses its output buffers
        o = r.infer({'squares': x[i:i + B]})
        parts.append({k: np.asarray(o[k]).astype(np.float64) for k in ('policy', 'value', 'value2')})
    y = {k: np.concatenate([p[k] for p in parts]) for k in ('policy', 'value', 'value2')}
print(f'TensorRT {trt.__version__} + plugin: built + run {time.time() - t0:.0f} s, {N} positions, batch {B}')
if a.save:
    np.savez(a.save, **{k: y[k] for k in ('policy', 'value', 'value2')})

def wl(z):
    p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True); return p[:, 0] - p[:, 2]

line = []
for v in ('value', 'value2'):
    d = np.abs(wl(y[v]) - wl(ref[v].astype(np.float64)))
    line.append(f'{v} W-L mean {d.mean():.4f} max {d.max():.4f}')
pr = ref['policy'].astype(np.float64); keep = pr > -1e3
top = int(sum(np.argmax(np.where(keep[i], y['policy'][i], -1e9)) == np.argmax(np.where(keep[i], pr[i], -1e9)) for i in range(N)))
kl, mad = [], []
for i in range(N):
    k = keep[i]; p = pr[i][k]; q = y['policy'][i][k]
    lp = p - p.max(); lp -= np.log(np.exp(lp).sum())
    lq = q - q.max(); lq -= np.log(np.exp(lq).sum())
    kl.append(float((np.exp(lp) * (lp - lq)).sum())); mad.append(float(np.abs((p - p.mean()) - (q - q.mean())).mean()))
line.append(f'policy top-1 {top}/{N}, max |logit diff| {np.abs(y["policy"][keep] - pr[keep]).max():.3f}')
line.append(f'policy KL mean {np.mean(kl):.2e} max {np.max(kl):.2e}, mean |centred logit diff| {np.mean(mad):.4f}')
line.append(f'finite {all(np.isfinite(y[k]).all() for k in y)}')
print('  ' + ';  '.join(line))

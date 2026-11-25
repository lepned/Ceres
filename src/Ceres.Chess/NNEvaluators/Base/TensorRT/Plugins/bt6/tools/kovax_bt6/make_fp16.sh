set -e
cd /mnt/c/Dev/Chess/kovax_bt6_trt
PY=/home/lep/cerestrain-env/bin/python
for B in 16 64 128 256; do
  $PY bt6_export_onnx.py export --dtype fp16 --batch $B --out bt6_fp16_b${B}_raw.onnx 2>&1 | tail -1
  $PY /mnt/c/Users/lepne/source/repos/CeresTrain/src/CeresTrainPy/trt_gate_fix.py bt6_fp16_b${B}_raw.onnx bt6_fp16_b${B}.onnx
  $PY -c "
import numpy as np
a=np.load('ref_fp32_256.npz')['planes']
p=a[:$B] if $B<=256 else a
p.astype(np.float32).tofile('planes_b$B.bin'); print('planes_b$B.bin', p.shape)"
done

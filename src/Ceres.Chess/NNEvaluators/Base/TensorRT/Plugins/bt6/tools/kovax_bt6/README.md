# lc0 BT6 nets (Kovax's BT6 flagship) in TensorRT with the bt6 plugins

Scripts that turn an lc0ex carrier of an lc0 BT6 net (1024 wide, 24 layers, 32 heads, edge sites with triplet) into
ONNX files TensorRT runs with this folder's plugin library. Dev-box paths are hard-coded (as in CeresTrain's
make_plugin_onnx.py); the PyTorch reference of the net comes from CeresTrain `src/CeresTrainPy` (bt6_net.py,
bt6_relations.py, bt6_carrier.py, trt_gate_fix.py) on the Python path. Torch steps run in WSL, graph steps and gates
with the TensorRT Python package on Windows.

## Chain

| Step | Script | Output |
|---|---|---|
| plain ONNX from the carrier (forward_served rebuilt with MatMul/Gather; static batch) | `bt6_export_onnx.py export`, `make_fp16.sh` | `bt6_fp16_b<B>.onnx` (gate-fixed) |
| FP16 plugin ONNX (dynamic batch): BT6PlanesPrologue, 24 x BT6EgtAttention, 5 x BT6TripletSite | `bt6_k1_onnx.py` (tables: `triplet_site_tables.py`) | `k1_dyn_tri.onnx` |
| INT8 plugin ONNX: the carrier's static SmoothQuant ("q1i8") on the LayerNorm-fed GEMMs, per-token INT8 on attention out (BT6RowQuant) and FFN down (BT6SigGluQuant) | `k2_int8_onnx.py` | `k2_hyb_glurq.onnx` |
| Mycel / Ceres I/O: `/input/planes` FP16, `/output/policy`, `/output/wdl` (softmax), `/output/mlh` FP16 | `to_lc0_io.py in.onnx out.onnx` | engine-ready ONNX |
| gates vs PyTorch FP32 on 256 real positions | `trt_gate.py`, `metrics.py`, `npz_gate.py`, `lc0ex_ref_gate.py`, `run_lc0ex_int8.py` (the lc0ex INT8 program on the same positions), `emu.py` (INT8 emulator) | W-L, top-1, KL |
| build + gate / time | `k1_build_gate.sh`, `k1_time.sh`, `k2_build_gate.sh`, `k2_time.sh`, `time_trt.sh`; backward compatibility with our nets: `gate_n400M_k1.sh` | |

## Results (BT6 5.4M carrier, RTX 5090, TensorRT 11.3, 2026-11-25)

Gate on 256 real positions vs PyTorch FP32 (W-L mean, top-1, KL mean):

| | W-L | top-1 | KL |
|---|---|---|---|
| plain TensorRT FP16 | 0.00106 | 254 | 3.7e-5 |
| plugins FP16 (k1) | 0.00075 | 255 | 2.8e-5 |
| lc0ex FP16 program | 0.00082 | 255 | 1.8e-5 |
| plugins INT8 (k2) | 0.0093 | 251 | 1.6e-3 |
| lc0ex INT8 program | 0.0112 | 248 | 2.8e-3 |

trtexec median ms (CUDA graph, real positions):

| batch | plain FP16 | plugins FP16 | plugins INT8 | lc0ex INT8 |
|---|---|---|---|---|
| 16 | 5.05 | 2.9 | 2.34 | 2.00 |
| 64 | 17.1 | 9.3-9.6 | 6.3 | 5.99 |
| 128 | 35.3 | 18.0-18.5 | 12.8 | 11.80 |
| 256 | 75.6 | 37.1-38.0 | 25.9 | - |

Mycel (MCGS), EngineBattle analyze startpos 30 s: plugins INT8 10.1K eps vs lc0ex INT8 10.4K; plugins FP16 7.0K vs
lc0ex FP16 (batch-64 program) 6.2K. Puzzles without tablebases (4000 per band, rg2500 / rg2700): plugins FP16 equal
to lc0ex FP16 within 3, plugins INT8 equal to lc0ex INT8 within the noise.

# lc0 BT6 nets (Kovax's BT6 flagship) in TensorRT with the bt6 plugins

Scripts that turn an lc0ex carrier of an lc0 BT6 net (1024 wide, 24 layers, 32 heads, edge sites with triplet) into
ONNX files TensorRT runs with this folder's plugin library. The scripts that make the ONNX files (`bt6_export_onnx.py`,
`bt6_k1_onnx.py`, `cache_carrier.py`, `k2_int8_onnx.py`, `k4_onnx.py`, `to_lc0_io.py`) take their inputs from the
environment and the command line: `BT6_CARRIER` (the net's lc0ex carrier, required), `CERESTRAIN_PY` (CeresTrain's
`src/CeresTrainPy`) and optionally `BT6_POSITIONS` (an .npz with real positions; the export only traces the graph and
uses the start position without it; the graph and weights come out the same either way). The gate and timing scripts are the working
scripts of the port and still have dev-box paths hard-coded; they need files with real positions and reference outputs. The PyTorch reference of the
net comes from CeresTrain `src/CeresTrainPy` (bt6_net.py, bt6_relations.py, bt6_carrier.py, bt6_planes.py,
trt_gate_fix.py) on the Python path. Torch steps ran in WSL, graph steps and gates with the TensorRT Python package.

## Why plain TensorRT is slow on the usual ONNX, and what this chain does instead

The ONNX is NOT taken from the jax2onnx export (about 10K nodes, most of them shape plumbing for the attack graph).
`bt6_export_onnx.py` rebuilds the served forward pass in PyTorch from the carrier's tensors and exports that: 2.7K
nodes, the 34 relation channels computed with constant one-hot tables and MatMul / Less / And / Or only (no scatter,
no data-dependent indexing), static batch, FP16 with FP32 relations / prologue / LayerNorm / site rms. Plain
TensorRT 11.3 runs that file at 3.6K positions/s at batch 128 on an RTX 5090; the plugins then double it.

## Chain

A carrier is made from the net's ONNX export with the lc0ex builder (`tools/make_carrier.py EXPORT.pb.gz
CARRIER.pb.gz`); `BT6_CARRIER=<path>` selects it for the two export scripts. A carrier without the error head is fine.

| Step | Script | Output |
|---|---|---|
| plain ONNX from the carrier (static batch) | `bt6_export_onnx.py export`, `make_fp16.sh` | `bt6_fp16_b<B>.onnx` (gate-fixed) |
| FP16 plugin ONNX (dynamic batch): BT6PlanesPrologue, 24 x BT6EgtAttention, 5 x BT6TripletSite | `bt6_k1_onnx.py --dynamic --triplet-plugin` (tables: `triplet_site_tables.py`) | `k1_dyn_tri.onnx` |
| carrier tensors as .npz for the INT8 scripts | `cache_carrier.py` | `carrier_all.npz` |
| INT8, accurate variant "k2": the carrier's static SmoothQuant on the LayerNorm-fed GEMMs, per-token INT8 on attention out (BT6RowQuant) and FFN down (BT6SigGluQuant) | `k2_int8_onnx.py --lc-sites attn_in,ffn_in --glu-plugin --rq-plugin` | `k2_hyb_glurq.onnx` |
| INT8, fast variant "k4" (lc0ex's own static Q1 recipe on all four sites; the attention writes INT8 codes; FFN gate/up GEMM with GLU epilogue = BT6I8GemmGlu) | `k2_int8_onnx.py --lc-sites attn_in,ffn_in,attn_out,ffn_mid --attn-i8 [--d1]`, then `k4_onnx.py` | `k4.onnx` |
| Ceres I/O: `/input/planes` FP16, `/output/policy`, `/output/wdl` (softmax), `/output/mlh` FP16 | `to_lc0_io.py in.onnx out.onnx` | engine-ready ONNX |
| gates vs PyTorch FP32 on 256 real positions | `trt_gate.py`, `metrics.py`, `npz_gate.py`, `lc0ex_ref_gate.py`, `run_lc0ex_int8.py` (the lc0ex INT8 program on the same positions), `emu.py` (INT8 emulator) | W-L, top-1, KL |
| build + gate / time | `k1_build_gate.sh`, `k1_time.sh`, `k2_build_gate.sh`, `k2_time.sh`, `time_trt.sh`; backward compatibility with our nets: `gate_n400M_k1.sh` | |

"k5" and "k6" are not other ONNX files: they are the k4 ONNX with a newer plugin library. k5 = faster attention kernel
(bit-identical outputs); k6 = the current library, k5 plus the triplet edge site's third kernel on the tensor cores
(trtexec 5.87 / 11.37 ms at batch 64 / 128 against 6.01 / 11.65 for k5; Ceres 10.9K against 10.6K evaluations per
second; gate and puzzles within noise of k5: rg2700 2810 / 3312 / 2933).

### INT8 without the carrier's vectors (`int8_vectors/`)

The static INT8 recipe needs one prescale vector per quantised site ("Q1": `r`, the format of lc0ex
`lab/_quant.py`). The 5.4M carrier has them; a carrier made from a plain export (e.g. BT6 7M) does not. `an_calib.py`
collects per-channel statistics at the four site inputs on calibration positions, `analyser.py make a=1 OUT.npz
CAL...` writes the vectors with the rule `r_j = 127 / amax_j` (the rule that fits the 5.4M carrier's vectors best;
they cannot be reproduced exactly without the original calibration positions), and `analyser.py emu` emulates the
INT8 net from a vector set. The vectors are clip-sensitive: calibrate on several thousand positions that are not the
gate positions. The same vectors build an lc0ex INT8 program (`make_carrier.py` with `LC0EX_QUANT_PRESCALE`,
`build_lab_artifact_q1.sh`).

### lc0 `onnx-trt` (`lc0_onnx_trt/`)

The lc0 change is also available as a branch: https://github.com/lepned/lc0/tree/bt6-trt-plugins (lc0 master + the
patch below, with a guide in `BT6_TRT_PLUGINS.md`); its release `bt6-trt-plugins-v1` has a Windows plugin library and
the BT6 7M INT8 net.

lc0 master runs the same ONNX through ONNX Runtime's TensorRT provider after three things:
`lc0_trt_plugins.patch` (backend option `trt_plugins='<lib>'` -> `trt_extra_plugin_lib_paths`, and placeholder custom
ops for the `trt.plugins` nodes so ONNX Runtime accepts them), the ONNX Runtime variant of the plugin library
(`BT6_ORT=1`, built against the TensorRT major version ONNX Runtime uses), and an ONNX with typed plugin outputs
(`add_plugin_types.py in.onnx out.onnx <plugin lib>`) and, for INT8, opset >= 19 (FP16 scales in DequantizeLinear).
Wrap with `lc0 onnx2leela --input-format=INPUT_CLASSICAL_112_PLANE --policy-format=POLICY_ATTENTION
--value-format=VALUE_WDL --moves-left-format=MOVES_LEFT_V1 --onnx-input=/input/planes ...` and run with
`--backend=onnx-trt --backend-opts="trt_plugins='<lib>',optimize=6"`.

The patch also adds the optional backend option `cuda_graph=N` (default 0 = off): on the N-th use of a batch size lc0
captures `Session::Run` as a CUDA graph and replays it afterwards (bit-identical outputs). TensorRT 10 spends about
0.8 ms of CPU time per batch enqueueing this net's 214 kernels, which the replay removes. With `batch=128` (TensorRT
profile optimised for 128 instead of 256, for minibatch size 128) the search goes from 10.5K to 10.9K evaluations per
second on the 5.4M INT8 net. Run in analysis and on 4000 puzzles, not in tournament games; not supported with
`steps` > 1.

## Results (RTX 5090, TensorRT 11.3 unless noted, 2026-11-25/26)

Gate on 256 real positions vs PyTorch FP32, BT6 5.4M (W-L mean, top-1, KL mean):

| | W-L | top-1 | KL |
|---|---|---|---|
| plain TensorRT FP16 | 0.00106 | 254 | 3.7e-5 |
| plugins FP16 (k1) | 0.00075 | 255 | 2.8e-5 |
| lc0ex FP16 program | 0.00082 | 255 | 1.8e-5 |
| plugins INT8 k2 (per-token) | 0.0093 | 251 | 1.6e-3 |
| plugins INT8 k4 / k5 (static Q1) | 0.0109 | 251 | 2.7e-3 |
| lc0ex INT8 program | 0.0112 | 248 | 2.8e-3 |

trtexec median ms (CUDA graph, real positions):

| batch | plain FP16 | plugins FP16 | INT8 k2 | INT8 k5 | lc0ex INT8 |
|---|---|---|---|---|---|
| 16 | 5.05 | 2.9 | 2.34 | 2.09-2.12 | 2.00 |
| 64 | 17.1 | 9.3-9.6 | 6.3 | 5.93-6.02 | 5.99 |
| 128 | 35.3 | 18.0-18.5 | 12.8 | 11.55-11.65 | 11.80 |
| 256 | 75.6 | 37.1-38.0 | 25.9 | 23.4-23.6 | - |

Search speed in Ceres (EngineBattle analyze, start position, 30 s, evaluations per second), 5.4M INT8: plugins k5
10.7K vs lc0ex 10.6K; lc0 master `onnx-trt` + plugins (ONNX Runtime 1.24, TensorRT 10.15) 10.4K.

Puzzles in Ceres, 5.4M, rg2700 without tablebases (policy / top-3 / value): plugins FP16 2810 / 3328 / 2938, lc0ex FP16
2809 / 3328 / 2940, plugins INT8 k2 2809 / 3328 / 2939, plugins INT8 k5 2809 / 3315 / 2929, lc0ex INT8 2806 / 3313 / 2934.

BT6 7M (carrier from the plain export, own Q1 vectors), Ceres, puzzles rg2700 (policy / top-3 / value): plugins FP16
2817 / 3318 / 2949, lc0ex FP16 2815 / 3318 / 2949, plugins INT8 k5 2813 / 3317 / 2947, lc0ex INT8 (own program)
2814 / 3312 / 2947.

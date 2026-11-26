# BT6 TensorRT plugins

TensorRT IPluginV3 plugins that make BT6-style nets fast in TensorRT: nets with BT6 term layers and the move-token
decoder (our 768x16 flat BT6 HL-Gauss net), and lc0 BT6 nets (Kovax's 1024x24x32 BT6 flagship). One library holds
all of them; every older plugin name / version is kept, so plugin ONNX files from earlier rounds still build.

| Plugin | Version | What it replaces |
|---|---|---|
| `BT6TermsAttention` | 1 | QK-norm, q.k, template mix and A@V of one term layer |
| `BT6TermsAttention` | 2 | as v1, plus fields `export_h` (raw logits H as a second output) and `fused_coef` |
| `BT6TermsAttention` | 3 | as v2, plus `out_int8` / `out_scale` (INT8 output straight into the next GEMM) |
| `BT6Prologue` | 1 | the whole prologue: squares -> 34 relation channels E, S, e0 |
| `BT6EdgeSite` | 1 | one edge site: readback, RMSNorm, edge FFN with reverse edge, RMSNorm |
| `BT6GluQuant` | 1 | SiLU(gate + bias) * up and per-token INT8 quantization of the FFN down input |
| `BT6DecAttn` | 1 | one attention of a move-token decoder block (self / cross / post-move) |
| `BT6DecHead` | 1 | decoder read-out: out_ln, policy with per-move bias, mean / max pooling |
| `BT6EgtAttention` | 1 | lc0 BT6 attention (32 heads, [Q\|K\|V\|G] packing, 2·sigmoid output gate, per-head qk_scale, door-scaled H for the sites, optional INT8 output) |
| `BT6PlanesPrologue` | 1 | the prologue of an lc0 BT6 net read from the 112 lc0 planes |
| `BT6TripletSite` | 1 | lc0 BT6 edge site with the triplet branch (contraction path / ag) |
| `BT6SigGluQuant` | 1 | sigmoid GLU (sigmoid(gate) · (up + b)) and per-token INT8 of the FFN down input |
| `BT6RowQuant` | 1 | per-token INT8 quantisation of a tensor (attention output before the out GEMM) |
| `BT6LnQuant` | 1 | residual add + LayerNorm + static INT8 quantise (TensorRT's own fusion is as fast; not used by default) |
| `BT6I8GemmGlu` | 1 | INT8 GEMM (CUTLASS) with the sigmoid-GLU epilogue writing INT8 codes (lc0ex's Q1 recipe) |

## Quick start: an lc0 BT6 net in Ceres

1. **Ceres with plugin support** = this fork. Build the small native TensorRT wrapper from
   `../../Native` (`TensorRTWrapper.cpp`; see `build_instructions_windows.txt` / `build_instructions_linux.txt`,
   `make NVINFER_MAJOR=10|11` on Linux) against your TensorRT, and put it next to `Ceres.dll` in place of the
   binary shipped in the repo (the shipped binaries predate the plugin call `TRT_LoadPluginLibrary`).
2. **The plugin library**, built on your machine against the SAME TensorRT major version (see Build below).
3. **A plugin ONNX of the net** (see `tools/kovax_bt6/README.md` for how they are made from the lc0ex carrier):
   FP16 or INT8, input `/input/planes`, outputs `/output/policy`, `/output/wdl`, `/output/mlh`.
4. Net and device spec:

   ```
   Network: ONNX_TRT:<path to the plugin .onnx>|PLUGIN=<path to bt6_trt_plugin.dll or libbt6_trt_plugin.so>
   Device:  GPU:0#TensorRTNative
   ```

   The first start builds the TensorRT engines (one per batch size; a few minutes for a 1024x24 net), later starts
   load them from the cache. The environment variable `CERES_TRT_PLUGINS` (path list) works instead of `|PLUGIN=`.

lc0 (`onnx-trt`): see `tools/kovax_bt6/lc0_onnx_trt/` (a small lc0 patch, the ONNX Runtime variant of the library and
two changes to the ONNX file).

## Build

The library must be built against the same TensorRT major version as the runtime (TensorRT 10 or newer). Engine
plans store only the plugin name and fields, so the library must be loaded before every build and every
deserialization. `BT6I8GemmGlu` needs the CUTLASS 4.x headers (https://github.com/NVIDIA/cutlass, tested with 4.8.1):
point `CUTLASS_INCLUDE` at its `include` directory.

- Windows: `set CUTLASS_INCLUDE=<cutlass>\include` then `build_win.bat [TensorRT root] [CUDA root]`
  -> `out\bt6_trt_plugin.dll`
- Linux: `NVCC=... TRT_INCLUDE=... CUTLASS_INCLUDE=<cutlass>/include ./build_linux.sh ["-gencode ..."]`
  -> `out/libbt6_trt_plugin.so`
- `BT6_ORT=1` builds the ONNX Runtime variant (`bt6_trt_plugin_ort`), which registers its plugins when the library
  is loaded (ONNX Runtime's TensorRT provider loads plugin libraries as plain shared libraries).

Default architectures: sm_80, sm_86, sm_89, sm_90, sm_120. B200 / GB300 / GB10 need sm_100 / sm_103 / sm_121 added
(env `BT6_ARCHS` on Windows, the first argument on Linux).

What has actually been run (RTX 5090, sm_120): Windows with TensorRT 11.3 (Ceres, trtexec) and with TensorRT
10.15 through ONNX Runtime (lc0 `onnx-trt`); Linux x86_64 (WSL, CUDA 13.0) with TensorRT 10.16, where the FP16 and
INT8 nets give the same gate numbers as on Windows (`tools/kovax_bt6/linux_gate.py`). The code for the other
architectures (sm_80 .. sm_90, sm_103, sm_121) compiles but has not been executed, and aarch64 has not been built.

Environment switches of the lc0 BT6 attention kernel: `BT6_EGT_K4=1` selects the previous kernel path and schedule
(bit-identical outputs), `BT6_EGT_SCHED=0` only the previous schedule.

`BT6TripletSite` runs its third kernel (output map and edge FFN) on the tensor cores: FP16 operands, FP32 accumulation,
weights split in two FP16 halves. About 2.4 % faster end to end on the 1024x24 net, outputs equal within the INT8 gate's
noise but not bit-identical; `BT6_SITE_K5=1` selects the previous FP32 kernels (bit-identical to the previous library).

## Our own nets: making the plugin ONNX (CeresTrain, `src/CeresTrainPy`)

One command per checkpoint:

```
python make_plugin_onnx.py <served m80.onnx> <net.ts | ckpt> <out_dir> --fp16 | --int8 [--qdq F] [--gate] [--keep]
```

The steps are functions in `bt6_trt_plugin_export.py` (one module; its command line runs a single step).
FP16 chain: `dump_params` -> `terms_attention` -> `prologue_sites` -> `merge_gemms` (coef, ffn) -> `decoder`.
INT8 chain: the plain INT8 QDQ ONNX (`qdq_export.py` percentile 99.9999 + shape allowlist, `qdq_to_fp16.py` with
fp32 islands; `--qdq` reuses an existing one), then `dump_params` -> `terms_attention` -> `prologue_sites` ->
`port_qdq` (exclude 1536x768) -> `merge_gemms` (coef) -> `glu_quant` -> `attn_int8_out` -> `decoder`.

`tools/gate.py` compares the plugin engine against PyTorch on real positions (W-L mean, top-1, policy KL).
Limits: FP16 W-L mean <= 0.0003 and KL <= 3e-6; INT8 W-L mean <= 0.0025, top-1 >= 253/256, KL <= 3e-4.
Small decoder errors are only caught by the FP16 limits. Always time with real positions (`tools/time_real.sh`):
random input sends the attention kernel down a slow fallback path.

768x16 at 200M, trtexec TensorRT 11.3, RTX 5090, batch 64 / 128 in ms:

| | no plugin | FP16 | INT8 |
|---|---|---|---|
| b64 | 10.06 | 5.14 | 4.00 |
| b128 | 21.29 | 9.58 | 7.22 |

Puzzles equal to the plain net within 2 (200M and 400M).

## lc0 BT6 nets (Kovax's BT6 flagship)

ONNX chain, INT8 recipes, gates and results: `tools/kovax_bt6/README.md`. Summary on one RTX 5090 (BT6 5.4M,
Ceres search speed in evaluations per second from the start position, only the backend differs):

| | TensorRT + plugins | lc0ex |
|---|---|---|
| INT8 | 10.7K | 10.6K |

Puzzles are equal between the two backends within noise.

## Origin of the kernels

Several kernels are derived from lc0ex by Kovax (https://github.com/Kovax007/lczero-triton): the term-layer and lc0
BT6 attention kernels (`bt6_attn_kernels.cu`, `bt6_egt_attn.cu`, from the lab's `terms_v9` / `egt_v9` head
pipelines), the triplet edge site (`bt6_triplet_site.cu`, from `triplet_mma`), and the sigmoid-GLU epilogue of the
INT8 GEMM (`bt6_i8gemm_kernels.cu`). The INT8 recipe "Q1" is lc0ex's. Each file's header says what was changed.
`BT6I8GemmGlu` builds on CUTLASS (NVIDIA, BSD 3-Clause).

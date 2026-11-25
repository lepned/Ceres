# BT6 TensorRT plugins (round 4)

TensorRT IPluginV3 plugins for nets with BT6 term layers and the move-token decoder (for example the 768x16 flat
BT6 HL-Gauss net). One library holds all of them; every older name / version is kept, so plugin ONNX files from
earlier rounds still build.

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

## Build

The library must be built against the same TensorRT major version as the runtime (TensorRT 10 or newer).
Engine plans store only the plugin name and fields, so the library must be loaded before every build and every
deserialization.

- Windows: `build_win.bat [TensorRT root] [CUDA root]` -> `out\bt6_trt_plugin.dll`
- Linux: `NVCC=... TRT_INCLUDE=... ./build_linux.sh` -> `out/libbt6_trt_plugin.so`

Default architectures: sm_80, sm_86, sm_89, sm_90, sm_120 plus compute_120 PTX. B200 / B300 (GB300) need
sm_100 / sm_103 added (env `BT6_ARCHS` on Windows, the first argument on Linux).

## Use

- Ceres: add `|PLUGIN=<path>[,<path>]` to the net spec, or set the environment variable `CERES_TRT_PLUGINS`.
- Mycel: UCI option `TrtPlugins` (path list).

The ONNX file must be the plugin ONNX (see below); a plain ONNX file runs without plugins.

## Making the plugin ONNX (CeresTrain, `src/CeresTrainPy`)

One command per checkpoint:

```
python make_plugin_onnx.py <served m80.onnx> <net.ts | ckpt> <out_dir> --fp16 | --int8 [--qdq F] [--gate] [--keep]
```

The steps are functions in `bt6_trt_plugin_export.py` (one module; its command line runs a single step).
FP16 chain: `dump_params` -> `terms_attention` -> `prologue_sites` -> `merge_gemms` (coef, ffn) -> `decoder`.
INT8 chain: the plain INT8 QDQ ONNX (`qdq_export.py` percentile 99.9999 + shape allowlist, `qdq_to_fp16.py` with
fp32 islands; `--qdq` reuses an existing one), then `dump_params` -> `terms_attention` -> `prologue_sites` ->
`port_qdq` (exclude 1536x768) -> `merge_gemms` (coef) -> `glu_quant` -> `attn_int8_out` -> `decoder`.

## Check

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

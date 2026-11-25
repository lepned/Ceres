#!/bin/bash
# gate_n400M_k1.sh <onnx> [dll] (copy of trt_bt6_plugin/work/n400M/gate_n400M.sh, default DLL build_k1) : tools/gate3.sh sets with the 400M net's own references
#   [64] / [x4]: tpl_arms ref_..._400011264.npz (PyTorch FP32, .ts, squares_real_b64.bin)
#   [256]: ref256_h768_400M.npz (PyTorch FP32 from the 400M .ts on eval_set13 0..255, same positions as r3/ref_h768_200M_lc0ex.npz)
ONNX=$1; DLL=${2:-C:/Dev/Chess/trt_bt6_plugin/build_k1/bt6_trt_plugin_k1.dll}
export PATH="/c/Dev/TensorRT-11.3.0.99/bin:$PATH"
PY=/c/Dev/Chess/lc0ex_work/terms/trt113_venv/Scripts/python.exe
W=/c/Dev/Chess/trt_bt6_plugin/work
G=/c/Dev/Chess/trt_bt6_plugin/tools/gate_r3.py
REF64=/c/Dev/Chess/lc0ex_work/terms/tpl_arms/ref_fry_hlg_tpl8qkbt6Ant135notripei_t80t91_4B_400011264.npz
REF256=$W/n400M/ref256_h768_400M.npz
echo "[64]   $($PY $G $ONNX "$DLL" $W/squares_real_b64.bin 64 $REF64 2>&1 | tail -1)"
echo "[x4]   $($PY $G $ONNX "$DLL" $W/squares_real_b64.bin 64 $REF64 --tile 4 2>&1 | tail -1)"
echo "[256]  $($PY $G $ONNX "$DLL" $REF256 256 $REF256 --batch 64 2>&1 | tail -1)"

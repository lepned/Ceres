#!/bin/bash
# Build out/libbt6_trt_plugin.so: TensorRT plugins for our nets (BT6TermsAttention v1-v3, BT6Prologue, BT6EdgeSite,
# BT6GluQuant, BT6DecAttn, BT6DecHead) and for lc0 BT6 nets (BT6EgtAttention, BT6PlanesPrologue, BT6TripletSite,
# BT6SigGluQuant, BT6RowQuant, BT6LnQuant, BT6I8GemmGlu).
#   NVCC=<nvcc> TRT_INCLUDE=<TensorRT include dir> CUTLASS_INCLUDE=<cutlass/include> ./build_linux.sh ["-gencode ..."]
# The TensorRT headers must be the same major version as the TensorRT the engine runs on (10 or newer).
#   CUTLASS_INCLUDE  the include directory of CUTLASS 4.x (https://github.com/NVIDIA/cutlass, tested with 4.8.1); needed by
#                    BT6I8GemmGlu. Default: third_party/cutlass/include next to this script.
#   $1               the -gencode list (default sm_80/86/89/90/120; add sm_103 / sm_121 for GB300 / GB10, e.g.
#                    "-gencode arch=compute_103,code=sm_103").
#   BT6_ORT=1        ONNX Runtime variant out/libbt6_trt_plugin_ort.so: registers its plugins when the library is loaded
#                    (ONNX Runtime's TensorRT provider, e.g. lc0 onnx-trt, loads plugin libraries as plain shared
#                    objects). Needs TRT_LIB=<directory of libnvinfer.so>.
set -e
cd "$(dirname "$0")"
NVCC=${NVCC:-nvcc}
TRT_INCLUDE=${TRT_INCLUDE:-/usr/include/x86_64-linux-gnu}
CUTLASS_INCLUDE=${CUTLASS_INCLUDE:-$(pwd)/third_party/cutlass/include}
[ -f "$CUTLASS_INCLUDE/cutlass/cutlass.h" ] || { echo "CUTLASS headers not found in $CUTLASS_INCLUDE: set CUTLASS_INCLUDE"; exit 1; }
ARCHS=${1:-"-gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_120,code=sm_120"}
NAME=libbt6_trt_plugin
EXTRA=""
if [ "${BT6_ORT:-0}" = 1 ]; then
  NAME=libbt6_trt_plugin_ort
  EXTRA="src_ort/bt6_autoreg.cpp -L${TRT_LIB:?TRT_LIB (directory of libnvinfer.so) is needed with BT6_ORT=1} -lnvinfer"
fi
mkdir -p out
$NVCC -O3 -std=c++17 -shared -Xcompiler -fPIC,-fvisibility=hidden -cudart static --expt-relaxed-constexpr $ARCHS \
  -I"$TRT_INCLUDE" -Isrc -I"$CUTLASS_INCLUDE" \
  src/bt6_attn_kernels.cu src/bt6_prologue_kernels.cu src/bt6_site_kernels.cu src/bt6_glu_kernels.cu \
  src/bt6_rowquant_kernels.cu src/bt6_lnquant_kernels.cu src/bt6_i8gemm_kernels.cu src/bt6_dec_kernels.cu \
  src/bt6_egt_attn.cu src/bt6_triplet_site.cu src/bt6_plugin.cpp src/bt6_triplet_site_plugin.cpp $EXTRA \
  -o out/$NAME.so > out/build_$NAME.log 2>&1 || { cat out/build_$NAME.log; exit 1; }
echo "built $(pwd)/out/$NAME.so against $TRT_INCLUDE"

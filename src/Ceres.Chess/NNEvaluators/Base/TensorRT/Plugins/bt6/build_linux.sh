#!/bin/bash
# Build out/libbt6_trt_plugin.so: TensorRT plugins for our nets (BT6TermsAttention v1-v3, BT6Prologue, BT6EdgeSite,
# BT6GluQuant, BT6DecAttn, BT6DecHead) and for lc0 BT6 nets (BT6EgtAttention, BT6PlanesPrologue, BT6TripletSite,
# BT6SigGluQuant, BT6RowQuant, BT6LnQuant, BT6I8GemmGlu).
#   NVCC=<nvcc> TRT_INCLUDE=<TensorRT include dir> CUTLASS_INCLUDE=<cutlass/include> ./build_linux.sh ["-gencode ..."]
# The TensorRT headers must be the same major version as the TensorRT the engine runs on (10 or newer).
#   CUTLASS_INCLUDE  the include directory of CUTLASS 4.x (https://github.com/NVIDIA/cutlass, tested with 4.8.1); needed by
#                    BT6I8GemmGlu. Default: third_party/cutlass/include next to this script.
#   $1               the -gencode list (default sm_80/86/89/90/120 plus PTX for compute_80, which the driver compiles for
#                    any other architecture >= sm_80; add sm_103 / sm_121 for GB300 / GB10, e.g.
#                    "-gencode arch=compute_103,code=sm_103").
#   BT6_T5           GB300 attention (bt6_egt_t5.cu, tcgen05, compiled for sm_103a only and chosen at run time on compute
#                    capability 10.3; every other GPU keeps v9): auto (default) = when the -gencode list has compute_103,
#                    1 = always, 0 = never. Needs nvcc 12.9 or newer. Env BT6_EGT_T5=0 at run time keeps v9 on the GB300 too.
#   BT6_ORT=1        ONNX Runtime variant out/libbt6_trt_plugin_ort.so: registers its plugins when the library is loaded
#                    (ONNX Runtime's TensorRT provider, e.g. lc0 onnx-trt, loads plugin libraries as plain shared
#                    objects). Needs TRT_LIB=<directory of libnvinfer.so>.
set -e
cd "$(dirname "$0")"
NVCC=${NVCC:-nvcc}
TRT_INCLUDE=${TRT_INCLUDE:-/usr/include/x86_64-linux-gnu}
CUTLASS_INCLUDE=${CUTLASS_INCLUDE:-$(pwd)/third_party/cutlass/include}
[ -f "$CUTLASS_INCLUDE/cutlass/cutlass.h" ] || { echo "CUTLASS headers not found in $CUTLASS_INCLUDE: set CUTLASS_INCLUDE"; exit 1; }
ARCHS=${1:-"-gencode arch=compute_80,code=[sm_80,compute_80] -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_120,code=sm_120"}
NAME=libbt6_trt_plugin
EXTRA=""
if [ "${BT6_ORT:-0}" = 1 ]; then
  NAME=libbt6_trt_plugin_ort
  EXTRA="src_ort/bt6_autoreg.cpp -L${TRT_LIB:?TRT_LIB (directory of libnvinfer.so) is needed with BT6_ORT=1} -lnvinfer"
fi
mkdir -p out
T5_OBJ=""; T5_DEF=""
if [ "${BT6_T5:-auto}" = 1 ] || { [ "${BT6_T5:-auto}" = auto ] && echo "$ARCHS" | grep -q compute_103; }; then
  $NVCC -O3 -std=c++17 -c -Xcompiler -fPIC,-fvisibility=hidden --expt-relaxed-constexpr -DBT6_WITH_T5 \
    -gencode arch=compute_103a,code=sm_103a -I"$TRT_INCLUDE" -Isrc src/bt6_egt_t5.cu -o out/bt6_egt_t5.o \
    > out/build_t5.log 2>&1 || { cat out/build_t5.log; exit 1; }
  T5_OBJ=out/bt6_egt_t5.o; T5_DEF=-DBT6_WITH_T5
fi
$NVCC -O3 -std=c++17 -shared -Xcompiler -fPIC,-fvisibility=hidden -cudart static --expt-relaxed-constexpr $T5_DEF $ARCHS \
  -I"$TRT_INCLUDE" -Isrc -I"$CUTLASS_INCLUDE" \
  src/bt6_attn_kernels.cu src/bt6_prologue_kernels.cu src/bt6_site_kernels.cu src/bt6_glu_kernels.cu \
  src/bt6_rowquant_kernels.cu src/bt6_lnquant_kernels.cu src/bt6_i8gemm_kernels.cu src/bt6_dec_kernels.cu \
  src/bt6_egt_attn.cu src/bt6_triplet_site.cu src/bt6_plugin.cpp src/bt6_triplet_site_plugin.cpp $T5_OBJ $EXTRA \
  -o out/$NAME.so > out/build_$NAME.log 2>&1 || { cat out/build_$NAME.log; exit 1; }
echo "built $(pwd)/out/$NAME.so against $TRT_INCLUDE${T5_OBJ:+ (with the GB300 t5 attention)}"

#!/bin/bash
# Build out/libbt6_trt_plugin.so (round 4: BT6TermsAttention v1-v3, BT6Prologue, BT6EdgeSite, BT6GluQuant, BT6DecAttn, BT6DecHead).
#   NVCC=<nvcc> TRT_INCLUDE=<TensorRT include dir> ./build_linux.sh ["-gencode ..."]
# The TensorRT headers must be the same major version as the TensorRT the engine runs on (10 or newer).
set -e
cd "$(dirname "$0")"
NVCC=${NVCC:-nvcc}
TRT_INCLUDE=${TRT_INCLUDE:-/usr/include/x86_64-linux-gnu}
ARCHS=${1:-"-gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_120,code=sm_120 -gencode arch=compute_120,code=compute_120"}
mkdir -p out
$NVCC -O3 -std=c++17 -shared -Xcompiler -fPIC,-fvisibility=hidden -cudart static $ARCHS -I"$TRT_INCLUDE" -Isrc \
  src/bt6_attn_kernels.cu src/bt6_prologue_kernels.cu src/bt6_site_kernels.cu src/bt6_glu_kernels.cu src/bt6_dec_kernels.cu src/bt6_plugin.cpp \
  -o out/libbt6_trt_plugin.so > out/build.log 2>&1 || { cat out/build.log; exit 1; }
echo "built $(pwd)/out/libbt6_trt_plugin.so against $TRT_INCLUDE"

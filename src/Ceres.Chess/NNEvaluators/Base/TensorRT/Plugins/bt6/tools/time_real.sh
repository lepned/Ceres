#!/bin/bash
# time_real.sh <onnx> <tag> [plugin.dll] [batches] [extra trtexec args]
# Round 4 (copy of time_real_r3.sh, engines in work/r4/engines): trtexec 11.3 strongly typed, CUDA graph, no transfers, REAL positions (--loadInputs squares_real_b<B>.bin).
# Engines + logs in work/r4/engines/<tag>_b<B>.{plan,log}; prints the median GPU compute time per batch.
# If the plan exists it is re-timed (--loadEngine) instead of rebuilt (set REBUILD=1 to force a build).
ONNX=$1; TAG=$2; PLUG=$3; BATCHES=${4:-"16 64 128 256"}; EXTRA=$5
export PATH="/c/Dev/TensorRT-11.3.0.99/bin:$PATH"
W=/c/Dev/Chess/trt_bt6_plugin/work
mkdir -p $W/r4/engines
cd $W
for B in $BATCHES; do
  P=""; [ -n "$PLUG" ] && P="--dynamicPlugins=$PLUG"
  E=r4/engines/${TAG}_b$B.plan
  if [ -f "$E" ] && [ -z "$REBUILD" ]; then SRC="--loadEngine=$E"; else SRC="--onnx=$ONNX --stronglyTyped --shapes=squares:${B}x64x137 --saveEngine=$E $EXTRA"; fi
  trtexec $SRC --useCudaGraph --noDataTransfers --loadInputs=squares:squares_real_b$B.bin --warmUp=1000 --duration=6 $P \
     > r4/engines/${TAG}_b$B.log 2>&1
  MED=$(grep -E 'GPU Compute Time: min' r4/engines/${TAG}_b$B.log | sed 's/.*median = \([0-9.]*\) ms.*/\1/')
  echo "$TAG b$B median ${MED:-FAILED} ms"
done

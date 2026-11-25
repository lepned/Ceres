#!/bin/bash
# k1_time.sh <dll|-> <engine_tag:B> ...: median GPU compute (trtexec 11.3, CUDA graph, no transfers, real positions
# planes_b<B>.bin), under the lock file kovax_bt6_trt/gpu_timing.lock. Engine paths: plugin/engines/<tag>_b<B>.engine,
# or ../<tag>.engine for the plain engines (tag starting with bt6_fp16).
DLL=$1; shift
export PATH="/c/Dev/TensorRT-11.3.0.99/bin:/c/Dev/TensorRT-11.3.0.99/lib:$PATH"
K=/c/Dev/Chess/kovax_bt6_trt
LOCK=$K/gpu_timing.lock
cd $K/int8
while [ -f $LOCK ] && [ $(( $(date +%s) - $(stat -c %Y $LOCK) )) -lt 1800 ]; do echo "waiting for lock: $(cat $LOCK)"; sleep 20; done
echo "agent k2 (INT8) $(date '+%F %T')" > $LOCK
trap 'rm -f $LOCK' EXIT
for spec in "$@"; do
  tag=${spec%%:*}; B=${spec##*:}
  for try in 1 2 3 4 5 6; do
    busy=$(tasklist | grep -i -E "lc0.exe|mycel|ceres" | grep -v -i CrossDevice)
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    [ -z "$busy" ] && [ "$util" -lt 10 ] && break
    echo "GPU busy ($util %) $busy, waiting"; sleep 30
  done
  if [[ $tag == bt6_fp16* ]]; then E=../$tag.engine; P=""; else if [[ $tag == k1* ]]; then E=../plugin/engines/${tag}_b$B.engine; else E=engines/${tag}_b$B.engine; fi; P="--dynamicPlugins=$DLL"; fi
  trtexec --loadEngine=$E $P --loadInputs=planes:../planes_b$B.bin --useCudaGraph --noDataTransfers --warmUp=1000 --duration=6 > engines/time_${tag}_b$B.log 2>&1
  med=$(grep "] GPU Compute Time" engines/time_${tag}_b$B.log | sed -E 's/.*median = ([0-9.]+) ms.*/\1/')
  echo "$tag B=$B median ${med:-FAILED} ms  $(awk -v b=$B -v m=${med:-0} 'BEGIN{if (m > 0) printf "%.0f pos/s", b*1000/m}')" | tee -a k2_timing.log
done

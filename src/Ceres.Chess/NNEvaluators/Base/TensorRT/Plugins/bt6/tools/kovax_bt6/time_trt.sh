# median GPU compute time per engine, real positions; checks the GPU is free before each run
export PATH="/c/Dev/TensorRT-11.3.0.99/bin:/c/Dev/TensorRT-11.3.0.99/lib:$PATH"
cd /c/Dev/Chess/kovax_bt6_trt
for spec in "$@"; do
  eng=${spec%%:*}; B=${spec##*:}
  busy=$(tasklist | grep -i -E "trtexec|lc0.exe|mycel|ceres" | grep -v -i CrossDevice)
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
  if [ -n "$busy" ] || [ "$util" -ge 10 ]; then echo "GPU BUSY ($util %): $busy"; exit 1; fi
  trtexec --loadEngine=$eng.engine --loadInputs=planes:planes_b$B.bin --useCudaGraph --noDataTransfers --warmUp=1000 --duration=6 > time_$eng.log 2>&1
  med=$(grep "] GPU Compute Time" time_$eng.log | sed -E 's/.*median = ([0-9.]+) ms.*/\1/')
  echo "$eng B=$B median GPU compute $med ms  -> $(awk -v b=$B -v m=$med 'BEGIN{printf "%.0f", b*1000/m}') pos/s"
done

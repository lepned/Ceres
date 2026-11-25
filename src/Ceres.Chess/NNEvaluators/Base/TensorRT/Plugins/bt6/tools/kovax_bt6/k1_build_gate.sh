#!/bin/bash
# k1_build_gate.sh <onnx> <tag> <dll> [batches]: build static-shape engines (trtexec 11.3, strongly typed, plugin
# library loaded) as plugin/engines/<tag>_b<B>.engine and gate each against PyTorch FP32 (trt_gate.py, real positions).
ONNX=$1; TAG=$2; DLL=$3; BATCHES=${4:-256}
export PATH="/c/Dev/TensorRT-11.3.0.99/bin:/c/Dev/TensorRT-11.3.0.99/lib:$PATH"
K=/c/Dev/Chess/kovax_bt6_trt
mkdir -p $K/plugin/engines
cd $K/plugin
for B in $BATCHES; do
  busy=$(tasklist | grep -i -E "lc0.exe|mycel|ceres" | grep -v -i CrossDevice)
  if [ -n "$busy" ]; then echo "GPU BUSY: $busy"; exit 1; fi
  E=engines/${TAG}_b$B.engine
  if [ ! -f "$E" ] || [ -n "$REBUILD" ]; then
    trtexec --onnx=$ONNX --stronglyTyped --shapes=planes:${B}x112x8x8 --saveEngine=$E --dynamicPlugins=$DLL \
      --skipInference $EXTRA > engines/${TAG}_b$B.build.log 2>&1 || { echo "BUILD FAILED $TAG b$B"; tail -20 engines/${TAG}_b$B.build.log; exit 2; }
  fi
  trtexec --loadEngine=$E --dynamicPlugins=$DLL --loadInputs=planes:../planes_b$B.bin --exportOutput=engines/out_${TAG}_b$B.json \
    --iterations=1 --warmUp=0 --duration=0 > engines/${TAG}_b$B.run.log 2>&1 || { echo "RUN FAILED $TAG b$B"; tail -20 engines/${TAG}_b$B.run.log; exit 2; }
  MSYS_NO_PATHCONV=1 wsl -e bash -c "cd /mnt/c/Dev/Chess/kovax_bt6_trt && /home/lep/cerestrain-env/bin/python trt_gate.py plugin/engines/out_${TAG}_b$B.json ${TAG}_b$B" | tee engines/${TAG}_b$B.gate.txt
  [ -z "$KEEP" ] && rm -f engines/out_${TAG}_b$B.json
done

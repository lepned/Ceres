@echo off
rem Build out\bt6_trt_plugin.dll: TensorRT plugins for our nets (BT6TermsAttention v1-v3, BT6Prologue, BT6EdgeSite, BT6GluQuant,
rem BT6DecAttn, BT6DecHead) and for lc0 BT6 nets (BT6EgtAttention, BT6PlanesPrologue, BT6TripletSite, BT6SigGluQuant,
rem BT6RowQuant, BT6LnQuant, BT6I8GemmGlu).
rem   build_win.bat [TensorRT root] [CUDA root]
rem The TensorRT headers must be the same major version as the TensorRT the engine runs on (10 or 11).
rem env CUTLASS_INCLUDE  the include directory of CUTLASS 4.x (https://github.com/NVIDIA/cutlass, tested with 4.8.1); needed
rem                      by BT6I8GemmGlu. Default: third_party\cutlass\include next to this script.
rem env BT6_ARCHS        overrides the -gencode list. Default: sm_80/86/89/90/100/103/120/121 plus PTX for compute_80 (the driver
rem                      compiles that for any other architecture >= sm_80); sm_100 and newer need CUDA 12.8 / 12.9 or newer.
rem env BT6_ORT=1        ONNX Runtime variant out\bt6_trt_plugin_ort.dll: registers its plugins when the DLL is loaded
rem                      (ONNX Runtime's TensorRT provider, e.g. lc0 onnx-trt, loads plugin libraries as plain DLLs).
setlocal
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (echo vcvars64 failed & exit /b 1)
set "TRT=%~1"
if "%TRT%"=="" set "TRT=C:\Dev\TensorRT-11.3.0.99"
set "CUDA=%~2"
if "%CUDA%"=="" set "CUDA=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
cd /d %~dp0
if "%CUTLASS_INCLUDE%"=="" set "CUTLASS_INCLUDE=%~dp0third_party\cutlass\include"
if not exist "%CUTLASS_INCLUDE%\cutlass\cutlass.h" (echo CUTLASS headers not found in %CUTLASS_INCLUDE%: set CUTLASS_INCLUDE & exit /b 1)
set ARCHS=%BT6_ARCHS%
if "%ARCHS%"=="" set ARCHS=-gencode arch=compute_80,code=[sm_80,compute_80] -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_100,code=sm_100 -gencode arch=compute_103,code=sm_103 -gencode arch=compute_120,code=sm_120 -gencode arch=compute_121,code=sm_121
set NAME=bt6_trt_plugin
set EXTRA=
if not "%BT6_ORT%"=="1" goto build
set NAME=bt6_trt_plugin_ort
for %%f in ("%TRT%\lib\nvinfer_1?.lib") do set EXTRA=src_ort\bt6_autoreg.cpp "%%f"
if "%EXTRA%"=="" (echo nvinfer import library not found in %TRT%\lib & exit /b 1)
:build
if not exist out mkdir out
"%CUDA%\bin\nvcc.exe" -O3 -std=c++17 -shared -cudart static --expt-relaxed-constexpr %ARCHS% -Xcompiler "/MD /O2 /EHsc /Zc:__cplusplus" -I"%TRT%\include" -Isrc -I"%CUTLASS_INCLUDE%" src\bt6_attn_kernels.cu src\bt6_prologue_kernels.cu src\bt6_site_kernels.cu src\bt6_glu_kernels.cu src\bt6_rowquant_kernels.cu src\bt6_lnquant_kernels.cu src\bt6_i8gemm_kernels.cu src\bt6_dec_kernels.cu src\bt6_egt_attn.cu src\bt6_triplet_site.cu src\bt6_plugin.cpp src\bt6_triplet_site_plugin.cpp %EXTRA% -o out\%NAME%.dll > out\build_%NAME%.log 2>&1
if errorlevel 1 (type out\build_%NAME%.log & exit /b 1)
echo built %~dp0out\%NAME%.dll against %TRT%

@echo off
rem Build out\bt6_trt_plugin.dll (round 4: BT6TermsAttention v1-v3, BT6Prologue, BT6EdgeSite, BT6GluQuant, BT6DecAttn, BT6DecHead).
rem   build_win.bat [TensorRT root] [CUDA root]
rem The TensorRT headers must be the same major version as the TensorRT the engine runs on (10 or 11).
rem env BT6_ARCHS overrides the -gencode list (default sm_80/86/89/90/120 + compute_120 PTX).
setlocal
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (echo vcvars64 failed & exit /b 1)
set "TRT=%~1"
if "%TRT%"=="" set "TRT=C:\Dev\TensorRT-11.3.0.99"
set "CUDA=%~2"
if "%CUDA%"=="" set "CUDA=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
set ARCHS=%BT6_ARCHS%
if "%ARCHS%"=="" set ARCHS=-gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_120,code=sm_120 -gencode arch=compute_120,code=compute_120
cd /d %~dp0
if not exist out mkdir out
"%CUDA%\bin\nvcc.exe" -O3 -std=c++17 -shared -cudart static %ARCHS% -Xcompiler "/MD /O2 /EHsc" -I"%TRT%\include" -Isrc src\bt6_attn_kernels.cu src\bt6_prologue_kernels.cu src\bt6_site_kernels.cu src\bt6_glu_kernels.cu src\bt6_dec_kernels.cu src\bt6_plugin.cpp -o out\bt6_trt_plugin.dll > out\build.log 2>&1
if errorlevel 1 (type out\build.log & exit /b 1)
echo built %~dp0out\bt6_trt_plugin.dll against %TRT%

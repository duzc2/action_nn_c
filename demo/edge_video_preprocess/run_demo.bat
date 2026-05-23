@echo off
setlocal

set "ACTION_C_ROOT=%~dp0..\.."
for %%I in ("%ACTION_C_ROOT%") do set "ACTION_C_ROOT=%%~fI"
set "BUILD_ROOT=%ACTION_C_ROOT%\build\demo\edge_video_preprocess"
set "GENERATED_DIR=%BUILD_ROOT%\data"

:: Try to auto-discover Visual Studio via vswhere; fall back to default path.
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" goto :use_default_vc
for /f "usebackq delims=" %%I in (`"%VSWHERE%" -latest -property installationPath 2^>nul`) do (
    set "VS_PATH=%%I"
    goto :found_vs
)
:use_default_vc
set "VS_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community"
:found_vs
set "VCVARS=%VS_PATH%\VC\Auxiliary\Build\vcvars64.bat"

echo ============================================================
echo Edge Video Preprocessing Demo - Full Pipeline
echo ============================================================
echo.

echo [edge_video_preprocess] step 1/7 run data preparation
echo ------------------------------------------------------------
python "%~dp0data_prep.py" %* || goto :fail
echo.

echo [edge_video_preprocess] step 2/7 configure + build generate
echo ------------------------------------------------------------
cmd /c "call \"%VCVARS%\" && cmake -S \"%ACTION_C_ROOT%\demo\edge_video_preprocess\generate\" -B \"%BUILD_ROOT%\generate\" -G Ninja -DCMAKE_C_COMPILER=clang && cmake --build \"%BUILD_ROOT%\generate\"" || goto :fail
echo.

echo [edge_video_preprocess] step 3/7 run generate
echo ------------------------------------------------------------
"%BUILD_ROOT%\generate\edge_video_preprocess_generate.exe" || goto :fail
echo.

echo [edge_video_preprocess] step 4/7 configure + build train
echo ------------------------------------------------------------
cmd /c "call \"%VCVARS%\" && cmake -S \"%ACTION_C_ROOT%\demo\edge_video_preprocess\train\" -B \"%BUILD_ROOT%\train\" -G Ninja -DCMAKE_C_COMPILER=clang -DACTION_C_GENERATED_DIR=\"%GENERATED_DIR%\" && cmake --build \"%BUILD_ROOT%\train\"" || goto :fail
echo.

echo [edge_video_preprocess] step 5/7 run train
echo ------------------------------------------------------------
"%BUILD_ROOT%\train\edge_video_preprocess_train.exe" || goto :fail
echo.

echo [edge_video_preprocess] step 6/7 configure + build infer
echo ------------------------------------------------------------
cmd /c "call \"%VCVARS%\" && cmake -S \"%ACTION_C_ROOT%\demo\edge_video_preprocess\infer\" -B \"%BUILD_ROOT%\infer\" -G Ninja -DCMAKE_C_COMPILER=clang -DACTION_C_GENERATED_DIR=\"%GENERATED_DIR%\" && cmake --build \"%BUILD_ROOT%\infer\"" || goto :fail
echo.

echo [edge_video_preprocess] step 7/7 run infer
echo ------------------------------------------------------------
"%BUILD_ROOT%\infer\edge_video_preprocess_infer.exe" || goto :fail
echo.

echo ============================================================
echo [edge_video_preprocess] demo completed successfully
echo ============================================================
goto :eof

:fail
echo.
echo ============================================================
echo [edge_video_preprocess] demo FAILED
echo ============================================================
exit /b 1

@echo off
setlocal

set "ACTION_C_ROOT=%~dp0..\.."
for %%I in ("%ACTION_C_ROOT%") do set "ACTION_C_ROOT=%%~fI"
set "BUILD_ROOT=%ACTION_C_ROOT%\build\demo\mnist_cnn"
set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

echo [mnist_cnn] step 1/6 configure + build generate
cmd /c "call \"%VCVARS%\" && cmake -S \"%ACTION_C_ROOT%\demo\mnist_cnn\generate\" -B \"%BUILD_ROOT%\generate\" -G Ninja -DCMAKE_C_COMPILER=clang && cmake --build \"%BUILD_ROOT%\generate\"" || goto :fail

echo [mnist_cnn] step 2/6 run generate
"%BUILD_ROOT%\generate\mnist_cnn_generate.exe" || goto :fail

echo [mnist_cnn] step 3/6 configure + build train
cmd /c "call \"%VCVARS%\" && cmake -S \"%ACTION_C_ROOT%\demo\mnist_cnn\train\" -B \"%BUILD_ROOT%\train\" -G Ninja -DCMAKE_C_COMPILER=clang && cmake --build \"%BUILD_ROOT%\train\"" || goto :fail

echo [mnist_cnn] step 4/6 run train
"%BUILD_ROOT%\train\mnist_cnn_train.exe" || goto :fail

echo [mnist_cnn] step 5/6 configure + build infer
cmd /c "call \"%VCVARS%\" && cmake -S \"%ACTION_C_ROOT%\demo\mnist_cnn\infer\" -B \"%BUILD_ROOT%\infer\" -G Ninja -DCMAKE_C_COMPILER=clang && cmake --build \"%BUILD_ROOT%\infer\"" || goto :fail

echo [mnist_cnn] step 6/6 run infer
"%BUILD_ROOT%\infer\mnist_cnn_infer.exe" || goto :fail

echo [mnist_cnn] demo completed successfully
goto :eof

:fail
echo [mnist_cnn] demo failed
exit /b 1

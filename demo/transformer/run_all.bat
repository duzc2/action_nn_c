@echo off
REM =============================================================================
REM Transformer Demo: 完整流程脚本
REM 步骤: 编译运行生成器 -> 编译运行训练 -> 编译运行推理
REM =============================================================================

echo.
echo ========================================
echo Transformer Demo - Full Pipeline
echo ========================================
echo.

REM 步骤1: 配置 CMake
echo [Step 1/6] Configuring CMake...
call cmake -S .. -B . -G "MinGW Makefiles"
if errorlevel 1 (
    echo ERROR: CMake configuration failed
    exit /b 1
)

REM 步骤2: 编译并运行生成器
echo [Step 2/6] Building and running transformer_generate...
call cmake --build . --target transformer_generate
if errorlevel 1 (
    echo ERROR: Build transformer_generate failed
    exit /b 1
)
echo Running transformer_generate...
cd demo\transformer\Debug
transformer_generate.exe
cd ..\..\..
if errorlevel 1 (
    echo ERROR: transformer_generate failed
    exit /b 1
)

REM 步骤3: 编译并运行训练
echo [Step 3/6] Building and running transformer_train...
call cmake --build . --target transformer_train
if errorlevel 1 (
    echo ERROR: Build transformer_train failed
    exit /b 1
)
echo Running transformer_train...
cd demo\transformer\Debug
transformer_train.exe
cd ..\..\..
if errorlevel 1 (
    echo ERROR: transformer_train failed
    exit /b 1
)

REM 步骤4: 编译并运行推理
echo [Step 4/6] Building and running transformer_infer...
call cmake --build . --target transformer_infer
if errorlevel 1 (
    echo ERROR: Build transformer_infer failed
    exit /b 1
)
echo Running transformer_infer...
echo.
echo Input: text (type quit to exit)
echo.
echo hello > temp_input.txt
echo quit >> temp_input.txt
cd demo\transformer\Debug
transformer_infer.exe < ..\..\temp_input.txt
cd ..\..\..
del temp_input.txt

echo.
echo ========================================
echo Transformer Demo - Completed Successfully!
echo ========================================
echo.

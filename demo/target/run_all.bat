@echo off
REM =============================================================================
REM Target Demo: 完整流程脚本
REM 步骤: 编译运行生成器 -> 编译运行训练 -> 编译运行推理
REM =============================================================================

echo.
echo ========================================
echo Target Demo - Full Pipeline
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
echo [Step 2/6] Building and running target_generate...
call cmake --build . --target target_generate
if errorlevel 1 (
    echo ERROR: Build target_generate failed
    exit /b 1
)
echo Running target_generate...
cd demo\target\Debug
target_generate.exe
cd ..\..\..
if errorlevel 1 (
    echo ERROR: target_generate failed
    exit /b 1
)

REM 步骤3: 编译并运行训练
echo [Step 3/6] Building and running target_train...
call cmake --build . --target target_train
if errorlevel 1 (
    echo ERROR: Build target_train failed
    exit /b 1
)
echo Running target_train...
cd demo\target\Debug
target_train.exe
cd ..\..\..
if errorlevel 1 (
    echo ERROR: target_train failed
    exit /b 1
)

REM 步骤4: 编译并运行推理
echo [Step 4/6] Building and running target_infer...
call cmake --build . --target target_infer
if errorlevel 1 (
    echo ERROR: Build target_infer failed
    exit /b 1
)
echo Running target_infer...
echo.
echo Input: current_x current_y target_x target_y
echo.
echo 0 0 10 10 > temp_input.txt
cd demo\target\Debug
target_infer.exe < ..\..\temp_input.txt
cd ..\..\..
del temp_input.txt

echo.
echo ========================================
echo Target Demo - Completed Successfully!
echo ========================================
echo.

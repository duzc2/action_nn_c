@echo off
REM =============================================================================
REM SevenSeg Demo: 完整流程脚本
REM 步骤: 编译运行生成器 -> 编译运行训练 -> 编译运行推理
REM =============================================================================

echo.
echo ========================================
echo SevenSeg Demo - Full Pipeline
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
echo [Step 2/6] Building and running sevenseg_generate...
call cmake --build . --target sevenseg_generate
if errorlevel 1 (
    echo ERROR: Build sevenseg_generate failed
    exit /b 1
)
echo Running sevenseg_generate...
cd demo\sevenseg\Debug
sevenseg_generate.exe
cd ..\..\..
if errorlevel 1 (
    echo ERROR: sevenseg_generate failed
    exit /b 1
)

REM 步骤3: 编译并运行训练
echo [Step 3/6] Building and running sevenseg_train...
call cmake --build . --target sevenseg_train
if errorlevel 1 (
    echo ERROR: Build sevenseg_train failed
    exit /b 1
)
echo Running sevenseg_train...
cd demo\sevenseg\Debug
sevenseg_train.exe
cd ..\..\..
if errorlevel 1 (
    echo ERROR: sevenseg_train failed
    exit /b 1
)

REM 步骤4: 编译并运行推理
echo [Step 4/6] Building and running sevenseg_infer...
call cmake --build . --target sevenseg_infer
if errorlevel 1 (
    echo ERROR: Build sevenseg_infer failed
    exit /b 1
)
echo Running sevenseg_infer...
echo.
echo Input: digit 0-9
echo.
echo 5 > temp_input.txt
cd demo\sevenseg\Debug
sevenseg_infer.exe < ..\..\temp_input.txt
cd ..\..\..
del temp_input.txt

echo.
echo ========================================
echo SevenSeg Demo - Completed Successfully!
echo ========================================
echo.

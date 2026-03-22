@echo off
REM =============================================================================
REM Move Demo: 完整流程脚本
REM 步骤: 编译运行生成器 -> 编译运行训练 -> 编译运行推理
REM =============================================================================

echo.
echo ========================================
echo Move Demo - Full Pipeline
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
echo [Step 2/6] Building and running move_generate...
call cmake --build . --target move_generate
if errorlevel 1 (
    echo ERROR: Build move_generate failed
    exit /b 1
)
echo Running move_generate...
cd demo\move\Debug
move_generate.exe
cd ..\..\..
if errorlevel 1 (
    echo ERROR: move_generate failed
    exit /b 1
)

REM 步骤3: 编译并运行训练
echo [Step 3/6] Building and running move_train...
call cmake --build . --target move_train
if errorlevel 1 (
    echo ERROR: Build move_train failed
    exit /b 1
)
echo Running move_train...
cd demo\move\Debug
move_train.exe
cd ..\..\..
if errorlevel 1 (
    echo ERROR: move_train failed
    exit /b 1
)

REM 步骤4: 编译并运行推理
echo [Step 4/6] Building and running move_infer...
call cmake --build . --target move_infer
if errorlevel 1 (
    echo ERROR: Build move_infer failed
    exit /b 1
)
echo Running move_infer...
echo.
echo Input: 5 5 (start position), then commands: 0 0 0 4 (3x up, then stop)
echo.
echo 5 5 > temp_input.txt
echo 0 >> temp_input.txt
echo 0 >> temp_input.txt
echo 0 >> temp_input.txt
echo 4 >> temp_input.txt
cd demo\move\Debug
move_infer.exe < ..\..\temp_input.txt
cd ..\..\..
del temp_input.txt

echo.
echo ========================================
echo Move Demo - Completed Successfully!
echo ========================================
echo.

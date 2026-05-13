@echo off
setlocal enabledelayedexpansion

set PHASE=%1
if "%PHASE%"=="" set PHASE=all

set ROOT=%~dp0..\..
set BUILD_TYPE=%BUILD_TYPE%
if "%BUILD_TYPE%"=="" set BUILD_TYPE=Debug

if "%PHASE%"=="all" (
    call :build_and_run generate
    if errorlevel 1 goto :end
    call :build_and_run train
    if errorlevel 1 goto :end
    call :build_and_run infer
    if errorlevel 1 goto :end
) else (
    call :build_and_run %PHASE%
    if errorlevel 1 goto :end
)

echo [weather] demo completed successfully
goto :end

:build_and_run
set PH=%~1
set BUILD_DIR=%ROOT%build\demo\weather\%PH%

echo [weather] configure + build %PH%
cmake -S "%~dp0%PH%" -B "%BUILD_DIR%" -DCMAKE_BUILD_TYPE=%BUILD_TYPE%
if errorlevel 1 exit /b 1
cmake --build "%BUILD_DIR%" --config %BUILD_TYPE%
if errorlevel 1 exit /b 1

echo [weather] run %PH%
set EXE=%BUILD_DIR%\%BUILD_TYPE%\weather_%PH%.exe
if not exist "%EXE%" set EXE=%BUILD_DIR%\weather_%PH%.exe

set EXE_DATA_DIR=%BUILD_DIR%\%BUILD_TYPE%\..\data
if not exist "%EXE_DATA_DIR%" mkdir "%EXE_DATA_DIR%"

if "%PH%"=="infer" (
    set SHARED_DATA=%ROOT%build\demo\weather\data
    if exist "%SHARED_DATA%\weights_beijing.bin" (
        copy /Y "%SHARED_DATA%\weights_*.bin" "%EXE_DATA_DIR%\" >nul
    )
)

"%EXE%"
if errorlevel 1 exit /b 1

if "%PH%"=="generate" (
    set SHARED_DATA=%ROOT%build\demo\weather\data
    if exist "%EXE_DATA_DIR%\infer.c" (
        if exist "%SHARED_DATA%" rmdir /S /Q "%SHARED_DATA%" 2>nul
        mkdir "%SHARED_DATA%"
        xcopy /Y /Q "%EXE_DATA_DIR%\*" "%SHARED_DATA%\" >nul
    )
)

if "%PH%"=="train" (
    set SHARED_DATA=%ROOT%build\demo\weather\data
    if not exist "%SHARED_DATA%" mkdir "%SHARED_DATA%"
    if exist "%EXE_DATA_DIR%\weights_beijing.bin" (
        copy /Y "%EXE_DATA_DIR%\weights_*.bin" "%SHARED_DATA%\" >nul
    )
)

exit /b 0

:end

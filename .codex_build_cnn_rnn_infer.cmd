@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
cmake --build build\demo\cnn_rnn_react\infer --config Debug -- -v

# 从零开始用户故事

## 场景

用户希望训练一个控制模型，并且未来可扩展到复杂拓扑（分支、共享输入、异构子网）。

## 路径

1. 在 CMakeLists 通过开关启用需要的网络类型  
2. 第一次编译构建 profiler 组件  
3. 用户程序构建网络结构体并调用 profiler 生成训练/推理代码  
4. 第二次编译训练工程并运行，导出权重 `.bin` 与权重 `.c`  
5. 第三次编译推理工程并运行，加载 `.bin` 或使用权重 `.c`  
6. 跑测试确认一致性

## 最小闭环命令

```bash
cmake -S . -B build -DENABLE_NN_TRANSFORMER=ON
cmake --build build --config Debug --target profiler
build/Debug/profiler.exe
cmake --build build --config Debug --target sevenseg_train
cmake --build build --config Debug --target sevenseg_infer_bin
ctest --test-dir build -C Debug --output-on-failure
```

说明：`sevenseg` 是 `demo/sevenseg` 下的示例工程，不属于 `src` 下的 `nn` 网络类型。

## 成果

- 网络类型启用由 CMake 开关控制，未启用类型不会进入编译
- `src/nn/` 必须实现常见网络结构：`rnn`、`cnn`、`knn`、`transformer`
- 新增网络类型只需新增实现并更新注册配置/宏与开关
- 训练与推理分工程构建，依赖边界清晰且可独立部署

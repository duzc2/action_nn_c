# 从零开始用户故事

## 场景

用户希望训练一个控制模型，并且未来可扩展到复杂拓扑（分支、共享输入、异构子网）。

## 路径

1. 在 `config_user.h` 定义图节点与边  
2. 运行 profiler 生成 `network_def.h`  
3. 训练导出权重  
4. 初始化推理运行时并在线推理  
5. 跑测试确认一致性

## 最小闭环命令

```bash
cmake -S . -B build
cmake --build build --config Debug
build/Debug/profiler.exe -o src/include/network_def.h
cmake --build build --config Debug --target sevenseg_train
cmake --build build --config Debug --target sevenseg_infer_bin
ctest --test-dir build -C Debug --output-on-failure
```

## 成果

- 网络配置从“模型类型选择”升级为“拓扑图描述”
- 训练与推理使用同一规格对象
- demo 与测试走同一配置链路，避免文档与实现分叉

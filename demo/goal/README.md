# goal_demo

`goal_demo` 演示“仅状态驱动”的训练与推理流程。

## 运行

```bash
cmake --build build --config Debug --target goal_demo
build/demo/goal/Debug/goal_demo.exe
```

## 配置来源

- 网络图配置：`src/include/config_user.h`
- 规格生成：`src/include/network_def.h` 中 `network_def_build_spec`

## 关键特性

- 训练与推理使用同一 `NetworkSpec`
- 权重数量通过 `workflow_weights_count(spec)` 校验
- 不存在旧模型类型兼容分支

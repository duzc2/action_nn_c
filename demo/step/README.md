# step_demo

`step_demo` 演示“命令 + 状态”驱动的一步推理闭环。

## 运行

```bash
cmake --build build --config Debug --target step_demo
build/demo/step/Debug/step_demo.exe
```

## 配置来源

- 图拓扑：`config_user.h`
- spec 构建：`network_def_build_spec`

## 说明

- 推理入口统一为 `workflow_runtime_init(..., spec)`
- 训练入口统一为 `workflow_train_from_memory(...network_spec...)`
- 旧兼容接口已移除

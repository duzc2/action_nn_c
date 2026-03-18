# sevenseg demo

sevenseg 子工程覆盖三种推理路径：

- 二进制权重运行时：`sevenseg_infer_bin`
- C 数组导出权重：`sevenseg_infer_c_array`
- C 函数导出网络：`sevenseg_infer_c_func`

并提供 `sevenseg_benchmark` 进行对比测试。

## 构建

```bash
cmake --build build --config Debug --target sevenseg_train
cmake --build build --config Debug --target sevenseg_infer_bin
cmake --build build --config Debug --target sevenseg_infer_c_array
cmake --build build --config Debug --target sevenseg_infer_c_func
cmake --build build --config Debug --target sevenseg_benchmark
```

## 运行

```bash
build/demo/sevenseg/Debug/sevenseg_train.exe --export-only demo/sevenseg/data
build/demo/sevenseg/Debug/sevenseg_infer_bin.exe
build/demo/sevenseg/Debug/sevenseg_infer_c_array.exe
build/demo/sevenseg/Debug/sevenseg_infer_c_func.exe
build/demo/sevenseg/Debug/sevenseg_benchmark.exe
```

## 数据目录

程序会自动尝试多个候选目录解析数据文件；也可手动传入目录参数：

```bash
build/demo/sevenseg/Debug/sevenseg_infer_bin.exe demo/sevenseg/data
```

## 与新设计的关系

- sevenseg 也使用图拓扑配置生成 spec
- 所有训练/推理入口都显式依赖同一 spec
- 文档与实现保持单一流程，不保留旧接口说明

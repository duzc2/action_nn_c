# tokenizer 模块说明

## 目标

将命令文本映射为 token id 序列，并在训练与推理中复用同一词表。

## 相关文件

- `tokenizer.c`
- `tokenizer_runtime.c`
- `src/include/tokenizer.h`

## 在流程中的位置

- 训练：`workflow_train_*` 使用词表处理命令样本
- 推理：`workflow_prepare_tokenizer` 初始化运行时分词器

## 与网络拓扑设计的关系

tokenizer 与网络图结构解耦。  
图拓扑决定网络执行结构；tokenizer 只负责输入序列编码。

## 常见检查项

- 词表文件是否存在
- 未知 token 的处理是否符合预期
- `MAX_SEQ_LEN` 是否与样本长度匹配

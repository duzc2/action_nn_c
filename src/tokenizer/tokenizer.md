tokenizer 目录用于放置分词与输入文本解析相关模块。

当前文件：
- tokenizer.c：分词、词表管理、编码解码、词表二进制读写的主体实现。
- tokenizer_runtime.c：对核心 tokenizer API 的目录级封装入口，供上层按目录职责调用。

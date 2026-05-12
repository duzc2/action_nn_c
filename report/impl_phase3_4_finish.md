# 阶段 3+4 — 前端修复与最终验证

**关联**: [implementation_steps.md](implementation_steps.md) (总索引)
**前置**: 阶段 0 完成（阶段 3 可与阶段 1-2 并行）
**阶段 4 前置**: 阶段 1-3 全部完成

---

# 阶段 3：前端与独立修复

> 可与阶段 1-2 并行执行。不依赖轨 A/B/C 的输出。

---

## 步骤 D1 — 删除 Web Editor 遗留代码

### 代码变更

```bash
rm web_editor/js/editor.js
rm web_editor/js/nodes.js
rm web_editor/css/style.css
rm web_editor/start.sh
```

### 步骤测试

**测试用例**:

| # | 测试 | 命令 | 预期 |
|---|------|------|------|
| 1 | 文件已删除 | `ls web_editor/js/editor.js 2>&1` | `No such file` |
| 2 | 文件已删除 | `ls web_editor/js/nodes.js 2>&1` | `No such file` |
| 3 | 文件已删除 | `ls web_editor/css/style.css 2>&1` | `No such file` |
| 4 | 文件已删除 | `ls web_editor/start.sh 2>&1` | `No such file` |
| 5 | Web Editor 仍可构建 | `cd web_editor && npm run build 2>&1` | 无 import 错误 |

### 单步验证

```bash
ls web_editor/js/editor.js web_editor/js/nodes.js web_editor/css/style.css web_editor/start.sh
```

### 全量回归

```bash
cd web_editor && npm install && npm run build
```

### 门控

- [x] 4 个文件已删除
- [x] `npm run build` 成功
- [ ] → **git commit** `"cleanup(D1): remove legacy Web Editor files"`
- [ ] → 进入 D2

**消灭问题**: FUN-H01, FUN-H06

---

## 步骤 D2 — Web Editor 安全加固

### 代码变更

**修改**: `web_editor/index.html`

```html
<head>
    <meta http-equiv="Content-Security-Policy"
          content="default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline';">
</head>
```

```html
<button onclick="addNode('input')" class="toolbar-btn"
        aria-label="Add input node">[+Input]</button>
<button onclick="addNode('mlp')" class="toolbar-btn"
        aria-label="Add MLP hidden layer">[+MLP]</button>
<button onclick="addNode('output')" class="toolbar-btn"
        aria-label="Add output node">[+Output]</button>
<div id="editor" role="application" aria-label="Neural network editor"></div>
```

**修改**: `web_editor/src/main.js` — innerHTML → textContent

```javascript
// 旧
outputDiv.innerHTML = `...${content.replace(/</g, '&lt;').replace(/>/g, '&gt;')}...`;

// 新
const pre = document.createElement('pre');
pre.textContent = content;
outputDiv.textContent = '';
outputDiv.appendChild(pre);
```

**修改**: `web_editor/src/main.js` — 添加异步错误处理

```javascript
initEditor().then(() => {
    window.editor = editor;
    window.area = area;
}).catch((err) => {
    document.getElementById('editor').innerHTML =
        '<div style="color:red;padding:20px;">Editor failed to initialize: '
        + err.message + '</div>';
    console.error('Editor init error:', err);
});
```

**修改**: `web_editor/src/main.js` — 添加 editor 未初始化检查

```javascript
function addNode(type) {
    if (!editor) {
        console.error('Editor not initialized yet');
        return;
    }
    // ...
}
```

**修改**: `web_editor/vite.config.js`

```javascript
server: {
    port: 5173,
    host: '127.0.0.1'
}
```

### 步骤测试

**测试用例**:

| # | 测试 | 操作 | 预期 |
|---|------|------|------|
| 1 | CSP 头存在 | 浏览器 DevTools → Network → 查看 index.html 响应头 | 含 `Content-Security-Policy` |
| 2 | 键盘可访问性 | Tab 键导航 | 可聚焦到按钮 |
| 3 | 错误处理 | 故意让 editor 初始化失败 | 显示红色错误信息而非空白 |
| 4 | innerHTML 消除 | `grep innerHTML web_editor/src/main.js` | 无输出（或仅保留安全的） |
| 5 | vite 仅本地绑定 | `netstat -an \| grep 5173` | 监听 `127.0.0.1:5173` |

### 单步验证

```bash
cd web_editor && npm run dev
# 在浏览器中打开 http://localhost:5173
# 手动操作：添加节点、连接、导出
```

### 全量回归

```bash
cd web_editor && npm run build
```

### 门控

- [x] CSP 已添加
- [x] aria-label 已添加
- [x] `npm run build` 成功
- [x] 浏览器中操作正常
- [ ] → **git commit** `"security(D2): harden Web Editor (CSP, XSS, error handling, aria)"`
- [ ] → 进入 D3

**消灭问题**: SEC-L04, SEC-L05, SEC-M05, SEC-M06, SEC-M07, USA-L05

---

## 步骤 D3 — Web Editor 工程完善

### 代码变更

```bash
cd web_editor
npm install
git add package-lock.json
```

**修改**: `web_editor/package.json`

```json
{
    "engines": {
        "node": ">=18.0.0"
    }
}
```

**修改**: `web_editor/vite.config.js` — 删除冗余的 `port: 5173`

### 步骤测试

**测试用例**:

| # | 测试 | 命令 | 预期 |
|---|------|------|------|
| 1 | package-lock.json 存在 | `ls web_editor/package-lock.json` | 存在 |
| 2 | npm ci 成功 | `cd web_editor && rm -rf node_modules && npm ci` | 成功 |
| 3 | engines 字段 | `node -e "const p=require('./web_editor/package.json'); console.log(p.engines.node)"` | `>=18.0.0` |

### 单步验证

```bash
ls web_editor/package-lock.json
cd web_editor && npm ci && npm run build
```

### 门控

- [x] package-lock.json 已提交
- [x] `npm ci` 成功
- [ ] → **git commit** `"chore(D3): improve Web Editor project hygiene (lockfile, engines)"`
- [ ] → 进入 D4

**消灭问题**: FUN-H02, USA-L06, USA-L07, USA-L12

---

## 步骤 D4 — Web Editor 连接代码生成

### 代码变更

**修改**: `web_editor/src/main.js` — `buildNetworkState` 函数

```javascript
function buildNetworkState() {
    const connections = [];
    if (editor) {
        editor.getNodes().forEach(node => {
            Object.entries(node.inputs).forEach(([key, input]) => {
                if (input.connections && input.connections.length > 0) {
                    input.connections.forEach(conn => {
                        connections.push({
                            source_node: conn.node,
                            source_key: conn.key,
                            target_node: node,
                            target_key: key
                        });
                    });
                }
            });
        });
    }
    return { nodes: collectNodes(), connections };
}
```

**修改**: `web_editor/src/main.js` — C 代码导出函数

```javascript
function exportC() {
    const state = buildNetworkState();
    let code = '';
    // ... 节点创建代码 ...

    state.connections.forEach((conn, i) => {
        code += `    conn_${i} = nn_connection_def_create(\n`;
        code += `        "conn_${conn.source_key}_to_${conn.target_key}",\n`;
        code += `        "${conn.source_key}", "${conn.target_key}");\n`;
    });
    // ...
}
```

### 步骤测试

**测试文件**: `web_editor/tests/connections.test.js`

```javascript
import { describe, it, expect } from 'vitest';

describe('buildNetworkState connections', () => {
    it('returns empty connections when no nodes connected', () => {
        // 此测试需模拟 Rete editor 状态
        const state = buildNetworkState();
        expect(state.connections).toBeDefined();
        expect(Array.isArray(state.connections)).toBe(true);
    });

    it('exportC produces connection_def_create calls', () => {
        const code = exportC();
        expect(code).toContain('nn_connection_def_create');
    });
});
```

### 单步验证

```bash
cd web_editor && npm test
# 或手动：浏览器中拖入 Input → MLP → Output，连接后导出 C 代码
```

### 门控

- [x] 导出 C 代码含 `nn_connection_def_create`
- [x] `npm run build` 成功
- [ ] → **git commit** `"feat(D4): implement Web Editor connection code generation"`
- [ ] → 进入 D5

**消灭问题**: FUN-H03

---

## 步骤 D5 — Wasm 运行时补全

### 代码变更

**修改**: `wasm/src/wasm_exports.c`

```c
WASM_EXPORT
int action_c_wasm_infer_step(const void* request_ptr) {
    if (!request_ptr) return ACTION_C_ERR_NULL_POINTER;
    const NNInferRequest* req = (const NNInferRequest*)request_ptr;
    return nn_infer_runtime_step(req);
}

WASM_EXPORT
void action_c_wasm_infer_destroy(void* ctx_ptr) {
    if (!ctx_ptr) return;
    NNInferContext* ctx = (NNInferContext*)ctx_ptr;
    if (ctx->backend && ctx->backend->destroy) {
        ctx->backend->destroy(ctx->backend_ctx);
    }
    free(ctx_ptr);
}
```

**修改**: `wasm/examples/browser_demo.html` — 修复 JS 优先级错误

```javascript
// 旧（错误）
${(memSize - heapSize) / 1024 / 1024.toFixed(2)}

// 新
${((memSize - heapSize) / 1024 / 1024).toFixed(2)}
```

**修改**: `wasm/examples/browser_demo.html` — 添加 try-catch

```javascript
try {
    const inferStep = wasmModule.cwrap('action_c_wasm_infer_step', 'number', ['number']);
} catch (e) {
    console.error('Failed to wrap infer_step:', e);
    return;
}
```

**修改**: `wasm/js/action_nn_c.js` — 将 `throw` 改为 `Promise.reject`

```javascript
export default function ActionNnC(config) {
    return Promise.reject(new Error(
        'Wasm module not built. Please run ./scripts/build_wasm.sh first.'
    ));
}
```

### 步骤测试

**测试用例**:

| # | 测试 | 命令/操作 | 预期 |
|---|------|----------|------|
| 1 | Wasm 构建 | `bash wasm/scripts/build_wasm.sh` | 成功 |
| 2 | 导出函数列表 | `wasm-objdump -x build/wasm/*.wasm \| grep action_c_wasm` | 含 `infer_step` `infer_destroy` |
| 3 | 浏览器 demo | 打开 `wasm/examples/browser_demo.html` | 正常加载 |
| 4 | JS 模板语法 | `node -e "..."` 验证 `.toFixed` 调用 | 无语法错误 |
| 5 | TypeScript 声明 | `npx tsc --noEmit wasm/js/action_nn_c.d.ts` | 无类型错误 |

### 单步验证

```bash
# 需 Emscripten
bash wasm/scripts/build_wasm.sh

# 导出检查
wasm-objdump -x build/wasm/action_c_wasm.wasm | grep -i "export" | grep "action_c_wasm"
```

### 全量回归

```bash
cmake --preset debug
cmake --build build/debug --config Debug 2>&1 | grep -E "error|warning"
```

### 门控

- [x] Wasm 构建成功
- [x] 导出函数含 infer_step / infer_destroy
- [x] 全量编译通过
- [ ] → **git commit** `"fix(D5): complete Wasm runtime exports and fix JS errors"`
- [ ] → 进入 D6

**消灭问题**: STB-M09, STB-M10, FUN-H04, FUN-L10

---

## 步骤 D6 — 独立小修复

### 代码变更

| 动作 | 文件 | 消灭问题 |
|------|------|---------|
| 添加 `#ifdef __linux__` 检查 `/proc/self/exe` | `demo/demo_runtime_paths.h` | STB-L06 |
| 删除 Markdown 代码围栏 | `.gitignore:1,74` | USA-L09 |
| 替换 `"????????"` 为有意义的描述 | `cnn_rnn_react_scene.c:535`, `nested_nav_scene.h:536` | USA-L02, USA-L03 |
| 删除 `CsLabelSegment` 废弃结构体 | `cs_tool_common.h` | MNT-L04 |
| 删除废弃函数 | `cs_tool_common.c` | MNT-L04 |
| 添加 `cnn_rnn_react/run_demo.sh` | `demo/cnn_rnn_react/` | MNT-L06 |

### 步骤测试

**测试用例**:

| # | 测试 | 验证内容 | 命令 |
|---|------|---------|------|
| 1 | `/proc/self/exe` 宏保护 | 在 Linux 上编译通过 | `gcc -c demo/demo_runtime_paths.h` |
| 2 | `.gitignore` 格式正确 | `git check-ignore foo.md` | 行为正确 |
| 3 | 无 `"????????"` 残留 | `grep -r '????????' src/ demo/` | 无输出 |
| 4 | `CsLabelSegment` 已删除 | `grep CsLabelSegment demo/cs/tools/*.h` | 无输出 |
| 5 | run_demo.sh 存在 | `ls demo/cnn_rnn_react/run_demo.sh` | 存在 |

### 单步验证

```bash
grep -r '????????' src/ demo/ 2>/dev/null
grep CsLabelSegment demo/cs/tools/*.h 2>/dev/null
ls demo/cnn_rnn_react/run_demo.sh
```

### 全量回归

```bash
cmake --build build/ --config Debug 2>&1 | grep -E "error|warning"
ctest --output-on-failure
```

### 门控

- [x] 6 项修复均完成
- [x] 全量编译零错误
- [x] ctest 全量通过
- [ ] → **git commit** `"fix(D6): six independent small fixes (paths, docs, dead code)"`
- [ ] → 进入 D7

---

## 步骤 D7 — CS 工具 JSON 解析加固

### 代码变更

**修改**: `demo/cs/tools/cs_dataset_build.c`

```c
/* 旧：硬编码偏移 */
cursor += 9;  /* 跳过 "key_name" */

/* 新：基于键名查找 */
static const char* cs_json_find_key(const char* cursor, const char* key) {
    char search[64];
    snprintf(search, sizeof(search), "\"%s\"", key);
    const char* found = strstr(cursor, search);
    if (!found) return NULL;
    return found + strlen(search);
}
```

同样修改 `cs_dataset_report.c` 中的硬编码偏移。

### 步骤测试

**测试文件**: `tests/demo/test_cs_json_parse.c`

```c
#include "test_harness.h"
#include <string.h>

/* 复制 cs_json_find_key 函数用于测试 */
static const char* cs_json_find_key(const char* cursor, const char* key) {
    char search[64];
    snprintf(search, sizeof(search), "\"%s\"", key);
    const char* found = strstr(cursor, search);
    if (!found) return NULL;
    return found + strlen(search);
}

TEST(finds_key_in_json) {
    const char* json = "{\"network_type\": \"mlp\", \"layers\": 3}";

    const char* val = cs_json_find_key(json, "network_type");
    ASSERT_NOT_NULL(val, "network_type found");
    ASSERT_TRUE(strstr(val, "mlp") != NULL, "value contains mlp");

    const char* val2 = cs_json_find_key(json, "layers");
    ASSERT_NOT_NULL(val2, "layers found");
    ASSERT_TRUE(strstr(val2, "3") != NULL, "value contains 3");
}

TEST(returns_null_for_missing_key) {
    const char* json = "{\"network_type\": \"mlp\"}";
    const char* val = cs_json_find_key(json, "nonexistent");
    ASSERT_NULL(val, "missing key returns NULL");
}

int main(void) {
    RUN_TEST(finds_key_in_json);
    RUN_TEST(returns_null_for_missing_key);
    printf("1..%d\n", _tests_run);
    printf("Results: %d pass, %d fail, %d total\n",
           _tests_pass, _tests_fail, _tests_run);
    return _tests_fail > 0 ? 1 : 0;
}
```

### 单步验证

```bash
cmake --build build/ --target test_cs_json_parse
ctest -R test_cs_json_parse --output-on-failure

# 构建 CS 工具
cmake -B build/cs_tools -S demo/cs/tools/
cmake --build build/cs_tools --config Debug
```

### 门控

- [x] test_cs_json_parse 全部通过
- [x] CS 工具构建成功
- [ ] → **git commit** `"fix(D7): replace hardcoded JSON offsets with key-based lookup in CS tools"`
- [ ] → 进入 D8

**消灭问题**: FUN-M07

---

## 步骤 D8 — demo 脚本去重

### 代码变更

**新建**: `scripts/run_demo.sh` (单一参数化脚本)

```bash
#!/bin/bash
set -e

DEMO=$1
PHASE=${2:-"all"}

case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) EXE=".exe" ;;
    *)                    EXE="" ;;
esac

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_TYPE="${BUILD_TYPE:-Debug}"

build_and_run() {
    local demo=$1 phase=$2
    local build_dir="$ROOT/build/demo/$demo/$phase"
    cmake -S "$ROOT/demo/$demo/$phase" -B "$build_dir" \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    cmake --build "$build_dir" --config "$BUILD_TYPE"
    "$build_dir/$BUILD_TYPE/${demo}_${phase}${EXE}"
}

if [ "$PHASE" = "all" ]; then
    build_and_run "$DEMO" generate
    build_and_run "$DEMO" train
    build_and_run "$DEMO" infer
else
    build_and_run "$DEMO" "$PHASE"
fi
```

### 步骤测试

**测试用例**:

| # | 测试 | 命令 | 预期 |
|---|------|------|------|
| 1 | move 三阶段 | `bash scripts/run_demo.sh move all` | 三阶段 exit 0 |
| 2 | sevenseg 单 generate | `bash scripts/run_demo.sh sevenseg generate` | exit 0 |
| 3 | target 训练后推理 | `bash scripts/run_demo.sh target train && bash scripts/run_demo.sh target infer` | exit 0 |
| 4 | 无参数报错 | `bash scripts/run_demo.sh` | `set -e` 后 `$1` 为空时退出 |
| 5 | 不存在的 demo | `bash scripts/run_demo.sh nonexistent generate` | 报错退出 |

### 单步验证

```bash
bash scripts/run_demo.sh move all
bash scripts/run_demo.sh sevenseg all
bash scripts/run_demo.sh target all
```

### 全量回归

```bash
for demo in move sevenseg target transformer mnist mnist_cnn \
            nested_nav road_graph_nav cnn_rnn_react hybrid_route; do
    echo "=== $demo ==="
    bash scripts/run_demo.sh "$demo" all 2>&1 | tail -2
done
```

### 门控

- [x] 全量 10 个 demo 通过
- [ ] → **git commit** `"refactor(D8): create unified run_demo.sh replacing per-demo scripts"`
- [ ] → 阶段 3 完成

---

# 阶段 4：验证与清理

> 阶段 1-3 全部完成后执行。

---

## 步骤 V1 — 删除死代码和废弃目录

### 代码变更

```bash
# CNN Dual-Pool 已合并
rm -rf src/nn/types/cnn_dual_pool/

# 旧构建脚本
rm .codex_build_cnn_rnn_infer.cmd
```

### 步骤测试

| # | 测试 | 命令 | 预期 |
|---|------|------|------|
| 1 | cnn_dual_pool 已删除 | `ls src/nn/types/cnn_dual_pool/ 2>&1` | No such file |
| 2 | old .cmd 已删除 | `ls .codex_build_cnn_rnn_infer.cmd 2>&1` | No such file |
| 3 | 全量编译 | `cmake --preset debug && cmake --build build/debug` | 零错误 |

### 单步验证

```bash
cmake --preset debug
cmake --build build/debug --config Debug 2>&1 | grep -E "error|warning"
```

### 门控

- [x] 死代码已删除
- [x] 全量编译零错误
- [ ] → **git commit** `"cleanup(V1): delete dead code and obsolete directories"`
- [ ] → 进入 V2

---

## 步骤 V2 — 全量构建与测试验证

### 操作

```bash
# 1. 全量构建（debug preset）
cmake --preset debug
cmake --build build/debug --config Debug -- -j$(nproc)

# 2. 运行全量 ctest
cd build/debug && ctest --output-on-failure

# 3. 构建+运行所有 10 个 demo
for demo in move sevenseg target transformer mnist mnist_cnn \
            nested_nav road_graph_nav cnn_rnn_react hybrid_route; do
    bash scripts/run_demo.sh "$demo" all
done
```

### 测试用例

| # | 测试 | 预期 |
|---|------|------|
| 1 | 全量编译零错误零警告 | PASS |
| 2 | ctest 全量通过 | 所有测试 PASS |
| 3 | move demo 全流程 | exit 0 |
| 4 | sevenseg demo 全流程 | exit 0 |
| 5 | target demo 全流程 | exit 0 |
| 6 | transformer demo 全流程 | exit 0 |
| 7 | mnist demo 全流程 | exit 0 |
| 8 | mnist_cnn demo 全流程 | exit 0 |
| 9 | nested_nav demo 全流程 | exit 0 |
| 10 | road_graph_nav demo 全流程 | exit 0 |
| 11 | cnn_rnn_react demo 全流程 | exit 0 |
| 12 | hybrid_route demo 全流程 | exit 0 |

### 单步验证

```bash
cd build/debug && ctest --output-on-failure --verbose
```

### 门控

- [x] 全量 ctest 通过 (预期 ~30 个测试)
- [x] 10 个 demo 全流程通过
- [ ] → **git commit** `"test(V2): full build + ctest + demo regression verification"`
- [ ] → 进入 V3

---

## 步骤 V3 — Sanitizer 验证

### 操作

```bash
# 1. 使用 sanitized preset
cmake --preset sanitized
cmake --build build/sanitized --config Debug -- -j$(nproc)

# 2. 运行全量 ctest
cd build/sanitized
ctest --output-on-failure

# 3. 运行代表性 demo
bash scripts/run_demo.sh move all
bash scripts/run_demo.sh transformer all
```

### 测试用例

| # | 测试 | 预期 |
|---|------|------|
| 1 | ASan: 无内存错误 | 零报错 |
| 2 | UBSan: 无未定义行为 | 零报错 |
| 3 | move demo under sanitizers | 正常 |
| 4 | transformer demo under sanitizers | 正常 |

### 单步验证

```bash
cd build/sanitized
ASAN_OPTIONS=detect_leaks=1 ctest --output-on-failure
```

### 门控

- [x] AddressSanitizer 零报错
- [x] UndefinedBehaviorSanitizer 零报错
- [x] move + transformer demo 通过
- [ ] → **git commit** `"test(V3): pass AddressSanitizer and UBSan validation"`
- [ ] → 进入 V4

---

## 步骤 V4 — 静态分析验证

### 操作

```bash
# cppcheck
cppcheck --enable=all --inconclusive --std=c11 \
    -I src/ -I src/utils/ -I src/nn/ -I src/profiler/ \
    src/ demo/ 2>&1 | tee build/cppcheck_report.txt

# clang-analyzer (需 scan-build)
scan-build cmake -B build/analyze -DCMAKE_BUILD_TYPE=Debug
scan-build --status-bugs cmake --build build/analyze
```

### 测试用例

| # | 工具 | 预期 |
|---|------|------|
| 1 | cppcheck | 零新警告，原 156 个问题绝大部分已消除 |
| 2 | clang-analyzer | 零 bug 报告 |

### 门控

- [x] cppcheck 通过（残存警告不超过 20 个，均为误报或低优先级）
- [x] clang-analyzer 零 bug
- [ ] → **git commit** `"test(V4): pass static analysis (cppcheck + clang-analyzer)"`
- [ ] → 进入 V5

---

## 步骤 V5 — 更新文档

### 代码变更

**修改**: `web_editor/README.md` — 更新为 Rete 2.x 结构描述
**修改**: `README.md` — 更新构建命令（`cmake --preset debug` 替代旧命令）
**修改**: `AGENTS.md` (如存在) — 添加 Arena、VTable、CMakePresets 的使用规范

### 步骤测试

| # | 测试 | 操作 | 预期 |
|---|------|------|------|
| 1 | README 构建命令 | `cmake --preset debug && cmake --build build/debug` | 成功 |
| 2 | web_editor README | 按 README 步骤操作 | 可成功启动 |

### 门控

- [x] 文档已更新
- [ ] → **git commit** `"docs(V5): update documentation for new architecture"`
- [ ] → 阶段 4 完成 · 全部实施完成

---

# 总检查清单

## 阶段 0: 准备
- [ ] P0: 创建分支与目录骨架
- [ ] P0.5: 测试基础设施 (harness + CTest + smoke test)

## 轨 A（基础设施 + 内存）
- [ ] A1: ActionCError 枚举 + 测试 (3 用例)
- [ ] A2: 日志 + 断言宏 + 测试 (4 用例)
- [ ] A3: 安全整数运算 + 测试 (8 用例)
- [ ] A4: Arena Allocator + 测试 (7 用例)
- [ ] A5: 全工程错误码替换 + 回归
- [ ] A6: MLP Context 集成 Arena + 测试
- [ ] A7: MLP 后端 Arena 接入 + 测试 (2 用例)
- [ ] A8: GNN 后端 Arena 接入 + 测试 (2 用例)
- [ ] A9: Transformer + RNN Arena 接入 + 测试 (2 用例)

## 轨 B（接口 + 去重）
- [ ] B1: nn_backend.h VTable 接口 + 测试 (2 用例)
- [ ] B2: nn_weight_header.h + 测试 (5 用例)
- [ ] B3: 注册表重写为 VTable + 测试 (8 用例)
- [ ] B4: 删除旧桥接调度层 + 回归
- [ ] B5: MLP VTable 实现 + 测试 (4 用例)
- [ ] B6: GNN/Transformer/CNN/RNN VTable + 测试 (4 用例)
- [ ] B7: CNN + CNN_Dual_Pool 合并 + 回归
- [ ] B8: Transformer 前向合并 + 回归
- [ ] B9: MNIST 读取器合并 + 测试 (2 用例)
- [ ] B10: restrict + ActivationFn + safe_math + 测试 (3 用例)

## 轨 C（构建系统）
- [ ] C1: cmake/compiler_flags.cmake + 测试
- [ ] C2: cmake/version.cmake + 测试
- [ ] C3: src/ + demo/ 使用 cmake/ 模块 + 回归
- [ ] C4: CONFIGURE_DEPENDS + 测试
- [ ] C5: demo 构建依赖 + 测试
- [ ] C6: 路径计算修复 + 回归
- [ ] C7: CMakePresets.json + 测试
- [ ] C8: C_EXTENSIONS OFF 统一 + 回归
- [ ] C9: wasm CMakeLists.txt 修复 + 测试
- [ ] C10: build_wasm.sh 修复 + 测试
- [ ] C11: CS 工具路径修复 + 回归
- [ ] C12: build_demos.ps1 重写 + 测试
- [ ] C13: demo 脚本平台检测 + 回归

## 阶段 2: Profiler 流水线集成
- [ ] I1: FlatNetwork 类型定义 + 测试 (3 用例)
- [ ] I2: 验证函数接收 FlatNetwork + 测试 (2 用例)
- [ ] I3: 管道中间结果类型 + 测试 (2 用例)
- [ ] I4: profiler_generate_v2 管道化 + 测试 (1 E2E)
- [ ] I5: write_file 错误码 + 测试 (2 用例)
- [ ] I6: StringHashMap 实现 + 测试 (4 用例)

## 阶段 3: 前端与独立修复
- [ ] D1: Web Editor 遗留删除
- [ ] D2: Web Editor 安全加固
- [ ] D3: Web Editor 工程完善
- [ ] D4: 连接代码生成 + 测试
- [ ] D5: Wasm 运行时补全 + 测试
- [ ] D6: 独立小修复 (6 项)
- [ ] D7: CS JSON 加固 + 测试 (2 用例)
- [ ] D8: 脚本去重 + 测试

## 阶段 4: 验证
- [ ] V1: 删除死代码
- [ ] V2: 全量构建 + ctest + demo 回归
- [ ] V3: Sanitizer (ASan + UBSan)
- [ ] V4: 静态分析 (cppcheck + clang-analyzer)
- [ ] V5: 文档更新

---

## 统计

| 阶段 | 步骤 | 单元测试用例 | 集成/E2E 测试 |
|------|------|------------|-------------|
| 阶段 0 | 2 | 3 | 0 |
| 轨 A | 9 | 28 | 5 demo 回归 |
| 轨 B | 10 | 28 | 12 demo 回归 |
| 轨 C | 13 | 0 | 13 构建验证 |
| 阶段 2 | 6 | 14 | 10 demo generate 回归 |
| 阶段 3 | 8 | 8 | Wasm + Web 验证 |
| 阶段 4 | 5 | 0 | 10 demo 全流程 |
| **合计** | **53** | **~81** | **50+ 回归** |

**总测试用例**: ~168 个（含单元测试 + 集成/E2E + 回归验证）。
**消灭问题**: 原 156 个问题中预计消灭 ~140 个（余下为 cppcheck 误报或低优先级）。

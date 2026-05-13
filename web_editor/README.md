# Action NN-C Web 可视化编辑器

一个基于浏览器的神经网络可视化构建工具，支持拖拽式创建网络结构、配置属性，并生成 C 代码配置文件。

## 功能特性

### 🎨 可视化编辑
- **拖拽式操作**: 从左侧组件库拖拽节点到画布
- **连线连接**: 在节点之间建立数据流连接
- **实时预览**: 所见即所得的网络结构展示
- **平移缩放**: 支持画布的平移和缩放操作

### 🔧 丰富的组件库
- **输入层 (Input)**: 定义网络输入形状和数据类型
- **全连接层 (Dense)**: 标准的全连接神经网络层
- **卷积层 (Conv2D)**: 2D 卷积操作，支持多种参数配置
- **池化层 (Pooling)**: 最大/平均池化，全局池化
- **激活函数 (Activation)**: ReLU、Sigmoid、Tanh 等多种激活函数
- **Dropout**: 正则化层，防止过拟合
- **展平层 (Flatten)**: 将多维张量展平为一维
- **输出层 (Output)**: 定义网络输出和损失函数
- **Transformer Block**: 自注意力机制模块
- **LSTM/RNN**: 循环神经网络层
- **图卷积 (GNN)**: 图神经网络卷积层

### ⚙️ 属性配置
每个节点都支持详细的属性配置：
- 神经元数量、滤波器数量
- 激活函数选择
- 卷积核大小、步长、填充方式
- Dropout 比率
- 初始化方法
- 以及更多...

### 📤 导出功能
- **JSON 配置**: 导出完整的网络结构配置
- **C 头文件**: 自动生成适用于 action_nn_c 的配置头文件
- **导入配置**: 从 JSON 文件恢复之前的工作

## 要求

- **Node.js** >= 18.0.0
- **npm** >= 9.0.0

## 快速开始

```bash
# 安装依赖
cd web_editor
npm install

# 开发模式 (热更新)
npm run dev

# 生产构建
npm run build

# 预览生产构建
npm run preview
```

浏览器访问 `http://127.0.0.1:5173` (Vite 默认端口)

### 部署到生产环境

```bash
npm run build
# 部署 dist/ 目录到任意静态文件服务器
```

## 使用说明

### 1. 创建网络结构

1. **选择网络类型**: 在左侧面板选择网络类型（MLP、CNN、Transformer 等）
2. **拖拽组件**: 从组件库拖拽需要的层到中间画布
3. **排列顺序**: 按照数据流方向排列各层（从上到下或从左到右）
4. **连接节点**: 从一个节点的输出点拖拽到下一个节点的输入点

### 2. 配置属性

1. **点击节点**: 在画布上点击要编辑的节点
2. **修改参数**: 在右侧属性面板修改各项参数
3. **实时保存**: 修改会自动保存到节点数据中

### 3. 导出配置

#### 导出 JSON 配置
1. 点击"导出配置"按钮
2. 查看生成的 JSON 数据
3. 可以下载文件或复制到剪贴板

#### 生成 C 代码
1. 确保网络包含输入层和输出层
2. 点击"生成 C 代码"按钮
3. 复制生成的头文件代码
4. 将代码保存到 `config/` 目录

### 4. 导入配置

1. 点击"导入配置"按钮
2. 选择之前导出的 JSON 文件
3. 网络结构会自动加载到画布

## 示例：创建一个简单的 MLP

1. 设置网络类型为 "MLP"
2. 拖拽一个 **输入层** 到画布
   - 配置输入形状：`784` (对于 MNIST)
   - 批次大小：`32`
3. 拖拽一个 **全连接层**
   - 神经元数量：`128`
   - 激活函数：`relu`
4. 拖拽另一个 **全连接层**
   - 神经元数量：`64`
   - 激活函数：`relu`
5. 拖拽一个 **输出层**
   - 激活函数：`softmax`
   - 损失函数：`categorical_crossentropy`
6. 依次连接各层
7. 点击"生成 C 代码"获取配置文件

## 项目结构

```
web_editor/
├── index.html          # 主页面 (含 CSP 安全策略)
├── package.json        # Vite + Rete.js 2.x 依赖
├── vite.config.js      # Vite 配置 (host: 127.0.0.1)
├── README.md           # 本文件
├── src/
│   └── main.js         # 编辑器主逻辑
└── dist/               # 构建产物 (npm run build)
```

## 技术栈

- **Rete.js 2.x**: 基于节点的可视化框架 (npm 依赖)
- **Vite 6**: 开发服务器与构建工具
- **Vue 3**: UI 组件 (rete-vue-plugin)
- **Vanilla JavaScript**: 编辑器业务逻辑

## 与 action_nn_c 集成

生成的 C 代码可以直接用于 action_nn_c 项目：

```c
// 将生成的代码保存为 config/my_network_config.h
#include "my_network_config.h"

// 在训练/推理代码中使用
#include "action_nn_c.h"

int main() {
    // 使用生成的配置初始化网络
    network_t net;
    network_init(&net, NETWORK_TYPE);
    
    // ... 继续训练或推理
    return 0;
}
```

## 注意事项

1. **浏览器兼容性**: 推荐使用 Chrome、Firefox、Edge 等现代浏览器
2. **节点连接**: 确保输入层在最前，输出层在最后，中间层正确连接
3. **参数验证**: 某些参数有取值范围限制，请注意提示
4. **保存工作**: 定期导出 JSON 配置以防丢失
5. **依赖安装**: 必须先运行 `npm install`，所有依赖通过 npm 管理
6. **安全**: 页面含有 CSP (Content Security Policy)，仅绑定 127.0.0.1

## 开发调试

打开浏览器开发者工具（F12），可以：
- 查看控制台日志
- 检查网络结构数据：`window.networkEditor.nodes`
- 手动调用编辑器方法

```javascript
// 在控制台中可以执行
window.networkEditor.addNode('dense', {x: 200, y: 300});
window.networkEditor.exportConfig();
```

## 离线使用

如需离线使用，可以：
1. 下载 Rete.js 库文件到 `lib/` 目录
2. 修改 `index.html` 中的 script 标签路径为本地文件

```html
<script src="lib/rete.min.js"></script>
<script src="lib/connection-plugin.min.js"></script>
<script src="lib/area-plugin.min.js"></script>
<script src="lib/render-plugin.min.js"></script>
```

## 未来计划

- [ ] 支持更多网络层类型
- [ ] 添加网络结构验证
- [ ] 实现训练过程可视化
- [ ] 支持模型性能预估
- [ ] 添加预训练模型模板
- [ ] 支持导出 ONNX 格式
- [ ] 添加批量归一化层 (BatchNorm)
- [ ] 添加残差连接 (Residual Connection)

## 许可证

MIT License

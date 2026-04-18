/**
 * Action NN-C Web Editor - Main Editor Logic
 * 使用 Rete.js 1.x 实现可视化神经网络编辑器
 */

class NetworkEditor {
    constructor() {
        this.editor = null;
        this.nodes = [];
        this.connections = [];
        this.selectedNode = null;
        this.nodeIdCounter = 0;
        
        this.init();
    }

    async init() {
        try {
            // 等待 Rete.js 库加载
            await this.waitForRete();
            
            // 创建编辑器实例
            this.createEditor();
            
            // 设置拖拽功能
            this.setupDragDrop();
            
            // 绑定按钮事件
            this.setupButtonEvents();
            
            // 设置模态框事件
            this.setupModalEvents();
            
            console.log('Editor initialized successfully');
        } catch (error) {
            console.error('Failed to initialize editor:', error);
            alert('编辑器初始化失败，请检查控制台日志');
        }
    }

    waitForRete() {
        return new Promise((resolve, reject) => {
            const checkInterval = setInterval(() => {
                if (window.Rete && window.Rete.Node) {
                    clearInterval(checkInterval);
                    resolve();
                }
            }, 100);
            
            // 超时处理
            setTimeout(() => {
                clearInterval(checkInterval);
                reject(new Error('Rete.js library not loaded'));
            }, 5000);
        });
    }

    createEditor() {
        const container = document.getElementById('editor');
        
        // 使用 Rete.js 1.x API - ID 必须是 name@version 格式
        this.editor = new Rete.Engine('demo@1.0.0');
        
        // 注册插件
        const render = new Rete.RenderPlugin();
        const connectionPlugin = new Rete.ConnectionPlugin();
        const area = new Rete.AreaPlugin();
        
        this.editor.use(render);
        this.editor.use(connectionPlugin);
        this.editor.use(area);
        
        // 配置区域插件
        area.zoomAt(0.8);
        
        // 绑定到容器
        container.appendChild(this.editor.root);
        
        console.log('Rete.js 1.x editor created');
    }

    setupDragDrop() {
        const componentItems = document.querySelectorAll('.component-item');
        const editorCanvas = document.getElementById('editor');
        
        let draggedType = null;
        let dragOverEditor = false;
        
        componentItems.forEach(item => {
            item.addEventListener('dragstart', (e) => {
                draggedType = item.dataset.type;
                e.dataTransfer.setData('type', draggedType);
                e.dataTransfer.effectAllowed = 'copy';
                item.style.opacity = '0.5';
            });
            
            item.addEventListener('dragend', (e) => {
                draggedType = null;
                item.style.opacity = '1';
            });
        });
        
        editorCanvas.addEventListener('dragover', (e) => {
            e.preventDefault();
            dragOverEditor = true;
            e.dataTransfer.dropEffect = 'copy';
        });
        
        editorCanvas.addEventListener('dragleave', (e) => {
            dragOverEditor = false;
        });
        
        editorCanvas.addEventListener('drop', (e) => {
            e.preventDefault();
            if (draggedType) {
                const rect = editorCanvas.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                
                this.addNode(draggedType, { x, y });
                draggedType = null;
            }
        });
    }

    addNode(type, position) {
        if (!this.editor) {
            console.error('Editor not initialized');
            return;
        }
        
        const nodeId = `node_${this.nodeIdCounter++}`;
        const nodeConfig = window.NEURAL_NETWORK_NODES[type];
        
        if (!nodeConfig) {
            console.error(`Unknown node type: ${type}`);
            return;
        }
        
        // 创建 Rete.js 1.x 节点
        const node = new Rete.Node(nodeId);
        node.label = nodeConfig.label;
        node.position = position;
        
        // 添加属性数据
        node.data.properties = {};
        nodeConfig.properties.forEach(prop => {
            node.data.properties[prop.name] = prop.default;
        });
        
        // 创建输入插槽
        if (nodeConfig.inputs && nodeConfig.inputs.length > 0) {
            nodeConfig.inputs.forEach(inputName => {
                const input = new Rete.Socket('Input socket');
                node.addInput(inputName, input);
            });
        }
        
        // 创建输出插槽
        if (nodeConfig.outputs && nodeConfig.outputs.length > 0) {
            nodeConfig.outputs.forEach(outputName => {
                const output = new Rete.Socket('Output socket');
                node.addOutput(outputName, output);
            });
        }
        
        // 添加节点到编辑器
        this.editor.addNode(node);
        this.nodes.push(node);
        
        // 自动选择新添加的节点
        setTimeout(() => {
            this.onNodeSelected(node);
        }, 100);
    }

    onNodeSelected(node) {
        this.selectedNode = node;
        this.updatePropertiesPanel(node);
    }

    updatePropertiesPanel(node) {
        const propertiesContent = document.getElementById('properties-content');
        
        if (!node || !node.data) {
            this.clearPropertiesPanel();
            return;
        }
        
        const nodeType = node.data.type;
        const nodeConfig = window.NEURAL_NETWORK_NODES[nodeType];
        
        if (!nodeConfig) {
            propertiesContent.innerHTML = '<p class="hint">未知节点类型</p>';
            return;
        }
        
        let html = `<div class="property-form">`;
        html += `<h3>${nodeConfig.label}</h3>`;
        html += `<p style="color: #888; font-size: 12px; margin-bottom: 15px;">ID: ${node.id}</p>`;
        
        // 属性表单
        nodeConfig.properties.forEach(prop => {
            const value = node.data.properties[prop.name] !== undefined 
                ? node.data.properties[prop.name] 
                : prop.default;
            
            html += `<div class="property-group">`;
            html += `<label for="${prop.name}">${prop.label}</label>`;
            
            if (prop.type === 'select') {
                html += `<select id="${prop.name}" data-prop="${prop.name}" class="prop-input">`;
                prop.options.forEach(opt => {
                    const selected = String(value) === String(opt) ? 'selected' : '';
                    html += `<option value="${opt}" ${selected}>${opt}</option>`;
                });
                html += `</select>`;
            } else if (prop.type === 'checkbox') {
                const checked = value ? 'checked' : '';
                html += `<input type="checkbox" id="${prop.name}" data-prop="${prop.name}" class="prop-input" ${checked} />`;
            } else if (prop.type === 'number') {
                const min = prop.min !== undefined ? `min="${prop.min}"` : '';
                const max = prop.max !== undefined ? `max="${prop.max}"` : '';
                const step = prop.step !== undefined ? `step="${prop.step}"` : '';
                html += `<input type="number" id="${prop.name}" data-prop="${prop.name}" class="prop-input" value="${value}" ${min} ${max} ${step} />`;
            } else {
                const placeholder = prop.placeholder ? `placeholder="${prop.placeholder}"` : '';
                html += `<input type="text" id="${prop.name}" data-prop="${prop.name}" class="prop-input" value="${value}" ${placeholder} />`;
            }
            
            if (prop.description) {
                html += `<small>${prop.description}</small>`;
            }
            
            html += `</div>`;
        });
        
        // 删除按钮
        html += `<div style="margin-top: 20px; padding-top: 15px; border-top: 1px solid #3c3c3c;">`;
        html += `<button id="delete-node-btn" class="btn btn-danger" style="width: 100%;">删除节点</button>`;
        html += `</div>`;
        
        html += `</div>`;
        
        propertiesContent.innerHTML = html;
        
        // 绑定属性变更事件
        this.bindPropertyChangeEvents(node);
        
        // 绑定删除按钮事件
        const deleteBtn = document.getElementById('delete-node-btn');
        if (deleteBtn) {
            deleteBtn.addEventListener('click', () => {
                this.deleteNode(node.id);
            });
        }
    }

    bindPropertyChangeEvents(node) {
        const inputs = document.querySelectorAll('.prop-input');
        
        inputs.forEach(input => {
            const propName = input.dataset.prop;
            
            if (input.type === 'checkbox') {
                input.addEventListener('change', (e) => {
                    this.updateNodeProperty(node, propName, e.target.checked);
                });
            } else {
                input.addEventListener('input', (e) => {
                    this.updateNodeProperty(node, propName, e.target.value);
                });
            }
        });
    }

    updateNodeProperty(node, propName, value) {
        if (!node || !node.data || !node.data.properties) {
            return;
        }
        
        node.data.properties[propName] = value;
        
        // 触发更新事件（如果需要实时预览）
        console.log(`Updated property ${propName} = ${value} for node ${node.id}`);
    }

    deleteNode(nodeId) {
        if (!this.editor) return;
        
        this.editor.removeNode(nodeId);
        this.clearPropertiesPanel();
    }

    clearPropertiesPanel() {
        const propertiesContent = document.getElementById('properties-content');
        propertiesContent.innerHTML = '<p class="hint">点击画布中的节点以编辑属性</p>';
        this.selectedNode = null;
    }

    setupButtonEvents() {
        // 导出配置
        document.getElementById('export-btn').addEventListener('click', () => {
            this.exportConfig();
        });
        
        // 导入配置
        document.getElementById('import-btn').addEventListener('click', () => {
            this.importConfig();
        });
        
        // 清空画布
        document.getElementById('clear-btn').addEventListener('click', () => {
            this.clearCanvas();
        });
        
        // 生成 C 代码
        document.getElementById('generate-code-btn').addEventListener('click', () => {
            this.generateCCode();
        });
    }

    setupModalEvents() {
        // 关闭模态框
        const closeButtons = document.querySelectorAll('.close');
        closeButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                btn.closest('.modal').style.display = 'none';
            });
        });
        
        // 点击模态框外部关闭
        window.addEventListener('click', (e) => {
            if (e.target.classList.contains('modal')) {
                e.target.style.display = 'none';
            }
        });
        
        // 下载 JSON
        document.getElementById('download-json-btn').addEventListener('click', () => {
            this.downloadJSON();
        });
        
        // 复制 JSON
        document.getElementById('copy-json-btn').addEventListener('click', () => {
            this.copyToClipboard('export-output');
        });
        
        // 复制代码
        document.getElementById('copy-code-btn').addEventListener('click', () => {
            this.copyToClipboard('generated-code');
        });
    }

    exportConfig() {
        const config = {
            networkName: document.getElementById('network-name').value,
            networkType: document.getElementById('network-type').value,
            nodes: this.nodes.map(node => ({
                id: node.id,
                type: node.data.type,
                position: node.position,
                properties: node.data.properties
            })),
            connections: this.connections.map(conn => ({
                source: conn.source,
                sourceOutput: conn.sourceOutput,
                target: conn.target,
                targetInput: conn.targetInput
            }))
        };
        
        const jsonStr = JSON.stringify(config, null, 2);
        document.getElementById('export-output').value = jsonStr;
        document.getElementById('export-modal').style.display = 'block';
    }

    importConfig() {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = '.json';
        
        input.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (!file) return;
            
            const reader = new FileReader();
            reader.onload = (event) => {
                try {
                    const config = JSON.parse(event.target.result);
                    this.loadConfig(config);
                } catch (error) {
                    alert('配置文件格式错误：' + error.message);
                }
            };
            reader.readAsText(file);
        });
        
        input.click();
    }

    loadConfig(config) {
        if (!this.editor) return;
        
        // 清空当前画布
        this.clearCanvas();
        
        // 设置全局配置
        if (config.networkName) {
            document.getElementById('network-name').value = config.networkName;
        }
        if (config.networkType) {
            document.getElementById('network-type').value = config.networkType;
        }
        
        // 添加节点
        if (config.nodes) {
            config.nodes.forEach(nodeData => {
                const nodeConfig = window.NEURAL_NETWORK_NODES[nodeData.type];
                if (!nodeConfig) return;
                
                const node = new Rete.Node(nodeData.id);
                node.label = nodeConfig.label;
                node.position = nodeData.position;
                node.data.properties = nodeData.properties || {};
                
                // 创建输入插槽
                if (nodeConfig.inputs && nodeConfig.inputs.length > 0) {
                    nodeConfig.inputs.forEach(inputName => {
                        const input = new Rete.Socket('Input socket');
                        node.addInput(inputName, input);
                    });
                }
                
                // 创建输出插槽
                if (nodeConfig.outputs && nodeConfig.outputs.length > 0) {
                    nodeConfig.outputs.forEach(outputName => {
                        const output = new Rete.Socket('Output socket');
                        node.addOutput(outputName, output);
                    });
                }
                
                this.editor.addNode(node);
                this.nodes.push(node);
            });
        }
        
        console.log('Config loaded successfully');
    }

    clearCanvas() {
        if (!this.editor) return;
        
        // 删除所有节点
        const nodeIds = this.nodes.map(n => n.id);
        nodeIds.forEach(id => {
            this.editor.removeNode(id);
        });
        
        this.nodes = [];
        this.connections = [];
        this.clearPropertiesPanel();
    }

    generateCCode() {
        const networkName = document.getElementById('network-name').value;
        const networkType = document.getElementById('network-type').value;
        
        // 验证网络结构
        if (this.nodes.length === 0) {
            alert('请先添加至少一个节点');
            return;
        }
        
        // 查找输入层和输出层
        const inputNodes = this.nodes.filter(n => n.data.type === 'input');
        const outputNodes = this.nodes.filter(n => n.data.type === 'output');
        
        if (inputNodes.length === 0) {
            alert('请添加一个输入层节点');
            return;
        }
        
        if (outputNodes.length === 0) {
            alert('请添加一个输出层节点');
            return;
        }
        
        // 生成 C 头文件代码
        let code = `/**\n`;
        code += ` * Auto-generated network configuration for Action NN-C\n`;
        code += ` * Network Name: ${networkName}\n`;
        code += ` * Network Type: ${networkType}\n`;
        code += ` * Generated by: Action NN-C Web Editor\n`;
        code += ` */\n\n`;
        
        code += `#ifndef __${networkName.toUpperCase()}_CONFIG_H__\n`;
        code += `#define __${networkName.toUpperCase()}_CONFIG_H__\n\n`;
        
        code += `#include "action_nn_c.h"\n\n`;
        
        // 生成网络配置结构
        code += `// Network Configuration\n`;
        code += `#define NETWORK_NAME "${networkName}"\n`;
        code += `#define NETWORK_TYPE NET_TYPE_${networkType.toUpperCase()}\n\n`;
        
        // 生成层配置
        code += `// Layer Definitions\n`;
        this.nodes.forEach((node, index) => {
            const nodeType = node.data.type;
            const props = node.data.properties;
            const layerName = `${nodeType}_${index}`.toUpperCase();
            
            code += `// Layer ${index}: ${window.NEURAL_NETWORK_NODES[nodeType].label}\n`;
            
            if (nodeType === 'dense') {
                code += `#define ${layerName}_UNITS ${props.units}\n`;
                code += `#define ${layerName}_ACTIVATION ACT_${props.activation.toUpperCase()}\n`;
                code += `#define ${layerName}_USE_BIAS ${props.use_bias ? 1 : 0}\n`;
            } else if (nodeType === 'conv2d') {
                code += `#define ${layerName}_FILTERS ${props.filters}\n`;
                code += `#define ${layerName}_KERNEL_SIZE ${props.kernel_size.replace(/,/g, ', ')}\n`;
                code += `#define ${layerName}_STRIDES ${props.strides.replace(/,/g, ', ')}\n`;
                code += `#define ${layerName}_PADDING PAD_${props.padding.toUpperCase()}\n`;
            } else if (nodeType === 'input') {
                code += `#define ${layerName}_SHAPE ${props.input_shape.replace(/,/g, ', ')}\n`;
                code += `#define ${layerName}_BATCH_SIZE ${props.batch_size}\n`;
            }
            
            code += `\n`;
        });
        
        // 生成网络拓扑
        code += `// Network Topology\n`;
        code += `#define NETWORK_LAYER_COUNT ${this.nodes.length}\n\n`;
        
        code += `#endif // __${networkName.toUpperCase()}_CONFIG_H__\n`;
        
        document.getElementById('generated-code').textContent = code;
        document.getElementById('code-modal').style.display = 'block';
    }

    downloadJSON() {
        const jsonStr = document.getElementById('export-output').value;
        const blob = new Blob([jsonStr], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `${document.getElementById('network-name').value}_config.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    copyToClipboard(elementId) {
        const element = document.getElementById(elementId);
        element.select();
        document.execCommand('copy');
        
        // 显示提示
        const btn = elementId === 'export-output' ? 'copy-json-btn' : 'copy-code-btn';
        const originalText = document.getElementById(btn).textContent;
        document.getElementById(btn).textContent = '已复制!';
        setTimeout(() => {
            document.getElementById(btn).textContent = originalText;
        }, 2000);
    }
}

// 初始化编辑器
let editorInstance = null;

document.addEventListener('DOMContentLoaded', () => {
    editorInstance = new NetworkEditor();
    
    // 暴露给全局以便调试
    window.networkEditor = editorInstance;
});

console.log('Network Editor script loaded');

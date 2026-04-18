/**
 * Action NN-C Web Editor - Node Definitions
 * 定义所有可用的神经网络层节点类型
 */

// 节点类型配置映射
const NODE_TYPES = {
    input: {
        label: '输入层',
        icon: '📥',
        color: '#4ec9b0',
        inputs: [],
        outputs: ['output'],
        properties: [
            { name: 'input_shape', label: '输入形状', type: 'text', default: '28,28,1', placeholder: '例如：28,28,1 或 784' },
            { name: 'batch_size', label: '批次大小', type: 'number', default: '32', min: 1 },
            { name: 'dtype', label: '数据类型', type: 'select', default: 'float32', options: ['float32', 'float16', 'int8'] }
        ]
    },
    dense: {
        label: '全连接层',
        icon: '🔵',
        color: '#569cd6',
        inputs: ['input'],
        outputs: ['output'],
        properties: [
            { name: 'units', label: '神经元数量', type: 'number', default: '128', min: 1 },
            { name: 'activation', label: '激活函数', type: 'select', default: 'relu', options: ['relu', 'sigmoid', 'tanh', 'softmax', 'linear', 'leaky_relu'] },
            { name: 'use_bias', label: '使用偏置', type: 'checkbox', default: true },
            { name: 'kernel_initializer', label: '权重初始化', type: 'select', default: 'glorot_uniform', options: ['glorot_uniform', 'he_normal', 'random_normal', 'zeros'] },
            { name: 'dropout_rate', label: 'Dropout 比率', type: 'number', default: '0.0', min: 0, max: 1, step: 0.1 }
        ]
    },
    conv2d: {
        label: '卷积层',
        icon: '🟦',
        color: '#ce9178',
        inputs: ['input'],
        outputs: ['output'],
        properties: [
            { name: 'filters', label: '滤波器数量', type: 'number', default: '32', min: 1 },
            { name: 'kernel_size', label: '卷积核大小', type: 'text', default: '3,3', placeholder: '例如：3,3 或 5,5' },
            { name: 'strides', label: '步长', type: 'text', default: '1,1', placeholder: '例如：1,1 或 2,2' },
            { name: 'padding', label: '填充方式', type: 'select', default: 'same', options: ['same', 'valid'] },
            { name: 'activation', label: '激活函数', type: 'select', default: 'relu', options: ['relu', 'sigmoid', 'tanh', 'linear', 'leaky_relu'] },
            { name: 'use_bias', label: '使用偏置', type: 'checkbox', default: true },
            { name: 'dilation_rate', label: '膨胀率', type: 'text', default: '1,1', placeholder: '例如：1,1 或 2,2' }
        ]
    },
    pooling: {
        label: '池化层',
        icon: '🔽',
        color: '#dcdcaa',
        inputs: ['input'],
        outputs: ['output'],
        properties: [
            { name: 'pool_type', label: '池化类型', type: 'select', default: 'max', options: ['max', 'average', 'global_max', 'global_average'] },
            { name: 'pool_size', label: '池化窗口大小', type: 'text', default: '2,2', placeholder: '例如：2,2 或 3,3' },
            { name: 'strides', label: '步长', type: 'text', default: '2,2', placeholder: '例如：2,2 或留空' },
            { name: 'padding', label: '填充方式', type: 'select', default: 'valid', options: ['same', 'valid'] }
        ]
    },
    activation: {
        label: '激活函数',
        icon: '⚡',
        color: '#c586c0',
        inputs: ['input'],
        outputs: ['output'],
        properties: [
            { name: 'activation_type', label: '激活类型', type: 'select', default: 'relu', options: ['relu', 'sigmoid', 'tanh', 'softmax', 'leaky_relu', 'elu', 'selu', 'swish'] },
            { name: 'alpha', label: 'Alpha 参数', type: 'number', default: '0.0', step: 0.1, description: '用于 leaky_relu, elu 等' }
        ]
    },
    dropout: {
        label: 'Dropout',
        icon: '🚫',
        color: '#f44747',
        inputs: ['input'],
        outputs: ['output'],
        properties: [
            { name: 'rate', label: 'Dropout 比率', type: 'number', default: '0.5', min: 0, max: 1, step: 0.1 },
            { name: 'noise_shape', label: '噪声形状', type: 'text', default: '', placeholder: '可选，留空表示与输入相同' },
            { name: 'training_only', label: '仅训练时使用', type: 'checkbox', default: true }
        ]
    },
    flatten: {
        label: '展平层',
        icon: '➖',
        color: '#9cdcfe',
        inputs: ['input'],
        outputs: ['output'],
        properties: [
            { name: 'start_dim', label: '起始维度', type: 'number', default: '1', min: 0 },
            { name: 'end_dim', label: '结束维度', type: 'number', default: '-1', min: -1, description: '-1 表示最后一个维度' }
        ]
    },
    output: {
        label: '输出层',
        icon: '📤',
        color: '#4ec9b0',
        inputs: ['input'],
        outputs: [],
        properties: [
            { name: 'output_name', label: '输出名称', type: 'text', default: 'output' },
            { name: 'activation', label: '激活函数', type: 'select', default: 'linear', options: ['linear', 'softmax', 'sigmoid', 'tanh'] },
            { name: 'loss_function', label: '损失函数', type: 'select', default: 'mse', options: ['mse', 'categorical_crossentropy', 'binary_crossentropy', 'sparse_categorical_crossentropy'] }
        ]
    },
    transformer_block: {
        label: 'Transformer Block',
        icon: '🔄',
        color: '#b180d7',
        inputs: ['input'],
        outputs: ['output'],
        properties: [
            { name: 'num_heads', label: '注意力头数', type: 'number', default: '8', min: 1 },
            { name: 'ff_dim', label: '前馈网络维度', type: 'number', default: '2048', min: 1 },
            { name: 'dropout_rate', label: 'Dropout 比率', type: 'number', default: '0.1', min: 0, max: 1, step: 0.1 },
            { name: 'attention_dropout', label: '注意力 Dropout', type: 'number', default: '0.1', min: 0, max: 1, step: 0.1 },
            { name: 'layer_norm_eps', label: 'LayerNorm Epsilon', type: 'number', default: '1e-6', step: 0.0000001 },
            { name: 'use_mask', label: '使用掩码', type: 'checkbox', default: false }
        ]
    },
    rnn_lstm: {
        label: 'LSTM/RNN',
        icon: '🔁',
        color: '#dc9656',
        inputs: ['input'],
        outputs: ['output'],
        properties: [
            { name: 'rnn_type', label: 'RNN 类型', type: 'select', default: 'lstm', options: ['lstm', 'gru', 'simple_rnn'] },
            { name: 'units', label: '隐藏单元数', type: 'number', default: '128', min: 1 },
            { name: 'return_sequences', label: '返回序列', type: 'checkbox', default: false },
            { name: 'return_state', label: '返回状态', type: 'checkbox', default: false },
            { name: 'dropout_rate', label: 'Dropout 比率', type: 'number', default: '0.0', min: 0, max: 1, step: 0.1 },
            { name: 'recurrent_dropout', label: '循环 Dropout', type: 'number', default: '0.0', min: 0, max: 1, step: 0.1 },
            { name: 'activation', label: '激活函数', type: 'select', default: 'tanh', options: ['tanh', 'relu', 'sigmoid'] }
        ]
    },
    gnn_conv: {
        label: '图卷积 (GNN)',
        icon: '🕸️',
        color: '#6a9955',
        inputs: ['input', 'adjacency'],
        outputs: ['output'],
        properties: [
            { name: 'conv_type', label: '卷积类型', type: 'select', default: 'gcn', options: ['gcn', 'gat', 'graph_sage', 'gin'] },
            { name: 'units', label: '输出维度', type: 'number', default: '64', min: 1 },
            { name: 'num_heads', label: '注意力头数 (GAT)', type: 'number', default: '4', min: 1, description: '仅 GAT 类型使用' },
            { name: 'activation', label: '激活函数', type: 'select', default: 'relu', options: ['relu', 'sigmoid', 'tanh', 'linear'] },
            { name: 'use_bias', label: '使用偏置', type: 'checkbox', default: true },
            { name: 'aggregation', label: '聚合方式', type: 'select', default: 'mean', options: ['mean', 'sum', 'max', 'attention'] }
        ]
    }
};

// 生成 Rete.js 节点类
class NeuralNetworkNode {
    constructor(type, data) {
        this.type = type;
        this.data = data || {};
        this.config = NODE_TYPES[type];
    }

    // 创建节点定义
    createNode(id, position = { x: 100, y: 100 }) {
        const node = {
            id: id,
            label: this.config.label,
            position: position,
            data: {
                type: this.type,
                properties: {}
            },
            inputs: {},
            outputs: {}
        };

        // 添加默认属性值
        this.config.properties.forEach(prop => {
            node.data.properties[prop.name] = prop.default;
        });

        // 创建输入插槽
        if (this.config.inputs && this.config.inputs.length > 0) {
            this.config.inputs.forEach((inputName, index) => {
                node.inputs[inputName] = {
                    id: `${id}-input-${inputName}`,
                    label: inputName,
                    type: 'tensor'
                };
            });
        } else {
            // 对于输入层，创建一个通用的 input 插槽
            node.inputs['data'] = {
                id: `${id}-input-data`,
                label: 'data',
                type: 'tensor'
            };
        }

        // 创建输出插槽
        if (this.config.outputs && this.config.outputs.length > 0) {
            this.config.outputs.forEach((outputName, index) => {
                node.outputs[outputName] = {
                    id: `${id}-output-${outputName}`,
                    label: outputName,
                    type: 'tensor'
                };
            });
        }

        return node;
    }

    // 获取属性表单 HTML
    getPropertiesForm(nodeData) {
        let html = '<div class="property-form">';
        
        this.config.properties.forEach(prop => {
            const value = nodeData.properties[prop.name] !== undefined 
                ? nodeData.properties[prop.name] 
                : prop.default;
            
            html += `<div class="property-group">`;
            html += `<label for="${prop.name}">${prop.label}</label>`;
            
            if (prop.type === 'select') {
                html += `<select id="${prop.name}" data-prop="${prop.name}">`;
                prop.options.forEach(opt => {
                    const selected = value === opt ? 'selected' : '';
                    html += `<option value="${opt}" ${selected}>${opt}</option>`;
                });
                html += `</select>`;
            } else if (prop.type === 'checkbox') {
                const checked = value ? 'checked' : '';
                html += `<input type="checkbox" id="${prop.name}" data-prop="${prop.name}" ${checked} />`;
            } else if (prop.type === 'number') {
                const min = prop.min !== undefined ? `min="${prop.min}"` : '';
                const max = prop.max !== undefined ? `max="${prop.max}"` : '';
                const step = prop.step !== undefined ? `step="${prop.step}"` : '';
                html += `<input type="number" id="${prop.name}" data-prop="${prop.name}" value="${value}" ${min} ${max} ${step} />`;
            } else {
                const placeholder = prop.placeholder ? `placeholder="${prop.placeholder}"` : '';
                html += `<input type="text" id="${prop.name}" data-prop="${prop.name}" value="${value}" ${placeholder} />`;
            }
            
            if (prop.description) {
                html += `<small>${prop.description}</small>`;
            }
            
            html += `</div>`;
        });
        
        html += '</div>';
        return html;
    }
}

// 导出节点类型配置和工厂函数
window.NEURAL_NETWORK_NODES = NODE_TYPES;
window.NeuralNetworkNode = NeuralNetworkNode;

// 辅助函数：根据类型创建节点
window.createNodeByType = function(type, id, position) {
    if (!NODE_TYPES[type]) {
        console.error(`Unknown node type: ${type}`);
        return null;
    }
    const nodeFactory = new NeuralNetworkNode(type);
    return nodeFactory.createNode(id, position);
};

// 辅助函数：获取节点属性表单
window.getNodePropertiesForm = function(type, nodeData) {
    if (!NODE_TYPES[type]) {
        return '<p class="hint">未知节点类型</p>';
    }
    const nodeFactory = new NeuralNetworkNode(type);
    return nodeFactory.getPropertiesForm(nodeData);
};

console.log('Neural Network Node definitions loaded:', Object.keys(NODE_TYPES).length, 'node types');

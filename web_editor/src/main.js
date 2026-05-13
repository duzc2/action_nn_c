/**
 * Action NN-C Web Editor - Visual Network Editor for Profiler
 * 结合 Rete.js 2.x 图形化编辑 + Profiler 接口输出
 */

import { NodeEditor, ClassicPreset } from 'rete'
import { AreaPlugin, AreaExtensions } from 'rete-area-plugin'
import { ConnectionPlugin, Presets as ConnectionPresets } from 'rete-connection-plugin'
import { VuePlugin, Presets as VuePresets } from 'rete-vue-plugin'

// 定义类型
const { Socket, Input, Output, InputControl, Node, Connection } = ClassicPreset

// 创建 Socket
const socket = new Socket('socket')

// 激活函数选项
const ACTIVATIONS = ['relu', 'sigmoid', 'tanh', 'softmax', 'linear', 'leaky_relu']

// 编辑器实例
let editor, area, connection

/**
 * 从节点获取数据
 */
function getNodeData(node) {
    const data = {
        type: node.meta?.type || 'mlp',
        input_size: 10,
        hidden_layers: 0,
        hidden_size: 64,
        output_size: 10,
        activation: 'relu'
    }
    
    // 从 controls 读取值
    if (node.controls) {
        Object.entries(node.controls).forEach(([key, control]) => {
            if (control && control.value !== undefined) {
                if (key === 'input_size') data.input_size = control.value
                else if (key === 'hidden_layers') data.hidden_layers = control.value
                else if (key === 'hidden_size') data.hidden_size = control.value
                else if (key === 'output_size') data.output_size = control.value
                else if (key === 'activation') data.activation = control.value
            }
        })
    }
    
    return data
}

/**
 * 创建子网节点
 */
function createSubnetNode(label, type, config = {}) {
    const node = new Node(label)
    node.meta = { type }
    
    // 输入端口
    node.addInput('input', new Input(socket, 'Input', true))
    
    // 输出端口
    node.addOutput('output', new Output(socket, 'Output'))
    
    // 控制参数
    node.addControl('input_size', new InputControl('number', { 
        initial: config.input_size || 10
    }))
    
    if (type === 'mlp') {
        node.addControl('hidden_layers', new InputControl('number', { 
            initial: config.hidden_layers || 0
        }))
        node.addControl('hidden_size', new InputControl('number', { 
            initial: config.hidden_size || 64
        }))
        node.addControl('output_size', new InputControl('number', { 
            initial: config.output_size || 10
        }))
        node.addControl('activation', new InputControl('select', {
            initial: config.activation || 'relu',
            options: ACTIVATIONS
        }))
    }
    
    return node
}

/**
 * 初始化编辑器
 */
async function initEditor() {
    const container = document.getElementById('editor')
    
    // 创建编辑器
    editor = new NodeEditor('nnc@1.0.0')
    
    // 创建区域插件
    area = new AreaPlugin(container)
    
    // 创建连接插件
    connection = new ConnectionPlugin()
    connection.addPreset(ConnectionPresets.classic.setup())
    
    // 创建 Vue 渲染插件
    const render = new VuePlugin()
    render.addPreset(VuePresets.classic.setup())
    
    // 使用插件
    editor.use(area)
    area.use(connection)
    area.use(render)
    
    console.log('Editor initialized')
}

/**
 * 添加示例网络
 */
async function addExampleNetwork() {
    // 如果已经有节点，跳过
    if (editor.nodes.length > 0) {
        console.log('Network already exists, skipping example')
        return
    }
    
    // 输入层
    const input = createSubnetNode('Input', 'mlp', {
        input_size: 10,
        output_size: 64,
        hidden_layers: 0,
        activation: 'linear'
    })
    input.position = [50, 200]
    await editor.addNode(input)
    area.update('node', input)
    // 直接操作 DOM 确保位置更新 - 位置在祖父元素的 transform: translate(X,Y) 上
    await new Promise(r => setTimeout(r, 50))
    const inputEl = document.querySelector(`[data-id="${input.id}"]`)
    if (inputEl) { inputEl.style.transform = 'translate(50px, 200px)' }

    // 隐藏层1
    const hidden1 = createSubnetNode('Hidden1', 'mlp', {
        input_size: 64,
        output_size: 32,
        hidden_layers: 1,
        hidden_size: 32,
        activation: 'relu'
    })
    hidden1.position = [300, 150]
    await editor.addNode(hidden1)
    area.update('node', hidden1)
    await new Promise(r => setTimeout(r, 50))
    const h1El = document.querySelector(`[data-id="${hidden1.id}"]`)
    if (h1El) { h1El.style.transform = 'translate(300px, 150px)' }

    // 隐藏层2
    const hidden2 = createSubnetNode('Hidden2', 'mlp', {
        input_size: 32,
        output_size: 10,
        hidden_layers: 0,
        activation: 'relu'
    })
    hidden2.position = [550, 200]
    await editor.addNode(hidden2)
    area.update('node', hidden2)
    await new Promise(r => setTimeout(r, 50))
    const h2El = document.querySelector(`[data-id="${hidden2.id}"]`)
    if (h2El) { h2El.style.transform = 'translate(550px, 200px)' }

    // 输出层
    const output = createSubnetNode('Output', 'mlp', {
        input_size: 10,
        output_size: 10,
        hidden_layers: 0,
        activation: 'softmax'
    })
    output.position = [800, 200]
    await editor.addNode(output)
    area.update('node', output)
    await new Promise(r => setTimeout(r, 50))
    const outputEl = document.querySelector(`[data-id="${output.id}"]`)
    if (outputEl) { outputEl.style.transform = 'translate(800px, 200px)' }

    // 连接 - 使用 editor.addConnection API
    await editor.addConnection(new Connection(input, 'output', hidden1, 'input'))
    await editor.addConnection(new Connection(hidden1, 'output', hidden2, 'input'))
    await editor.addConnection(new Connection(hidden2, 'output', output, 'input'))

    // 等待所有节点渲染完成，然后强制更新位置
    await new Promise(r => setTimeout(r, 100))
    area.update('render')

    // 直接更新每个节点 wrapper 的 transform
    const allNodeElements = document.querySelectorAll('[data-testid="node"]')
    editor.nodes.forEach((node, index) => {
        if (allNodeElements[index]) {
            // 找到这个节点的 wrapper (祖父元素)
            const wrapper = allNodeElements[index].parentElement?.parentElement
            if (wrapper && wrapper.style) {
                wrapper.style.transform = `translate(${node.position[0]}px, ${node.position[1]}px)`
            }
        }
    })

    console.log('Example network added. Nodes:', editor.nodes.length)
}

/**
 * 添加节点
 */
async function addNode(type) {
    if (!editor) {
        console.error('Editor not initialized yet')
        return
    }
    const configs = {
        'input': { input_size: 10, output_size: 64, hidden_layers: 0, activation: 'linear' },
        'mlp': { input_size: 64, output_size: 32, hidden_layers: 2, hidden_size: 64, activation: 'relu' },
        'output': { input_size: 10, output_size: 10, hidden_layers: 0, activation: 'softmax' }
    }

    const config = configs[type] || configs['mlp']
    const count = editor.nodes.length + 1
    const node = createSubnetNode(`${type}_${count}`, 'mlp', config)

    // 计算新节点位置 - 靠近现有节点中心，避免放在左上角
    let baseX = 400
    let baseY = 300
    if (editor.nodes.length > 0) {
        // 计算所有现有节点的中心位置
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
        for (const n of editor.nodes) {
            if (n.position[0] < minX) minX = n.position[0]
            if (n.position[1] < minY) minY = n.position[1]
            if (n.position[0] > maxX) maxX = n.position[0]
            if (n.position[1] > maxY) maxY = n.position[1]
        }
        // 在现有节点区域的右下侧放置新节点
        baseX = maxX + 150
        baseY = (minY + maxY) / 2
    }

    // 添加随机偏移量，避免节点完全重叠
    const offsetX = (Math.random() - 0.5) * 100
    const offsetY = (Math.random() - 0.5) * 100
    node.position = [baseX + offsetX, baseY + offsetY]

    await editor.addNode(node)

    // 等待 Vue 渲染完成
    await new Promise(r => setTimeout(r, 100))
    area.update('render')

    // 直接更新节点 wrapper 的 transform - 祖父元素的 transform 包含位置
    const allNodeElements = document.querySelectorAll('[data-testid="node"]')
    const latestNodeEl = allNodeElements[allNodeElements.length - 1]
    if (latestNodeEl) {
        const wrapper = latestNodeEl.parentElement?.parentElement
        if (wrapper) {
            wrapper.style.transform = `translate(${node.position[0]}px, ${node.position[1]}px)`
        }
    }

    console.log('Node added. position:', node.position, 'editor.nodes:', editor.nodes.length)
}

/**
 * 构建网络状态 - 从编辑器节点获取最新数据
 */
function buildNetworkState() {
    const state = {
        name: 'my_network',
        version: '1.0.0',
        subnetworks: [],
        connections: []
    }

    if (!editor) return state

    // 遍历所有节点
    const nodes = editor.getNodes()
    for (const node of nodes) {
        const data = getNodeData(node)
        const subnet = {
            id: node.id,
            type: data.type,
            name: node.label,
            input_size: data.input_size,
            hidden_layers: data.hidden_layers,
            hidden_size: data.hidden_size,
            output_size: data.output_size,
            activation: data.activation,
            output_ports: [{ id: `${node.id}_out`, name: 'output' }],
            input_ports: [{ id: `${node.id}_in`, name: 'input' }]
        }
        state.subnetworks.push(subnet)

        // 从节点的输入端口获取连接信息
        Object.entries(node.inputs).forEach(([key, input]) => {
            if (input.connections && input.connections.length > 0) {
                input.connections.forEach(conn => {
                    state.connections.push({
                        id: `conn_${conn.id || '0'}`,
                        source_subnet_id: conn.output.node.id,
                        source_key: conn.output.key,
                        target_subnet_id: node.id,
                        target_key: key,
                        merge_strategy: 'sum'
                    })
                })
            }
        })
    }

    return state
}

/**
 * 导出为 JSON
 */
function exportToJSON() {
    const state = buildNetworkState()
    return JSON.stringify(state, null, 2)
}

/**
 * 导出为 C 代码
 */
function exportToC() {
    const state = buildNetworkState()

    // 激活函数映射
    const activationMap = {
        'relu': 'MLP_ACT_RELU',
        'sigmoid': 'MLP_ACT_SIGMOID',
        'tanh': 'MLP_ACT_TANH',
        'softmax': 'MLP_ACT_SOFTMAX',
        'linear': 'MLP_ACT_NONE',
        'leaky_relu': 'MLP_ACT_LEAKY_RELU'
    }

    let code = `/**
 * @file network_gen.c
 * @brief Generated network for ${state.name} v${state.version}
 * Generated by Action NN-C Web Editor
 */

#include "profiler.h"
#include "network_def.h"
#include "types/mlp/mlp_config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * @brief Create network: ${state.name}
 */
static NN_NetworkDef* create_network(void) {
    NN_NetworkDef* network;
    NNSubnetDef* subnet;
    MlpConfig* infer_config;
    MlpTrainConfig train_config;
    int result;

    network = nn_network_def_create("${state.name}");
    if (network == NULL) {
        fprintf(stderr, "Failed to create network definition\\n");
        return NULL;
    }

`

    // 生成每个子网的创建代码
    state.subnetworks.forEach((subnet, i) => {
        const hiddenCount = subnet.hidden_layers || 0
        const hiddenSizes = []

        // 如果有隐藏层，生成隐藏层大小数组
        if (hiddenCount > 0) {
            // 使用 hidden_size 作为所有隐藏层的大小
            for (let h = 0; h < hiddenCount; h++) {
                hiddenSizes.push(`    size_t hidden_size_${i}_${h} = ${subnet.hidden_size || 64}U;`)
            }
        }

        code += `    /* Subnetwork ${i}: ${subnet.name} (${subnet.type}) */\n`
        code += `    subnet = nn_subnet_def_create("${subnet.id}", "${subnet.type}", ${subnet.input_size}U, ${subnet.output_size}U);\n`
        code += `    if (subnet == NULL) {\n`
        code += `        fprintf(stderr, "Failed to create subnet ${subnet.name}\\n");\n`
        code += `        nn_network_def_free(network);\n`
        code += `        return NULL;\n`
        code += `    }\n`

        if (hiddenCount > 0) {
            code += `\n`
            hiddenSizes.forEach(hs => code += hs + '\n')
            code += `\n`
            code += `    size_t hidden_sizes_${i}[] = { `
            for (let h = 0; h < hiddenCount; h++) {
                code += `hidden_size_${i}_${h}` + (h < hiddenCount - 1 ? ', ' : '')
            }
            code += ` };\n`
            code += `    result = nn_subnet_def_set_hidden_layers(subnet, ${hiddenCount}U, hidden_sizes_${i});\n`
            code += `    if (result != 0) {\n`
            code += `        fprintf(stderr, "Failed to set hidden layers for ${subnet.name}\\n");\n`
            code += `        nn_subnet_def_free(subnet);\n`
            code += `        nn_network_def_free(network);\n`
            code += `        return NULL;\n`
            code += `    }\n`
        }

        // 创建 MLP 推理配置
        const hiddenAct = activationMap[subnet.activation] || 'MLP_ACT_RELU'
        const outputAct = activationMap[subnet.activation] || 'MLP_ACT_SOFTMAX'

        code += `\n`
        code += `    infer_config = mlp_config_create(${hiddenCount}U);\n`
        code += `    if (infer_config == NULL) {\n`
        code += `        fprintf(stderr, "Failed to create infer config for ${subnet.name}\\n");\n`
        code += `        nn_subnet_def_free(subnet);\n`
        code += `        nn_network_def_free(network);\n`
        code += `        return NULL;\n`
        code += `    }\n`
        code += `    mlp_config_init(infer_config, ${subnet.input_size}U, ${hiddenCount}U, `
        if (hiddenCount > 0) {
            code += `hidden_sizes_${i}, `
        } else {
            code += `NULL, `
        }
        code += `${subnet.output_size}U, ${hiddenAct}, ${outputAct});\n`

        code += `    result = nn_subnet_def_set_infer_type_config(\n`
        code += `        subnet,\n`
        code += `        infer_config,\n`
        code += `        mlp_config_size_for_hidden_layers(infer_config->hidden_layer_count),\n`
        code += `        "types/mlp/mlp_config.h",\n`
        code += `        "MlpConfig");\n`
        code += `    free(infer_config);\n`
        code += `    if (result != 0) {\n`
        code += `        fprintf(stderr, "Failed to set infer type config for ${subnet.name}\\n");\n`
        code += `        nn_subnet_def_free(subnet);\n`
        code += `        nn_network_def_free(network);\n`
        code += `        return NULL;\n`
        code += `    }\n`

        // 训练配置
        code += `\n`
        code += `    memset(&train_config, 0, sizeof(train_config));\n`
        code += `    train_config.learning_rate = 0.001f;\n`
        code += `    train_config.momentum = 0.9f;\n`
        code += `    train_config.weight_decay = 0.0001f;\n`
        code += `    train_config.optimizer = MLP_OPT_ADAM;\n`
        code += `    train_config.loss_func = MLP_LOSS_CROSS_ENTROPY;\n`
        code += `    train_config.batch_size = 1U;\n`
        code += `    train_config.seed = 42U;\n`
        code += `    result = nn_subnet_def_set_train_type_config(\n`
        code += `        subnet,\n`
        code += `        &train_config,\n`
        code += `        sizeof(train_config),\n`
        code += `        "types/mlp/mlp_config.h",\n`
        code += `        "MlpTrainConfig");\n`
        code += `    if (result != 0) {\n`
        code += `        fprintf(stderr, "Failed to set train type config for ${subnet.name}\\n");\n`
        code += `        nn_subnet_def_free(subnet);\n`
        code += `        nn_network_def_free(network);\n`
        code += `        return NULL;\n`
        code += `    }\n`

        // 添加子网到网络
        code += `\n`
        code += `    result = nn_network_def_add_subnet(network, subnet);\n`
        code += `    if (result != 0) {\n`
        code += `        fprintf(stderr, "Failed to add subnet ${subnet.name}\\n");\n`
        code += `        nn_subnet_def_free(subnet);\n`
        code += `        nn_network_def_free(network);\n`
        code += `        return NULL;\n`
        code += `    }\n\n`
    })

    // 添加连接
    code += `    /* Connections */\n`
    state.connections.forEach((conn, i) => {
        code += `    {\n`
        code += `        NNConnectionDef* conn_${i} = nn_connection_def_create(\n`
        code += `            "conn_${conn.id}",\n`
        code += `            "${conn.source_subnet_id}", "${conn.source_key}",\n`
        code += `            "${conn.target_subnet_id}", "${conn.target_key}");\n`
        code += `        if (conn_${i} == NULL) {\n`
        code += `            fprintf(stderr, "Failed to create connection ${conn.id}\\\n");\n`
        code += `            nn_network_def_free(network);\n`
        code += `            return NULL;\n`
        code += `        }\n`
        code += `        result = nn_network_def_add_connection(network, conn_${i});\n`
        code += `        if (result != 0) {\n`
        code += `            fprintf(stderr, "Failed to add connection ${conn.id}\\\n");\n`
        code += `            nn_connection_def_free(conn_${i});\n`
        code += `            nn_network_def_free(network);\n`
        code += `            return NULL;\n`
        code += `        }\n`
        code += `    }\n\n`
    })

    code += `    return network;\n`
    code += `}\n\n`
    code += `int main(void) {\n`
    code += `    NN_NetworkDef* network = create_network();\n`
    code += `    if (network == NULL) {\n`
    code += `        fprintf(stderr, "Failed to create network\\n");\n`
    code += `        return 1;\n`
    code += `    }\n\n`
    code += `    printf("Network created: %s\\n", network->network_name);\n`
    code += `    printf("Subnets: %zu\\n", network->subnet_count);\n`
    code += `    printf("Connections: %zu\\n", network->connection_count);\n\n`
    code += `    /* Use profiler to generate code from network definition */\n`
    code += `    /* ... call profiler_generate_v2() here ... */\n\n`
    code += `    nn_network_def_free(network);\n`
    code += `    return 0;\n`
    code += `}\n`

    return code
}

/**
 * 显示输出
 */
function showOutput(content, type) {
    const existing = document.querySelector('.output-panel')
    if (existing) existing.remove()

    const outputDiv = document.createElement('div')
    outputDiv.className = 'output-panel'

    const header = document.createElement('div')
    header.style.cssText = 'display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;'

    const h3 = document.createElement('h3')
    h3.style.cssText = 'color: #569cd6; margin: 0;'
    h3.textContent = `Output (${type.toUpperCase()})`

    const btnDiv = document.createElement('div')
    const copyBtn = document.createElement('button')
    copyBtn.style.cssText = 'padding: 8px 15px; background: #0e639c; border: none; border-radius: 4px; color: white; cursor: pointer; margin-right: 10px;'
    copyBtn.textContent = 'Copy'
    copyBtn.onclick = function() { copyOutput(copyBtn) }

    const closeBtn = document.createElement('button')
    closeBtn.style.cssText = 'padding: 8px 15px; background: #4e1515; border: none; border-radius: 4px; color: #f48771; cursor: pointer;'
    closeBtn.textContent = 'Close'
    closeBtn.onclick = function() { outputDiv.remove() }

    btnDiv.appendChild(copyBtn)
    btnDiv.appendChild(closeBtn)
    header.appendChild(h3)
    header.appendChild(btnDiv)

    const pre = document.createElement('pre')
    pre.style.cssText = 'background: #1e1e1e; padding: 15px; border-radius: 6px; overflow: auto; max-height: 60vh; font-family: Consolas, Monaco, monospace; font-size: 12px; line-height: 1.5; white-space: pre-wrap; word-break: break-all;'
    pre.textContent = content

    outputDiv.appendChild(header)
    outputDiv.appendChild(pre)
    document.body.appendChild(outputDiv)
}

async function copyOutput(btn) {
    const pre = btn.closest('.output-panel').querySelector('pre')
    const text = pre.textContent

    try {
        await navigator.clipboard.writeText(text)
        btn.textContent = 'Copied!'
    } catch (err) {
        // Fallback for older browsers or insecure contexts
        try {
            const textarea = document.createElement('textarea')
            textarea.value = text
            textarea.style.cssText = 'position:fixed;left:-9999px;top:-9999px;'
            document.body.appendChild(textarea)
            textarea.focus()
            textarea.select()
            const success = document.execCommand('copy')
            document.body.removeChild(textarea)
            if (success) {
                btn.textContent = 'Copied!'
            } else {
                throw new Error('execCommand copy failed')
            }
        } catch (fallbackErr) {
            console.error('Copy failed:', fallbackErr)
            btn.textContent = 'Failed!'
        }
    }

    setTimeout(() => btn.textContent = 'Copy', 2000)
}

// 导出功能
function exportJSON() {
    showOutput(exportToJSON(), 'json')
}

function exportC() {
    showOutput(exportToC(), 'c')
}

// 初始化
initEditor().then(() => {
    // 初始化完成后导出到全局
    window.editor = editor
    window.area = area
    console.log('Editor initialized and exported to window')
}).catch((err) => {
    document.getElementById('editor').innerHTML =
        '<div style="color:red;padding:20px;">Editor failed to initialize: '
        + err.message + '</div>'
    console.error('Editor init error:', err)
})

// 导出到全局
window.addNode = addNode
window.exportJSON = exportJSON
window.exportC = exportC
window.addExampleNetwork = addExampleNetwork
window.copyOutput = copyOutput

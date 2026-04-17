# action_nn_c

Neural Network Library in C with Code Generation

## Project Overview

action_nn_c is a pure C implementation of a neural network library with code generation (profiler) capabilities. This project supports multiple network topologies, including MLP, CNN, RNN, GNN, and Transformer, making it suitable for embedded systems and game AI applications.

## Features

- **Support for Multiple Network Types**
  - MLP (Multi-Layer Perceptron)
  - CNN (Convolutional Neural Network)
  - CNN Dual Pool (Dual-Pooling Convolutional Network)
  - RNN (Recurrent Neural Network)
  - GNN (Graph Neural Network)
  - Transformer (Attention-Based Network)

- **Complete Training Pipeline**
  - generate: Network definition generation
  - train: Model training
  - infer: Model inference

- **Code Generation System (Profiler)**
  - Automatic network topology analysis
  - Code generation and weight management
  - Runtime support for training and inference

## Directory Structure

```
action_nn_c/
├── src/
│   ├── nn/                 # Core neural network implementation
│   │   └── types/          # Network type implementations
│   │       ├── mlp/         # MLP implementation
│   │       ├── cnn/         # CNN implementation
│   │       ├── cnn_dual_pool/
│   │       ├── gnn/        # GNN implementation
│   │       ├── rnn/        # RNN implementation
│   │       └── transformer/
│   ├── profiler/          # Code generator
│   ├── infer/             # Inference runtime
│   └── train/             # Training runtime
├── demo/                  # Example projects
│   ├── mnist/             # MNIST handwritten digit recognition
│   ├── mnist_cnn/         # MNIST CNN version
│   ├── move/               # Movement control demo
│   ├── target/             # Target tracking demo
│   ├── sevenseg/           # Seven-segment display recognition
│   ├── nested_nav/         # Nested navigation
│   ├── road_graph_nav/     # Road graph navigation
│   ├── cnn_rnn_react/     # CNN+RNN reactive control
│   ├── hybrid_route/       # Hybrid routing
│   ├── transformer/       # Transformer dialogue
│   └── cs/                # CS game demo
└── docs/                  # Development documentation
```

## Quick Start

### Build and Run Workflow

Each demo project follows the same 6-step process:

```bash
# Step 1: Configure and build the generator
mkdir -p build/generate && cd build/generate
cmake ../../demo/xxx -G "Unix Makefiles"  # Or use another generator
cmake --build .

# Step 2: Run the generator
./generate/xxx_generate

# Step 3: Configure and build the training runtime
mkdir -p build/train && cd build/train
cmake ../../demo/xxx -G "Unix Makefiles"
cmake --build .

# Step 4: Run training
./train/xxx_train

# Step 5: Configure and build the inference runtime
mkdir -p build/infer && cd build/infer
cmake ../../demo/xxx -G "Unix Makefiles"
cmake --build .

# Step 6: Run inference
./infer/xxx_infer
```

### Windows Build (using clang)

```powershell
# Use the provided script
.\demo\xxx\run_demo.bat
```

## Network Types

### MLP (Multi-Layer Perceptron)

A fully connected neural network supporting multiple activation functions (ReLU, Sigmoid, Tanh, Leaky ReLU), suitable for various classification and regression tasks.

### CNN (Convolutional Neural Network)

Includes convolutional and pooling layers, ideal for image feature extraction.

### RNN (Recurrent Neural Network)

Processes sequential data, suitable for sequence prediction and time series analysis.

### GNN (Graph Neural Network)

Neural network based on graph structures, suitable for road network navigation and graph analysis tasks.

### Transformer

Attention-based network, ideal for dialogue and text processing tasks.

## Demo Descriptions

| Demo | Description | Network Type |
|------|-------------|--------------|
| mnist | Handwritten digit recognition | MLP |
| mnist_cnn | Handwritten digit recognition (CNN version) | CNN |
| move | Movement control | MLP |
| target | Target tracking | MLP |
| sevenseg | Seven-segment display recognition | MLP |
| nested_nav | Nested navigation | MLP |
| road_graph_nav | Road graph navigation | GNN |
| cnn_rnn_react | Reactive control | CNN+RNN |
| hybrid_route | Hybrid routing | Transformer |
| transformer | Dialogue example | Transformer |
| cs | CS game demo | CNN+RNN |

## Development Guide

Refer to `docs/developer_manual.md` for detailed development procedures.

### Adding a New Network Type

1. Create a new directory under `src/nn/types/`
2. Implement inference-related functions
3. Register the network type in `nn_infer_registry.h`
4. Update CMakeLists.txt

## Documentation

- [User Manual](docs/user_manual.md)
- [Developer Manual](docs/developer_manual.md)
- [Network Design Manual](docs/network_design_manual.md)
- [Profiler Development Plan](docs/profiler_development_plan.md)

## License

See the LICENSE file.
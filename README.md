# CNN In C++ - High-Performance C++ CNN Library

**Header-only, C++20, zero-dependency**


---

## Overview

CNN In C++ is a header-only Convolutional Neural Network library implemented in pure C++20. It achieves CPU performance that matches or exceeds mainstream frameworks by using:

* Zero-overhead abstractions
* Directed Acyclic Graph (DAG) based autograd
* Implicit GEMM convolution (no memory explosion)
* AVX2 and FMA intrinsics
* OpenMP multithreading with adaptive thread control
* L1/L2 cache-aware tiling

No external dependencies beyond a C++20 compiler with OpenMP and AVX2 support.

---

## Architecture

CNN In C++ is built around a **Directed Acyclic Graph (DAG)** automatic differentiation engine.

Models in CNN In C++ are not simple lists; they are generalized DAGs managed by `Model.h`. This structure seamlessly supports complex network topologies like ResNet residual links and Inception parallel branches.

1. **Graph Compilation**: When `model.compile()` is called, CNN In C++ uses Kahn's algorithm to perform a topological sort of the graph. It mathematically proves execution order and propagates tensor shapes from the inputs to the outputs.
2. **Pre-allocation**: During compilation, *every single buffer* required for the forward and backward pass is pre-allocated.
3. **Forward Pass**: Walks the topological order, routing outputs to inputs and merging residual connections.
4. **Backward Pass (`backward()`)**: Walks the graph in **reverse topological order**. Gradients are gathered from all child nodes and accumulated into `grad_output_buffer` before triggering a layer's local `backward()` function, strictly implementing the chain rule.

---

## Why It's Fast

CNN In C++ aggressively targets CPU execution bottlenecks:

### 1. Zero-Overhead Memory Allocation
Memory allocation (`malloc`/`new`) is disastrous for multi-threaded performance due to OS lock contention. CNN In C++ allocates everything once during `compile()`. The "hot path" (training loops and inference) is entirely free of allocations, dynamically sized vectors, and virtual exceptions. Furthermore, `NoInitAllocator` bypasses zero-initialization, dropping buffer setup time from ~200ms to ~1ms.

### 2. Implicit GEMM in Conv2D
Traditional frameworks (including PyTorch CPU) lower convolution to matrix multiplication using `im2col`, which inflates memory usage by up to 3× and wastes memory bandwidth. CNN In C++'s `Conv2D` uses an **implicit GEMM** approach. It accumulates dot products directly into AVX2 registers (`_mm256_fmadd_ps`) by traversing the spatial dimensions without ever materializing an intermediate column matrix.

### 3. Vectorization Strategies
- **Manual AVX2 Intrinsics**: Inner loops for `Conv2D` and `Dense` are painstakingly unrolled to maximize hardware register occupancy (e.g. accumulating 32 floats simultaneously using 4 AVX registers).
- **#pragma omp simd**: For element-wise operations (Activations, ElementwiseAdd, Loss), OpenMP SIMD directives are used to auto-vectorize loops perfectly, turning simple loops into optimal assembly.
- **Adaptive Threading**: `if(N_batch > 1)` is used to disable OpenMP overhead entirely when the batch size is 1 (micro-latency inference).

---

## Layer Reference

| Layer | Mathematical Formula | Time Complexity |
| :--- | :--- | :--- |
| **Conv2D** | $Y = \sum (X_{i,j} \cdot W) + b$ | $O(N \cdot C_{out} \cdot C_{in} \cdot K^2 \cdot H_{out} \cdot W_{out})$ |
| **Dense** | $Y = X \cdot W^T + b$ | $O(N \cdot C_{in} \cdot C_{out})$ |
| **BatchNorm2D** | $Y = \gamma \frac{X - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$ | $O(N \cdot C \cdot H \cdot W)$ |
| **FusedConvBNReLU**| $Y = \max(0, \text{ConvBN}(X))$ | $O(N \cdot C_{out} \cdot C_{in} \cdot K^2 \cdot H_{out} \cdot W_{out})$ |
| **MaxPool2D** | $Y = \max_{K \times K}(X)$ | $O(N \cdot C \cdot K^2 \cdot H_{out} \cdot W_{out})$ |
| **ReLU** | $Y = \max(0, X)$ | $O(N \cdot C \cdot H \cdot W)$ |
| **LeakyReLU** | $Y = \max(\alpha X, X)$ | $O(N \cdot C \cdot H \cdot W)$ |
| **Sigmoid** | $Y = \frac{1}{1 + e^{-X}}$ | $O(N \cdot C \cdot H \cdot W)$ |
| **Softmax** | $Y_i = \frac{e^{X_i - \max(X)}}{\sum e^{X_j - \max(X)}}$ | $O(N \cdot C_{classes})$ |
| **Dropout** | $Y = X \cdot \text{mask} \cdot \frac{1}{1-p}$ | $O(N \cdot C \cdot H \cdot W)$ |
| **Flatten** | $Y = \text{reshape}(X)$ | $O(1)$ (Pointer copy overhead) |
| **Concat** | $Y = [X_1, X_2, \dots]$ | $O(N \cdot \sum(C_i) \cdot H \cdot W)$ |
| **ElementwiseAdd** | $Y = \sum X_i$ | $O(N \cdot C \cdot H \cdot W)$ |

---

## Getting Started

### Prerequisites

* C++20 compiler (GCC 10+, Clang 14+, or MSVC 2022 with `/std:c++20`)
* CPU with AVX2 and FMA support (Intel Haswell or later, AMD Excavator or later)
* OpenMP library (optional, for multithreading)

---

### Installation

Clone the repository:

```bash
git clone https://github.com/hardikhazari/Cnn-In-C-.git
cd Cnn_In_Cpp
```

No build or linking step is required – it is header-only. Just include the headers.

---

## Compilation Flags (Performance)

For best performance, compile with:

```bash
g++ -std=c++20 -O3 -mavx2 -mfma -fopenmp -march=native -ffast-math -I./include main.cpp -o my_program
```

| Flag | Purpose |
| :--- | :--- |
| `-mavx2 -mfma` | Enable AVX2 and FMA SIMD instructions |
| `-fopenmp` | Enable multithreading |
| `-march=native` | Optimize for host CPU |
| `-ffast-math` | Improves SIMD performance |
| `-O3` | Maximum optimization |

---

### Windows (MSVC)

```cmd
cl /std:c++20 /O2 /arch:AVX2 /openmp /Iinclude main.cpp
```

---

## Benchmarks

CNN In C++ consistently outperforms PyTorch CPU backend on VGG-style models:

* Up to **2.3× faster inference** on 8-core CPU
* **Optimized memory usage** for deep CNNs
* No memory duplication (im2col avoided)

Run the baseline benchmark using `main.cpp`. Output is recorded in `benchmarks/baseline_run.txt`.

---

## Repository Structure

```
Cnn_In_Cpp/
├── include/
│   └── CNN In C++/
│       ├── core/
│       │   ├── Tensor.h
│       │   └── Im2Col.h
│       ├── Layers/
│       │   ├── Layer.h
│       │   ├── Conv2D.h
│       │   ├── Dense.h
│       │   ├── BatchNorm2D.h
│       │   ├── FusedConvBNReLU.h
│       │   ├── MaxPool2D.h
│       │   ├── ReLU.h
│       │   ├── LeakyReLU.h
│       │   ├── Sigmoid.h
│       │   ├── Softmax.h
│       │   ├── Dropout.h
│       │   ├── Flatten.h
│       │   ├── Concat.h
│       │   └── ElementwiseAdd.h
│       ├── nn/
│       │   ├── Model.h
│       │   ├── Optimizer.h
│       │   ├── Adam.h
│       │   └── Loss.h
│       └── data/
│           ├── Dataset.h
│           └── DataLoader.h
├── benchmarks/
│   └── baseline_run.txt
├── main.cpp
└── README.md
```

---

## Licence 
BSD 3-Clause License - see LICENSE file.


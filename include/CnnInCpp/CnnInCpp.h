#pragma once

// Core Mathematical Structures
#include "core/Tensor.h"
#include "core/Im2Col.h"

// Neural Network Graph & Execution
#include "nn/Model.h"
#include "nn/Optimizer.h"
#include "nn/Adam.h"
#include "nn/Loss.h"

// Atomic Layers (Linear & DAG)
#include "Layers/Layer.h"
#include "Layers/Dense.h"
#include "Layers/Conv2D.h"
#include "Layers/MaxPool2D.h"
#include "Layers/Flatten.h"
#include "Layers/ReLU.h"
#include "Layers/LeakyReLU.h"
#include "Layers/Softmax.h"
#include "Layers/Sigmoid.h"
#include "Layers/BatchNorm2D.h"
#include "Layers/Dropout.h"
#include "Layers/Concat.h"
#include "Layers/ElementwiseAdd.h"
#include "Layers/FusedConvBNReLU.h"

// Data Processing Pipeline
#include "data/DataLoader.h"
#include "data/Dataset.h"

namespace CnnInCpp {

// API Layer Factory Functions
inline std::shared_ptr<Dense> dense(int in_size, int out_size) {
    return std::make_shared<Dense>(in_size, out_size);
}

inline std::shared_ptr<Conv2D> conv2d(int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 0) {
    return std::make_shared<Conv2D>(in_channels, out_channels, kernel_size, stride, padding);
}

inline std::shared_ptr<MaxPool2D> maxpool2d(int pool_size, int stride = 2) {
    return std::make_shared<MaxPool2D>(pool_size, stride);
}

inline std::shared_ptr<Flatten> flatten() {
    return std::make_shared<Flatten>();
}

inline std::shared_ptr<ReLU> relu() {
    return std::make_shared<ReLU>();
}

inline std::shared_ptr<LeakyReLU> leaky_relu(float negative_slope = 0.01f) {
    return std::make_shared<LeakyReLU>(negative_slope);
}

inline std::shared_ptr<Softmax> softmax() {
    return std::make_shared<Softmax>();
}

inline std::shared_ptr<Sigmoid> sigmoid() {
    return std::make_shared<Sigmoid>();
}

inline std::shared_ptr<BatchNorm2D> batchnorm2d(int num_features, float momentum = 0.1f, float eps = 1e-5f) {
    return std::make_shared<BatchNorm2D>(num_features, momentum, eps);
}

inline std::shared_ptr<Dropout> dropout(float rate = 0.5f) {
    return std::make_shared<Dropout>(rate);
}

inline std::shared_ptr<Concat> concat() {
    return std::make_shared<Concat>();
}

inline std::shared_ptr<ElementwiseAdd> elementwise_add() {
    return std::make_shared<ElementwiseAdd>();
}

inline std::shared_ptr<FusedConvBNReLU> fused_conv_bn_relu(int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 0, float momentum = 0.1f, float eps = 1e-5f) {
    return std::make_shared<FusedConvBNReLU>(in_channels, out_channels, kernel_size, stride, padding, momentum, eps);
}

} // namespace CnnInCpp

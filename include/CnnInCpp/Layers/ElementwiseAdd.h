// ============================================================================
// ElementwiseAdd.h — Residual/Skip Connections (f(x) + x).
//
// Adds two or more tensors of exactly the same shape. This is the core building
// block of ResNet architectures.
//
// Gradient routing:
//   If f(x) = A + B, then df/dA = 1 and df/dB = 1.
//   Therefore, during the backward pass, the incoming gradient is routed
//   (copied) exactly as-is to BOTH parent layers.
// ============================================================================
#pragma once
#include "Layer.h"
#include <stdexcept>

namespace CnnInCpp {

class ElementwiseAdd : public Layer {
public:
    inline void compile(const std::vector<std::vector<int>>& input_shapes) override {
        if (input_shapes.size()<2) throw std::runtime_error("ElementwiseAdd: needs >=2 inputs");
        output_buffer = Tensor(input_shapes[0]);
        multi_grad_input_buffers.clear();
        for (size_t i=0; i<input_shapes.size(); ++i) {
            multi_grad_input_buffers.push_back(Tensor(input_shapes[i]));
        }
    }

    inline Tensor& forward(const std::vector<const Tensor*>& inputs) override {
        float* d = output_buffer.data.data();
        const float* s0=inputs[0]->data.data();
        #pragma omp simd
        for (int j=0;j<output_buffer.size();++j) d[j]=s0[j];
        
        for (size_t i=1;i<inputs.size();++i) {
            const float* s=inputs[i]->data.data();
            #pragma omp simd
            for (int j=0;j<output_buffer.size();++j) d[j]+=s[j];
        }
        return output_buffer;
    }

    inline void backward_multi(const Tensor& go) override {
        const float* g=go.data.data();
        for (size_t i=0; i<multi_grad_input_buffers.size(); ++i) {
            float* d = multi_grad_input_buffers[i].data.data();
            #pragma omp simd
            for (int j=0; j<go.size(); ++j) d[j] = g[j];
        }
    }
    inline std::string name() const override { return "ElementwiseAdd"; }
};

} // namespace CnnInCpp

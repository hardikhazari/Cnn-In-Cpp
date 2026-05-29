// ============================================================================
// ReLU.h — Rectified Linear Unit activation.
//
// f(x) = max(0, x)
// Gradient = 1 if x > 0 else 0
//
// This is the default activation function for deep networks. It avoids the
// vanishing gradient problem of Sigmoid/Tanh, and it's computationally trivial.
//
// Performance note: The entire forward and backward loops are strictly mapped
// element-wise operations. We use `#pragma omp simd` to auto-vectorize them
// into `_mm256_max_ps` (forward) and a bitwise mask (backward).
// ============================================================================
#pragma once
#include "Layer.h"

namespace CnnInCpp {

class ReLU : public Layer {
public:
    inline void compile(const std::vector<std::vector<int>>& input_shapes) override {
        output_buffer = Tensor(input_shapes[0]);
        grad_input_buffer = Tensor(input_shapes[0]);
    }

    inline Tensor& forward(const Tensor& input) override {
        cached_input_ptr = &input;
        const float* s = input.data.data(); float* d = output_buffer.data.data();
        #pragma omp simd
        for (int i=0;i<input.size();++i) d[i] = s[i]>0?s[i]:0.0f;
        return output_buffer;
    }
    
    inline void backward(const Tensor& go) override {
        const float* g=go.data.data(); const float* ci=cached_input_ptr->data.data(); float* di=grad_input_buffer.data.data();
        #pragma omp simd
        for (int i=0;i<go.size();++i) di[i] = ci[i]>0?g[i]:0.0f;
    }
    inline std::string name() const override { return "ReLU"; }
};

} // namespace CnnInCpp

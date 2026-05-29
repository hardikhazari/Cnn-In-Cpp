// ============================================================================
// Sigmoid.h — Sigmoid activation function.
//
// f(x) = 1 / (1 + e^-x)
// Gradient = f(x) * (1 - f(x)) * grad_output
//
// Squashes values to the [0, 1] range. Useful for binary classification outputs,
// but rarely used in hidden layers today due to the vanishing gradient problem
// (the derivative approaches 0 when inputs are large positive or negative).
// ============================================================================
#pragma once
#include "Layer.h"
#include <cmath>

namespace CnnInCpp {

class Sigmoid : public Layer {
public:
    inline void compile(const std::vector<std::vector<int>>& input_shapes) override {
        output_buffer = Tensor(input_shapes[0]);
        grad_input_buffer = Tensor(input_shapes[0]);
    }

    inline Tensor& forward(const Tensor& input) override {
        cached_input_ptr = &input;
        const float* s = input.data.data(); float* d = output_buffer.data.data();
        #pragma omp simd
        for (int i=0;i<input.size();++i) d[i] = 1.0f / (1.0f + std::exp(-s[i]));
        return output_buffer;
    }
    inline void backward(const Tensor& go) override {
        const float* g = go.data.data(); 
        const float* out_val = output_buffer.data.data(); 
        float* di = grad_input_buffer.data.data();
        #pragma omp simd
        for (int i=0;i<go.size();++i) di[i] = g[i] * out_val[i] * (1.0f - out_val[i]);
    }
    inline std::string name() const override { return "Sigmoid"; }
};

} // namespace CnnInCpp

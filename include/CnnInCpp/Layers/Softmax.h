// ============================================================================
// Softmax.h — Softmax activation function over the last dimension.
//
// Converts a vector of logits into a probability distribution where all
// elements sum to 1. Usually used as the final layer before CrossEntropyLoss.
//
// Numerical Stability:
//   e^x can overflow very quickly if x is large. To prevent this, we subtract
//   the maximum value in the vector from all elements before exponentiating.
//   Since e^(x-max) / sum(e^(x-max)) == e^x / sum(e^x), the math is identical
//   but the exponents are all <= 0, guaranteeing no overflow.
// ============================================================================
#pragma once
#include "Layer.h"
#include <cmath>
#include <limits>
#include <algorithm>

namespace CnnInCpp {

class Softmax : public Layer {
public:
    inline void compile(const std::vector<std::vector<int>>& input_shapes) override {
        output_buffer = Tensor(input_shapes[0]);
        grad_input_buffer = Tensor(input_shapes[0]);
    }

    inline Tensor& forward(const Tensor& input) override {
        const int N=input.shape[0], C=input.shape[1];
        const float* src=input.data.data(); float* dst=output_buffer.data.data();
        for (int n=0;n<N;++n) {
            std::span<const float> row(src+n*C, C);
            std::span<float>       orow(dst+n*C, C);
            float mx=-std::numeric_limits<float>::infinity();
            for (float v:row) mx=std::max(mx,v);
            float se=0; for (float v:row) se+=std::exp(v-mx);
            for (int j=0;j<C;++j) orow[j]=std::exp(row[j]-mx)/se;
        }
        return output_buffer;
    }
    inline void backward(const Tensor& go) override {
        const int N=go.shape[0], C=go.shape[1];
        const float* g=go.data.data(); const float* oc=output_buffer.data.data(); float* di=grad_input_buffer.data.data();
        for (int n=0;n<N;++n) {
            std::span<const float> grow(g+n*C,C), ocrow(oc+n*C,C);
            std::span<float> dr(di+n*C,C);
            float dot=0; for (int j=0;j<C;++j) dot+=ocrow[j]*grow[j];
            for (int j=0;j<C;++j) dr[j]=ocrow[j]*(grow[j]-dot);
        }
    }
    inline std::string name() const override { return "Softmax"; }
};

} // namespace CnnInCpp

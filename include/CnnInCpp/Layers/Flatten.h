// ============================================================================
// Flatten.h — Flattens spatial dimensions into a single vector per batch.
//
// Converts a 4D tensor [Batch, Height, Width, Channels] into a 2D tensor
// [Batch, Height * Width * Channels]. This is usually the bridge between the
// convolutional feature extractors and the final Dense classifier head.
//
// Since CnnInCpp uses contiguous 1D std::vector buffers for its tensors,
// reshaping is literally just changing the `shape` metadata and doing a
// memcpy. It requires no complex data transposition.
// ============================================================================
#pragma once
#include "Layer.h"

namespace CnnInCpp {

class Flatten : public Layer {
public:
    inline void compile(const std::vector<std::vector<int>>& input_shapes) override {
        int N=input_shapes[0][0], fd=1;
        for (int i=1;i<(int)input_shapes[0].size();++i) fd*=input_shapes[0][i];
        output_buffer = Tensor(std::vector<int>{N,fd});
        grad_input_buffer = Tensor(input_shapes[0]);
    }

    inline Tensor& forward(const Tensor& input) override {
        const float* s=input.data.data(); float* d=output_buffer.data.data();
        #pragma omp simd
        for(int i=0;i<input.size();++i) d[i]=s[i];
        return output_buffer;
    }
    inline void backward(const Tensor& go) override {
        const float* g=go.data.data(); float* di=grad_input_buffer.data.data();
        #pragma omp simd
        for(int i=0;i<go.size();++i) di[i]=g[i];
    }
    inline std::string name() const override { return "Flatten"; }
};

} // namespace CnnInCpp

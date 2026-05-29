// ============================================================================
// Concat.h — Concatenates tensors along the channel dimension.
//
// Used for architectures like Inception or DenseNet where outputs from multiple
// parallel branches are merged together.
//
// Layout assumption: NHWC. Since we concatenate along the channel dimension (C),
// which is the innermost contiguous block in memory, this requires a stride-based
// memory copy. For each spatial pixel, we copy the channels from input A,
// then the channels from input B, etc.
// ============================================================================
#pragma once
#include "Layer.h"
#include <stdexcept>
#include <omp.h>

namespace CnnInCpp {

class Concat : public Layer {
public:
    std::vector<int> channel_splits;

    inline void compile(const std::vector<std::vector<int>>& input_shapes) override {
        if (input_shapes.empty()) throw std::runtime_error("Concat: needs >0 inputs");
        // [N, H, W, C]
        int total_c=0, N=input_shapes[0][0], H=input_shapes[0][1], W=input_shapes[0][2];
        channel_splits.clear();
        multi_grad_input_buffers.clear();
        for (const auto& sh: input_shapes) { 
            total_c += sh[3]; 
            channel_splits.push_back(sh[3]); 
            multi_grad_input_buffers.push_back(Tensor(sh));
        }
        output_buffer = Tensor(N, H, W, total_c);
    }

    inline Tensor& forward(const std::vector<const Tensor*>& inputs) override {
        if (inputs.empty()) throw std::runtime_error("Concat: needs >0 inputs");
        int total_c=0, N=inputs[0]->shape[0], H=inputs[0]->shape[1], W=inputs[0]->shape[2];
        for (int sp : channel_splits) total_c += sp;
        float* dst = output_buffer.data.data();
        
        #pragma omp parallel for schedule(static)
        for (int n=0;n<N;++n) {
            for (int hw=0;hw<H*W;++hw) {
                int cc=0;
                for (size_t i=0;i<inputs.size();++i) {
                    int ci=channel_splits[i];
                    const float* src=inputs[i]->data.data() + n*(H*W*ci) + hw*ci;
                    float* d=dst + n*(H*W*total_c) + hw*total_c + cc;
                    #pragma omp simd
                    for (int c=0;c<ci;++c) d[c] = src[c];
                    cc+=ci;
                }
            }
        }
        return output_buffer;
    }

    inline void backward_multi(const Tensor& go) override {
        const int N=go.shape[0], H=go.shape[1], W=go.shape[2];
        const int total_c=go.shape[3];
        const float* src=go.data.data();
        
        #pragma omp parallel for schedule(static)
        for (int n=0;n<N;++n) {
            for (int hw=0;hw<H*W;++hw) {
                int cc=0;
                for (size_t i=0; i<channel_splits.size(); ++i) {
                    int sp = channel_splits[i];
                    float* d=multi_grad_input_buffers[i].data.data() + n*(H*W*sp) + hw*sp;
                    const float* s=src + n*(H*W*total_c) + hw*total_c + cc;
                    #pragma omp simd
                    for (int c=0;c<sp;++c) d[c] = s[c];
                    cc+=sp;
                }
            }
        }
    }
    inline std::string name() const override { return "Concat"; }
};

} // namespace CnnInCpp

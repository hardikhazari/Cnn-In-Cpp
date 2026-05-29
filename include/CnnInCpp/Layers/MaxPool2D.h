// ============================================================================
// MaxPool2D.h — Spatial Max Pooling layer.
//
// Downsamples the spatial dimensions by taking the maximum value within a
// sliding window. This reduces computational load for deeper layers and provides
// a degree of translation invariance.
//
// Performance notes:
//   - Forward pass: We compute the max, but we ALSO need to save the index
//     of where that max came from. This is required for the backward pass so
//     we can route the gradient back to the exact input pixel that "won".
//   - The `#pragma omp simd` handles the channel loop, meaning we do pooling
//     across 8 channels simultaneously in AVX2 registers.
// ============================================================================
#pragma once
#include "Layer.h"
#include <limits>

namespace CnnInCpp {

class MaxPool2D : public Layer {
public:
    int pool_size, stride;
    inline MaxPool2D(int p, int s=2) : pool_size(p), stride(s) {}

    inline void compile(const std::vector<std::vector<int>>& input_shapes) override {
        // Input NHWC: N, H, W, C
        const int N = input_shapes[0][0], H = input_shapes[0][1];
        const int W = input_shapes[0][2], C = input_shapes[0][3];
        const int OH = (H - pool_size) / stride + 1, OW = (W - pool_size) / stride + 1;
        output_buffer = Tensor(N, OH, OW, C);
        grad_input_buffer = Tensor(input_shapes[0]);
    }

   inline Tensor& forward(const Tensor& input) override {
        cached_input_ptr = &input;
        const int N=input.shape[0],H=input.shape[1],W=input.shape[2],C=input.shape[3];
        const int OH=output_buffer.shape[1], OW=output_buffer.shape[2];
        const float* src = input.data.data();
        float* dst = output_buffer.data.data();

        // [LATENCY FIX] Bypass OpenMP overhead when Batch Size == 1
        #pragma omp parallel for schedule(static) if(N > 1)
        for (int n=0;n<N;++n) {
            const float* sn = src + n*(H*W*C);
            float* dn = dst + n*(OH*OW*C);
            for (int y=0;y<OH;++y) {
                for (int x=0;x<OW;++x) {
                    float* dc = dn + (y*OW+x)*C;
                    for (int c=0;c<C;++c) dc[c] = -std::numeric_limits<float>::infinity();
                    
                    for (int py=0;py<pool_size;++py) {
                        for (int px=0;px<pool_size;++px) {
                            const float* sc = sn + ((y*stride+py)*W + (x*stride+px))*C;
                            #pragma omp simd
                            for (int c=0;c<C;++c) {
                                // [AVX2 FIX] Replaced 'if' statement with std::max for proper vectorization
                                dc[c] = std::max(dc[c], sc[c]);
                            }
                        }
                    }
                }
            }
        }
        return output_buffer;
    }

    inline void backward(const Tensor& go) override {
        const int N=cached_input_ptr->shape[0],H=cached_input_ptr->shape[1];
        const int W=cached_input_ptr->shape[2],C=cached_input_ptr->shape[3];
        const int OH=go.shape[1], OW=go.shape[2];
        grad_input_buffer.fill(0.0f);
        const float* src = cached_input_ptr->data.data();
        const float* g   = go.data.data();
        float*       di  = grad_input_buffer.data.data();
        
        #pragma omp parallel for schedule(static)
        for (int n=0;n<N;++n) {
            const float* sn = src + n*(H*W*C);
            const float* gn = g   + n*(OH*OW*C);
            float*       din = di + n*(H*W*C);
            
            for (int y=0;y<OH;++y) {
                for (int x=0;x<OW;++x) {
                    const float* gc = gn + (y*OW+x)*C;
                    
                    for (int c=0;c<C;++c) {
                        float mv=-std::numeric_limits<float>::infinity(); 
                        int my=0, mx=0;
                        for (int py=0;py<pool_size;++py) {
                            for (int px=0;px<pool_size;++px) {
                                int iy=y*stride+py, ix=x*stride+px;
                                float v=sn[(iy*W+ix)*C + c]; 
                                if(v>mv){mv=v;my=iy;mx=ix;}
                            }
                        }
                        din[(my*W+mx)*C + c] += gc[c];
                    }
                }
            }
        }
    }
    inline std::string name() const override { return "MaxPool2D"; }
};

} // namespace CnnInCpp

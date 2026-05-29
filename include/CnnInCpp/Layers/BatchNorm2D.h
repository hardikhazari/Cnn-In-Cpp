// ============================================================================
// BatchNorm2D.h — Batch Normalization over a 4D tensor (NHWC layout).
//
// Normalizes the activations of the previous layer to have zero mean and unit
// variance, then applies learned scale (gamma) and shift (beta) parameters.
// This stabilizes the distribution of inputs to deep layers, preventing
// vanishing/exploding gradients and allowing higher learning rates.
//
// Training Mode:
//   Computes the mean and variance across the batch and spatial dimensions
//   for each channel. Uses these to normalize the current batch, and updates
//   a running EMA (Exponential Moving Average) of the global mean/variance.
//
// Inference Mode (Eval):
//   Uses the frozen running mean/variance to normalize. In a real deployment,
//   BatchNorm is usually mathematically "folded" into the preceding Conv2D
//   layer (modifying its weights and biases) so this layer disappears entirely!
//   See FusedConvBNReLU for an example of this.
//
// Layout optimizations:
//   NHWC layout means spatial dimensions and batch are the outer loops, and
//   the channel dimension is contiguous. This allows us to use `#pragma omp simd`
//   on the inner channel loop to compute statistics across all channels in parallel.
// ============================================================================
#pragma once
#include "Layer.h"
#include <cmath>

namespace CnnInCpp {

class BatchNorm2D : public Layer {
public:
    int   num_features;
    float momentum;
    float eps;
    Tensor gamma, beta;
    Tensor running_mean, running_var;
    Tensor x_norm, batch_var, batch_mean;

    inline void compile(const std::vector<std::vector<int>>& input_shapes) override {
        // [N, H, W, C]
        const int C=input_shapes[0][3];
        output_buffer = Tensor(input_shapes[0]);
        grad_input_buffer = Tensor(input_shapes[0]);
        x_norm = Tensor(input_shapes[0]);
        batch_var = Tensor(C,1);
        batch_mean = Tensor(C,1);
    }

    inline BatchNorm2D(int features, float mom=0.1f, float e=1e-5f)
        : num_features(features), momentum(mom), eps(e) {
        gamma=Tensor(features,1); beta=Tensor(features,1);
        running_mean=Tensor(features,1); running_var=Tensor(features,1);
        gamma.fill(1.0f); beta.fill(0.0f);
        running_mean.fill(0.0f); running_var.fill(1.0f);
    }

    inline std::vector<Tensor*> get_parameters() override { return {&gamma, &beta}; }
    inline std::vector<Tensor*> get_states()     override {
        return {&gamma, &beta, &running_mean, &running_var};
    }

   inline Tensor& forward(const Tensor& input) override {
        cached_input_ptr = &input;
        const int N=input.shape[0], H=input.shape[1], W=input.shape[2], C=input.shape[3];
        const int HW=H*W, m=N*HW;
        
        const float* inp = input.data.data();
        float* out = output_buffer.data.data();
        const float* gam = gamma.data.data();
        const float* bet = beta.data.data();

        if (is_training) {
            float* mean = batch_mean.data.data();
            float* var = batch_var.data.data();
            
            for (int c=0; c<C; ++c) { mean[c] = 0.0f; var[c] = 0.0f; }

            for (int b=0; b<N; ++b) {
                const float* ptr = inp + b*(HW*C);
                for (int hw=0; hw<HW; ++hw) {
                    const float* p = ptr + hw*C;
                    #pragma omp simd
                    for (int c=0; c<C; ++c) mean[c] += p[c];
                }
            }
            for (int c=0; c<C; ++c) mean[c] /= m;

            for (int b=0; b<N; ++b) {
                const float* ptr = inp + b*(HW*C);
                for (int hw=0; hw<HW; ++hw) {
                    const float* p = ptr + hw*C;
                    #pragma omp simd
                    for (int c=0; c<C; ++c) {
                        float d = p[c] - mean[c];
                        var[c] += d * d;
                    }
                }
            }
            for (int c=0; c<C; ++c) var[c] /= m;

            for (int c=0; c<C; ++c) {
                running_mean.data[c] = (1-momentum)*running_mean.data[c] + momentum*mean[c];
                running_var.data[c]  = (1-momentum)*running_var.data[c]  + momentum*var[c];
            }

            float* xn = x_norm.data.data();
            #pragma omp parallel for schedule(static)
            for (int b=0; b<N; ++b) {
                const float* sv = inp + b*(HW*C);
                float* dv = out + b*(HW*C);
                float* xv = xn  + b*(HW*C);
                for (int hw=0; hw<HW; ++hw) {
                    const float* s = sv + hw*C;
                    float* d = dv + hw*C;
                    float* x = xv + hw*C;
                    #pragma omp simd
                    for (int c=0; c<C; ++c) {
                        float inv_std = 1.0f/std::sqrt(var[c]+eps);
                        x[c] = (s[c] - mean[c]) * inv_std;
                        d[c] = gam[c] * x[c] + bet[c];
                    }
                }
            }
        } else {
            const float* rm = running_mean.data.data();
            const float* rv = running_var.data.data();
            
            // [LATENCY FIX] Bypass OpenMP overhead when Batch Size == 1
            #pragma omp parallel for schedule(static) if(N > 1)
            for (int b=0; b<N; ++b) {
                const float* sv = inp + b*(HW*C);
                float* dv = out + b*(HW*C);
                for (int hw=0; hw<HW; ++hw) {
                    const float* s = sv + hw*C;
                    float* d = dv + hw*C;
                    #pragma omp simd
                    for (int c=0; c<C; ++c) {
                        float inv_std = 1.0f/std::sqrt(rv[c]+eps);
                        d[c] = gam[c] * (s[c] - rm[c]) * inv_std + bet[c];
                    }
                }
            }
        }
        return output_buffer;
    }

    inline void backward(const Tensor& grad_output) override {
        const int N=grad_output.shape[0], H=grad_output.shape[1];
        const int W=grad_output.shape[2], C=grad_output.shape[3];
        const int HW=H*W, m=N*HW;

        const float* go = grad_output.data.data();
        const float* xn = x_norm.data.data();
        float*       di = grad_input_buffer.data.data();
        float*       dg = gamma.grad.data();
        float*       db = beta.grad.data();
        const float* bv = batch_var.data.data();

        std::vector<float> sum_d(C, 0.0f);
        std::vector<float> sum_dx(C, 0.0f);

        for (int b=0; b<N; ++b) {
            const float* gov = go + b*(HW*C);
            const float* xnv = xn + b*(HW*C);
            for (int hw=0; hw<HW; ++hw) {
                const float* g = gov + hw*C;
                const float* x = xnv + hw*C;
                #pragma omp simd
                for (int c=0; c<C; ++c) {
                    sum_d[c]  += g[c];
                    sum_dx[c] += g[c]*x[c];
                }
            }
        }

        #pragma omp simd
        for (int c=0; c<C; ++c) {
            dg[c] += sum_dx[c];
            db[c] += sum_d[c];
        }

        #pragma omp parallel for schedule(static)
        for (int b=0; b<N; ++b) {
            const float* gov = go + b*(HW*C);
            const float* xnv = xn + b*(HW*C);
            float*       div = di + b*(HW*C);
            
            for (int hw=0; hw<HW; ++hw) {
                const float* g = gov + hw*C;
                const float* x = xnv + hw*C;
                float*       d = div + hw*C;
                #pragma omp simd
                for (int c=0; c<C; ++c) {
                    float inv_std=1.0f/std::sqrt(bv[c]+eps);
                    d[c] = (gamma.data[c]*inv_std/m)*(m*g[c] - sum_d[c] - x[c]*sum_dx[c]);
                }
            }
        }
    }
    inline std::string name() const override { return "BatchNorm2D"; }
};

} // namespace CnnInCpp

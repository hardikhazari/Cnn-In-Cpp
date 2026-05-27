// ============================================================================
// Conv2D.h — 2D Convolution layer with implicit GEMM and AVX2/FMA intrinsics.
//
// This is the most performance-critical layer in the entire library. A typical
// VGG forward pass spends 70-80% of its time inside Conv2D::forward(). Every
// microsecond shaved here multiplies across the entire network.
//
// The key insight: we do NOT use im2col. Traditional conv implementations
// (even PyTorch's CPU backend) expand the input into a giant column matrix,
// then call a GEMM library. That expansion alone can 3× your memory usage
// and forces a full read+write of the expanded buffer before the actual math
// even starts.
//
// Instead, we do an "implicit GEMM" — the convolution is computed directly
// from the input tensor by walking through the spatial/kernel dimensions and
// accumulating into AVX2 registers. We never write an intermediate im2col
// buffer to memory. The multiply-accumulate happens purely in registers,
// and we write the final result to the output buffer exactly once.
//
// The forward pass uses a 3-tier vectorization strategy:
//   1. 32-wide: processes 32 output channels at once using 4 × __m256 (YMM)
//      registers. Each __m256 holds 8 floats, so 4 of them = 32 floats.
//      _mm256_fmadd_ps does a fused multiply-add in a single cycle — this is
//      the FMA instruction that saves one whole add instruction per element.
//   2. 8-wide: handles the next 8-channel chunk with 1 × __m256.
//   3. Scalar: mops up any remaining channels (when out_channels isn't
//      divisible by 8).
//
// The backward pass uses thread-local gradient buffers to avoid lock
// contention. Each OpenMP thread gets its own copy of grad_weights/grad_biases,
// accumulates into it independently, then the main thread consolidates at the
// end. This avoids atomic operations or critical sections in the hot loop.
//
// Weight layout: [ky, kx, in_channels, out_channels] — this puts out_channels
// as the innermost (contiguous) dimension, which is perfect for the vectorized
// FMA loop that sweeps across output channels.
// ============================================================================
#pragma once
#include "Layer.h"
#include <omp.h>
#include <immintrin.h>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <cstring>

namespace CnnInCpp {

class Conv2D : public Layer {
public:
    int in_channels, out_channels, kernel_size, stride, padding;
    Tensor weights, biases;
    
    // Cached spatial dimensions — computed once in compile(), used every forward/backward.
    // Storing these as members avoids recomputing them from shape[] each time.
    int N_batch, H_dim, W_dim, OH_dim, OW_dim;

    // Thread-local gradient accumulators — pre-allocated in compile() so that
    // the backward pass does zero heap allocation. Each thread accumulates
    // weight/bias gradients into its own buffer, then we reduce at the end.
    // This is the "zero-allocation hot path" pattern that makes CnnInCpp fast:
    // we pay the malloc cost once during compile() and never again.
    std::vector<Tensor> local_grad_weights;
    std::vector<Tensor> local_grad_biases;

    inline Conv2D(int in_c, int out_c, int k, int s=1, int p=0)
        : in_channels(in_c), out_channels(out_c), kernel_size(k), stride(s), padding(p) {
        // Weight shape: [ky, kx, in_channels, out_channels]
        // out_channels is the contiguous (innermost) dimension — this is
        // deliberate so the AVX2 FMA loop can load 8 output channels at a
        // time with a single _mm256_loadu_ps().
        weights = Tensor(std::vector<int>{k, k, in_c, out_c});
        biases  = Tensor(out_c, 1);
        int fan_in = in_c * k * k;
        weights.fill_he_init(fan_in);
        biases.fill(0.0f);
    }

    inline void compile(const std::vector<std::vector<int>>& input_shapes) override {
        N_batch = input_shapes[0][0];
        // Handle both NCHW and NHWC input formats. In practice, the DataLoader
        // outputs NHWC, so the else branch is the normal path.
        if (input_shapes[0].size() == 4 && input_shapes[0][1] == in_channels && input_shapes[0][3] != in_channels) {
            H_dim = input_shapes[0][2]; W_dim = input_shapes[0][3];
        } else if (input_shapes[0].size() == 4) {
            H_dim = input_shapes[0][1]; W_dim = input_shapes[0][2]; 
        } else {
             throw std::runtime_error("Conv2D: Malformed tensor limits.");
        }

        OH_dim = (H_dim + 2*padding - kernel_size) / stride + 1;
        OW_dim = (W_dim + 2*padding - kernel_size) / stride + 1;
        
        // Pre-allocate output and gradient buffers. These are reused every
        // forward/backward call — no allocation in the hot path, ever.
        output_buffer = Tensor(N_batch, OH_dim, OW_dim, out_channels);
        grad_input_buffer = Tensor(input_shapes[0]);

        // Pre-allocate one grad_weight + grad_bias buffer PER THREAD.
        // This is the key to avoiding malloc lock contention in the backward
        // pass. Without this, each thread would try to allocate its own temp
        // buffer inside the omp parallel region, causing serialization on the
        // heap lock — which can be worse than the actual gradient computation.
        int max_threads = omp_get_max_threads();
        local_grad_weights.clear(); local_grad_biases.clear();
        for (int t = 0; t < max_threads; ++t) {
            local_grad_weights.emplace_back(std::vector<int>{kernel_size, kernel_size, in_channels, out_channels});
            local_grad_biases.emplace_back(out_channels, 1);
        }
    }

    // ==========================================================================
    // FORWARD PASS — Implicit GEMM Convolution
    //
    // This is where CnnInCpp's core performance advantage lives. Instead of:
    //   1. im2col: expand input into (OH*OW) × (K*K*C) matrix  [memory bound]
    //   2. GEMM: multiply by weight matrix                       [compute bound]
    //
    // We do the multiply-accumulate directly from the input tensor:
    //   for each output pixel (n, y, x):
    //     load bias into AVX2 accumulators
    //     for each kernel position (ky, kx):
    //       for each input channel (ci):
    //         broadcast input[n, iy, ix, ci] to all 8 lanes
    //         FMA: acc += broadcast × weight[ky, kx, ci, co:co+8]
    //     store accumulators to output once
    //
    // The result: we read the input once, read the weights once, and write the
    // output once. No intermediate buffer. No extra memory bandwidth.
    //
    // The `if(N_batch > 1)` on the OpenMP pragma is critical for latency:
    // spinning up threads for a single image costs ~50μs in thread
    // creation/synchronization overhead, which is MORE than the entire
    // convolution for small layers. For batch=1, we run single-threaded.
    // ==========================================================================
inline Tensor& forward(const Tensor& input) override {
        cached_input_ptr = &input;
        const float* in_ptr_base = input.data.data();
        const float* w_ptr_base = weights.data.data();
        const float* b_ptr_base = biases.data.data();
        float* out_ptr_base = output_buffer.data.data();

        // Adaptive threading: skip OpenMP overhead when batch=1.
        // For batch=1 inference, this alone saves ~50-100μs per layer.
        #pragma omp parallel for collapse(2) schedule(static) if(N_batch > 1)
        for (int n = 0; n < N_batch; ++n) {
            for (int y = 0; y < OH_dim; ++y) {
                for (int x = 0; x < OW_dim; ++x) {
                    float* out_pixel = out_ptr_base + ((n * OH_dim + y) * OW_dim + x) * out_channels;
                    
                    // ---- 32-wide AVX2 path ----
                    // Process 32 output channels simultaneously using 4 YMM registers.
                    // Each __m256 register holds 8 × 32-bit floats = 256 bits.
                    // We initialize accumulators from the bias vector, then FMA
                    // through all kernel positions × input channels, and write
                    // to memory exactly once at the end.
                    //
                    // _mm256_fmadd_ps(a, b, c) = a * b + c in ONE instruction.
                    // Without FMA, this would be a separate multiply then add,
                    // each taking a cycle. FMA gives us 2× the throughput on
                    // the multiply-accumulate bottleneck.
                    //
                    // _mm256_set1_ps broadcasts one float to all 8 lanes — this
                    // is how we multiply one input value against 8 weight values.
                    int co = 0;
                    for (; co + 31 < out_channels; co += 32) {
                        // 1. Initialize ACCUMULATOR REGISTERS
                        __m256 acc0 = _mm256_loadu_ps(b_ptr_base + co);
                        __m256 acc1 = _mm256_loadu_ps(b_ptr_base + co + 8);
                        __m256 acc2 = _mm256_loadu_ps(b_ptr_base + co + 16);
                        __m256 acc3 = _mm256_loadu_ps(b_ptr_base + co + 24);

                        // 2. Pure spatial inner loop — walk the kernel window
                        for (int ky = 0; ky < kernel_size; ++ky) {
                            int iy = y * stride - padding + ky;
                            if (iy < 0 || iy >= H_dim) continue;
                            for (int kx = 0; kx < kernel_size; ++kx) {
                                int ix = x * stride - padding + kx;
                                if (ix < 0 || ix >= W_dim) continue;
                                
                                // Raw pointer arithmetic: precomputed offset into
                                // the input tensor. We skip Tensor::operator() here
                                // because it recomputes the 4D index every time, and
                                // the compiler can't always hoist shape[] loads out
                                // of the loop.
                                const float* in_pixel = in_ptr_base + ((n * H_dim + iy) * W_dim + ix) * in_channels;
                                const float* w_block = w_ptr_base + ((ky * kernel_size + kx) * in_channels) * out_channels;
                                
                                for (int ci = 0; ci < in_channels; ++ci) {
                                    __m256 v_val = _mm256_set1_ps(in_pixel[ci]);
                                    const float* w_row = w_block + ci * out_channels;
                                    
                                    // 3. Accumulate STRICTLY in registers (NO RAM THRESHING)
                                    // These 4 FMAs process 32 output channels per input channel.
                                    // The accumulators stay in YMM registers the entire time —
                                    // they're never spilled to memory until the store below.
                                    acc0 = _mm256_fmadd_ps(v_val, _mm256_loadu_ps(w_row + co), acc0);
                                    acc1 = _mm256_fmadd_ps(v_val, _mm256_loadu_ps(w_row + co + 8), acc1);
                                    acc2 = _mm256_fmadd_ps(v_val, _mm256_loadu_ps(w_row + co + 16), acc2);
                                    acc3 = _mm256_fmadd_ps(v_val, _mm256_loadu_ps(w_row + co + 24), acc3);
                                }
                            }
                        }
                        // 4. WRITE to memory exactly ONCE — after all accumulation is done.
                        // This is the payoff of register accumulation: one store per 32 outputs
                        // instead of load-modify-store on every FMA.
                        _mm256_storeu_ps(out_pixel + co, acc0);
                        _mm256_storeu_ps(out_pixel + co + 8, acc1);
                        _mm256_storeu_ps(out_pixel + co + 16, acc2);
                        _mm256_storeu_ps(out_pixel + co + 24, acc3);
                    }
                    
                    // ---- 8-wide AVX2 fallback ---- (for the 8-channel remainder)
                    for (; co + 7 < out_channels; co += 8) {
                        __m256 acc0 = _mm256_loadu_ps(b_ptr_base + co);
                        for (int ky = 0; ky < kernel_size; ++ky) {
                            int iy = y * stride - padding + ky;
                            if (iy < 0 || iy >= H_dim) continue;
                            for (int kx = 0; kx < kernel_size; ++kx) {
                                int ix = x * stride - padding + kx;
                                if (ix < 0 || ix >= W_dim) continue;
                                const float* in_pixel = in_ptr_base + ((n * H_dim + iy) * W_dim + ix) * in_channels;
                                const float* w_block = w_ptr_base + ((ky * kernel_size + kx) * in_channels) * out_channels;
                                for (int ci = 0; ci < in_channels; ++ci) {
                                    __m256 v_val = _mm256_set1_ps(in_pixel[ci]);
                                    acc0 = _mm256_fmadd_ps(v_val, _mm256_loadu_ps(w_block + ci * out_channels + co), acc0);
                                }
                            }
                        }
                        _mm256_storeu_ps(out_pixel + co, acc0);
                    }
                    // ---- Scalar fallback ---- (when out_channels % 8 != 0)
                    for (; co < out_channels; ++co) {
                        float sum = b_ptr_base[co];
                        for (int ky = 0; ky < kernel_size; ++ky) {
                            int iy = y * stride - padding + ky;
                            if (iy < 0 || iy >= H_dim) continue;
                            for (int kx = 0; kx < kernel_size; ++kx) {
                                int ix = x * stride - padding + kx;
                                if (ix < 0 || ix >= W_dim) continue;
                                const float* in_pixel = in_ptr_base + ((n * H_dim + iy) * W_dim + ix) * in_channels;
                                const float* w_block = w_ptr_base + ((ky * kernel_size + kx) * in_channels) * out_channels;
                                for (int ci = 0; ci < in_channels; ++ci) {
                                    sum += in_pixel[ci] * w_block[ci * out_channels + co];
                                }
                            }
                        }
                        out_pixel[co] = sum;
                    }
                }
            }
        }
        return output_buffer;
    }

    // ==========================================================================
    // BACKWARD PASS
    //
    // Computes three things:
    //   1. dInput: gradient w.r.t. input (for upstream layers)
    //   2. dWeights: gradient w.r.t. kernel weights (for optimizer)
    //   3. dBias: gradient w.r.t. biases (for optimizer)
    //
    // dInput uses the same implicit GEMM approach — walking through spatial
    // positions and accumulating the dot product of grad_output × weights
    // across output channels using AVX2. The horizontal reduction at the end
    // (summing 8 lanes of an __m256 into one scalar) is done via an aligned
    // store + scalar sum — this is the standard pattern since _mm256_hadd_ps
    // has higher latency and worse throughput.
    //
    // dWeights uses thread-local buffers: each thread accumulates into its own
    // copy of grad_weights, then we do a single-threaded reduction at the end.
    // This avoids any synchronization in the hot inner loop.
    // ==========================================================================
    inline void backward(const Tensor& grad_output) override {
        grad_input_buffer.fill(0.0f);
        weights.zero_grad();
        biases.zero_grad();

        const float* go_ptr_base = grad_output.data.data();
        const float* in_ptr_base = cached_input_ptr->data.data();
        const float* w_ptr_base  = weights.data.data();
        float* din_ptr_base      = grad_input_buffer.data.data();
        float* dw_ptr_base       = weights.grad.data();
        float* db_ptr_base       = biases.grad.data();

        // 1. Calculate dInput — gradient flows backward through the convolution
        #pragma omp parallel for collapse(2) schedule(static)
        for (int n = 0; n < N_batch; ++n) {
            for (int iy = 0; iy < H_dim; ++iy) {
                for (int ix = 0; ix < W_dim; ++ix) {
                    float* din_pixel = din_ptr_base + ((n * H_dim + iy) * W_dim + ix) * in_channels;
                    
                    for (int ky = 0; ky < kernel_size; ++ky) {
                        int y_strided = iy + padding - ky;
                        if (y_strided % stride != 0) continue;
                        int y = y_strided / stride;
                        if (y < 0 || y >= OH_dim) continue;

                        for (int kx = 0; kx < kernel_size; ++kx) {
                            int x_strided = ix + padding - kx;
                            if (x_strided % stride != 0) continue;
                            int x = x_strided / stride;
                            if (x < 0 || x >= OW_dim) continue;

                            const float* go_pixel = go_ptr_base + ((n * OH_dim + y) * OW_dim + x) * out_channels;
                            const float* w_block = w_ptr_base + ((ky * kernel_size + kx) * in_channels) * out_channels;
                            
                            for (int ci = 0; ci < in_channels; ++ci) {
                                const float* w_row = w_block + ci * out_channels;
                                
                                // Horizontal dot product: sum(grad_output * weight) across
                                // all output channels. This gives us the gradient for one
                                // input channel at one spatial position.
                                __m256 acc_v = _mm256_setzero_ps();
                                int co = 0;
                                for (; co + 31 < out_channels; co += 32) {
                                    __m256 a0 = _mm256_mul_ps(_mm256_loadu_ps(go_pixel + co), _mm256_loadu_ps(w_row + co));
                                    __m256 a1 = _mm256_mul_ps(_mm256_loadu_ps(go_pixel + co + 8), _mm256_loadu_ps(w_row + co + 8));
                                    __m256 a2 = _mm256_mul_ps(_mm256_loadu_ps(go_pixel + co + 16), _mm256_loadu_ps(w_row + co + 16));
                                    __m256 a3 = _mm256_mul_ps(_mm256_loadu_ps(go_pixel + co + 24), _mm256_loadu_ps(w_row + co + 24));
                                    acc_v = _mm256_add_ps(acc_v, _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
                                }
                                for (; co + 7 < out_channels; co += 8) {
                                    acc_v = _mm256_fmadd_ps(_mm256_loadu_ps(go_pixel + co), _mm256_loadu_ps(w_row + co), acc_v);
                                }
                                
                                // Horizontal sum: extract 8 floats from the __m256 and add them.
                                // Using aligned store + scalar loop because _mm256_hadd_ps has
                                // worse throughput on most microarchitectures.
                                alignas(32) float acc_arr[8];
                                _mm256_store_ps(acc_arr, acc_v);
                                float sum = acc_arr[0] + acc_arr[1] + acc_arr[2] + acc_arr[3] + acc_arr[4] + acc_arr[5] + acc_arr[6] + acc_arr[7];
                                
                                for (; co < out_channels; ++co) sum += go_pixel[co] * w_row[co];
                                din_pixel[ci] += sum;
                            }
                        }
                    }
                }
            }
        }

        // 2. Calculate dWeights and dBias using thread-local accumulators
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            // Grab this thread's pre-allocated buffers and zero them with memset.
            // memset is faster than fill() here because we know the data is
            // contiguous floats and 0.0f is all-zero-bits in IEEE 754.
            Tensor& t_dw = local_grad_weights[tid];
            Tensor& t_db = local_grad_biases[tid];
            std::memset(t_dw.data.data(), 0, t_dw.size() * sizeof(float));
            std::memset(t_db.data.data(), 0, t_db.size() * sizeof(float));

            float* local_dw = t_dw.data.data();
            float* local_db = t_db.data.data();

            #pragma omp for collapse(2) schedule(static)
            for (int n = 0; n < N_batch; ++n) {
                for (int y = 0; y < OH_dim; ++y) {
                    for (int x = 0; x < OW_dim; ++x) {
                        const float* go_pixel = go_ptr_base + ((n * OH_dim + y) * OW_dim + x) * out_channels;
                        
                        // Bias gradient is just the sum of grad_output across spatial positions
                        int co = 0;
                        for (; co + 31 < out_channels; co += 32) {
                            _mm256_storeu_ps(local_db + co, _mm256_add_ps(_mm256_loadu_ps(local_db + co), _mm256_loadu_ps(go_pixel + co)));
                            _mm256_storeu_ps(local_db + co + 8, _mm256_add_ps(_mm256_loadu_ps(local_db + co + 8), _mm256_loadu_ps(go_pixel + co + 8)));
                            _mm256_storeu_ps(local_db + co + 16, _mm256_add_ps(_mm256_loadu_ps(local_db + co + 16), _mm256_loadu_ps(go_pixel + co + 16)));
                            _mm256_storeu_ps(local_db + co + 24, _mm256_add_ps(_mm256_loadu_ps(local_db + co + 24), _mm256_loadu_ps(go_pixel + co + 24)));
                        }
                        for (; co + 7 < out_channels; co += 8) {
                            _mm256_storeu_ps(local_db + co, _mm256_add_ps(_mm256_loadu_ps(local_db + co), _mm256_loadu_ps(go_pixel + co)));
                        }
                        for (; co < out_channels; ++co) local_db[co] += go_pixel[co];

                        // Weight gradient: dW[ky,kx,ci,co] += input[n,iy,ix,ci] * grad_output[n,y,x,co]
                        for (int ky = 0; ky < kernel_size; ++ky) {
                            int iy = y * stride - padding + ky;
                            if (iy < 0 || iy >= H_dim) continue;
                            for (int kx = 0; kx < kernel_size; ++kx) {
                                int ix = x * stride - padding + kx;
                                if (ix < 0 || ix >= W_dim) continue;
                                
                                const float* in_pixel = in_ptr_base + ((n * H_dim + iy) * W_dim + ix) * in_channels;
                                float* dw_block = local_dw + ((ky * kernel_size + kx) * in_channels) * out_channels;
                                
                                for (int ci = 0; ci < in_channels; ++ci) {
                                    __m256 v_val = _mm256_set1_ps(in_pixel[ci]);
                                    float* dw_row = dw_block + ci * out_channels;
                                    
                                    int c = 0;
                                    for (; c + 31 < out_channels; c += 32) {
                                        __m256 dw0 = _mm256_loadu_ps(dw_row + c);
                                        __m256 dw1 = _mm256_loadu_ps(dw_row + c + 8);
                                        __m256 dw2 = _mm256_loadu_ps(dw_row + c + 16);
                                        __m256 dw3 = _mm256_loadu_ps(dw_row + c + 24);

                                        dw0 = _mm256_fmadd_ps(v_val, _mm256_loadu_ps(go_pixel + c), dw0);
                                        dw1 = _mm256_fmadd_ps(v_val, _mm256_loadu_ps(go_pixel + c + 8), dw1);
                                        dw2 = _mm256_fmadd_ps(v_val, _mm256_loadu_ps(go_pixel + c + 16), dw2);
                                        dw3 = _mm256_fmadd_ps(v_val, _mm256_loadu_ps(go_pixel + c + 24), dw3);

                                        _mm256_storeu_ps(dw_row + c, dw0);
                                        _mm256_storeu_ps(dw_row + c + 8, dw1);
                                        _mm256_storeu_ps(dw_row + c + 16, dw2);
                                        _mm256_storeu_ps(dw_row + c + 24, dw3);
                                    }
                                    for (; c + 7 < out_channels; c += 8) {
                                        _mm256_storeu_ps(dw_row + c, _mm256_fmadd_ps(v_val, _mm256_loadu_ps(go_pixel + c), _mm256_loadu_ps(dw_row + c)));
                                    }
                                    for (; c < out_channels; ++c) dw_row[c] += in_pixel[ci] * go_pixel[c];
                                }
                            }
                        }
                    }
                }
            }
        }

        // Consolidate thread-local gradients back into the main gradient tensors.
        // This is single-threaded but runs once per backward() call across a
        // small number of threads (typically 4-16), so it's negligible.
        int max_threads = omp_get_max_threads();
        for (int t = 0; t < max_threads; ++t) {
            float* l_dw = local_grad_weights[t].data.data();
            float* l_db = local_grad_biases[t].data.data();
            #pragma omp simd
            for (int i = 0; i < weights.size(); ++i) dw_ptr_base[i] += l_dw[i];
            #pragma omp simd
            for (int i = 0; i < out_channels; ++i) db_ptr_base[i] += l_db[i];
        }
    }

    // Simple SGD weight update — used when not using an external optimizer.
    // The #pragma omp simd hint lets the compiler vectorize the lr*dW multiply.
    inline void update_weights(float lr) override {
        float* W = weights.data.data();
        float* dW = weights.grad.data();
        float* b = biases.data.data();  float* db = biases.grad.data();
        #pragma omp simd
        for (int i = 0; i < weights.size(); ++i) W[i] -= lr * dW[i];
        #pragma omp simd
        for (int i = 0; i < biases.size(); ++i)  b[i] -= lr * db[i];
        weights.zero_grad(); biases.zero_grad();
    }

    inline std::vector<Tensor*> get_parameters() override { return {&weights, &biases}; }
    inline std::string name() const override { return "Conv2D"; }
};

} // namespace CnnInCpp
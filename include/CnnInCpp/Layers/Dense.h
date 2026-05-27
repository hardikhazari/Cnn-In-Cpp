// ============================================================================
// Dense.h — Fully-connected (Dense/Linear) layer with AVX2/FMA GEMM.
//
// A Dense layer computes: output = input × weights + bias
// This is a standard General Matrix Multiply (GEMM), which is the same
// fundamental operation that Conv2D does after im2col. The difference is that
// Dense operates on 2D tensors directly — no spatial dimensions to deal with.
//
// Performance strategy:
// - Cache tiling with BLOCK_SIZE=64: the GEMM is split into 64×64 tiles that
//   fit in L2 cache (~256KB on most CPUs). Without tiling, the weight matrix
//   for a 8192→10 Dense layer is 320KB — fine. But for 8192→512 it's 16MB,
//   and random access across that blows past L2 into DRAM, which is ~10× slower.
//   By processing 64-wide blocks at a time, we keep the working set under ~32KB
//   (64*64*4 bytes = 16KB per tile), which fits comfortably in L1.
//
// - AVX2 FMA in the inner loop: each iteration does a broadcast-multiply-add
//   of one input element against 8 or 16 weight elements simultaneously.
//   _mm256_fmadd_ps does multiply+add in one cycle instead of two.
//
// - Thread-local gradient buffers: same pattern as Conv2D — each thread gets
//   its own grad_weights/grad_biases buffer to avoid synchronization overhead.
//
// - Adaptive threading: `if(N > 1)` skips OpenMP for single-sample inference.
// ============================================================================
#pragma once
#include "Layer.h"
#include <span>
#include <omp.h>
#include <immintrin.h>

namespace CnnInCpp {

class Dense : public Layer {
public:
    int    input_size, output_size;
    Tensor weights, biases;
    std::vector<Tensor> local_grad_weights, local_grad_biases;

    inline Dense(int in_sz, int out_sz) : input_size(in_sz), output_size(out_sz) {
        // Weight shape: [input_size, output_size] — output_size is contiguous.
        // This layout means sweeping across output neurons is a contiguous
        // memory access, perfect for AVX2 loads.
        weights = Tensor(in_sz, out_sz);
        biases  = Tensor(1, out_sz);
        int fan_in = in_sz;
        weights.fill_he_init(fan_in);
        biases.fill(0.0f);
    }

    inline void compile(const std::vector<std::vector<int>>& input_shapes) override {
        if (input_shapes.empty()) throw std::runtime_error("Dense: input_shapes is empty");
        int in_features = input_shapes[0].back(); 
        if (in_features != input_size) {
            throw std::runtime_error("Dense: dimension mismatch.");
        }
        const int N = input_shapes[0][0];
        output_buffer = Tensor(N, output_size);
        grad_input_buffer = Tensor(input_shapes[0]);

        // Pre-allocate one grad buffer per thread — same zero-allocation pattern as Conv2D
        int max_threads = omp_get_max_threads();
        local_grad_weights.clear(); local_grad_biases.clear();
        for (int t = 0; t < max_threads; ++t) {
            local_grad_weights.emplace_back(input_size, output_size);
            local_grad_biases.emplace_back(1, output_size);
        }
    }

    // ==========================================================================
    // FORWARD: output[i, j] = bias[j] + Σ_k input[i, k] * weight[k, j]
    //
    // The loop nest is: for each sample i, for each input tile k0, for each
    // output tile j0, do the broadcast-FMA. The tiling order (i → k → j) is
    // chosen so that:
    //   - i_row stays fixed across the k0/j0 loops (spatial locality on input)
    //   - w_row walks contiguously across the output dimension (streaming load)
    //   - o_row gets accumulated in-place (temporal locality)
    //
    // BLOCK_SIZE = 64 is deliberately chosen so that one tile of weights
    // (64 × 64 × 4 bytes = 16KB) fits in L1 data cache. Anything bigger and
    // we start evicting our own data before we're done with it.
    // ==========================================================================
inline Tensor& forward(const Tensor& input) override {
        cached_input_ptr = &input;
        const int N = input.shape[0];
        const int K = input_size;
        const int M = output_size;
        
        const float* inp = input.data.data();
        const float* W   = weights.data.data();
        const float* b   = biases.data.data();
        float* out = output_buffer.data.data();
        
        constexpr int BLOCK_SIZE = 64;
        
        // Parallelize over samples. For batch=16, each of the 16 threads
        // processes exactly 1 sample — perfect load balance, zero sharing.
        #pragma omp parallel for schedule(static) if(N > 1)
        for (int i = 0; i < N; ++i) {
            float* o_row = out + i * M;
            
            // Initialize output row from bias vector
            #ifdef __AVX2__
            int j = 0;
            for (; j + 7 < M; j += 8) {
                _mm256_storeu_ps(o_row + j, _mm256_loadu_ps(b + j));
            }
            for (; j < M; ++j) o_row[j] = b[j];
            #else
            #pragma omp simd
            for (int j = 0; j < M; ++j) o_row[j] = b[j];
            #endif

            const float* i_row = inp + i * K;
            // Cache-tiled GEMM: process in 64×64 blocks to keep working set in L1/L2.
            // The key insight: by blocking both K and M dimensions, we ensure that
            // the same 64-wide strip of output values gets updated by 64 consecutive
            // input values before we move to the next strip. This maximizes reuse of
            // the output row (which stays in L1) and the weight tile.
            for (int k0 = 0; k0 < K; k0 += BLOCK_SIZE) {
                int k_max = std::min(k0 + BLOCK_SIZE, K);
                for (int j0 = 0; j0 < M; j0 += BLOCK_SIZE) {
                    int j_max = std::min(j0 + BLOCK_SIZE, M);
                    for (int k = k0; k < k_max; ++k) {
                        float val = i_row[k];
                        const float* w_row = W + k * M;
                        
                        // Broadcast input[i,k] to all 8 lanes, then FMA against
                        // 16 weights at a time (2 × __m256 = 16 floats per iteration)
                        #ifdef __AVX2__
                        __m256 v_val = _mm256_set1_ps(val);
                        int jj = j0;
                        for (; jj + 15 < j_max; jj += 16) {
                            __m256 out_v0 = _mm256_loadu_ps(o_row + jj);
                            __m256 out_v1 = _mm256_loadu_ps(o_row + jj + 8);
                            out_v0 = _mm256_fmadd_ps(v_val, _mm256_loadu_ps(w_row + jj), out_v0);
                            out_v1 = _mm256_fmadd_ps(v_val, _mm256_loadu_ps(w_row + jj + 8), out_v1);
                            _mm256_storeu_ps(o_row + jj, out_v0);
                            _mm256_storeu_ps(o_row + jj + 8, out_v1);
                        }
                        for (; jj + 7 < j_max; jj += 8) {
                            __m256 out_v = _mm256_loadu_ps(o_row + jj);
                            out_v = _mm256_fmadd_ps(v_val, _mm256_loadu_ps(w_row + jj), out_v);
                            _mm256_storeu_ps(o_row + jj, out_v);
                        }
                        for (; jj < j_max; ++jj) o_row[jj] += val * w_row[jj];
                        #else
                        #pragma omp simd
                        for (int jj = j0; jj < j_max; ++jj) {
                            o_row[jj] += val * w_row[jj];
                        }
                        #endif
                    }
                }
            }
        }
        return output_buffer;
    }

    // ==========================================================================
    // BACKWARD: computes dInput, dWeights, and dBias simultaneously.
    //
    // dInput[i, k] = Σ_j grad_output[i, j] * weight[k, j]   (dot product across M)
    // dWeight[k, j] = Σ_i input[i, k] * grad_output[i, j]    (outer product across N)
    // dBias[j] = Σ_i grad_output[i, j]                        (column sum)
    //
    // All three computations are tiled the same way as forward (BLOCK_SIZE=64)
    // and use thread-local accumulation for dWeight/dBias to avoid contention.
    // ==========================================================================
    inline void backward(const Tensor& grad_output) override {
        const int N = grad_output.shape[0];
        const int K = input_size;
        const int M = output_size;

        const float* inp = cached_input_ptr->data.data();
        const float* go  = grad_output.data.data();
        const float* W   = weights.data.data();
        float*       di  = grad_input_buffer.data.data();
        float*       dW  = weights.grad.data();
        float*       db  = biases.grad.data();
        
        grad_input_buffer.fill(0.0f);
        constexpr int BLOCK_SIZE = 64;

        // Zero all thread-local buffers before the parallel region
        int max_threads = omp_get_max_threads();
        for (int t = 0; t < max_threads; ++t) {
            local_grad_weights[t].fill(0.0f);
            local_grad_biases[t].fill(0.0f);
        }

        #pragma omp parallel for schedule(static)
        for (int i0 = 0; i0 < N; i0 += BLOCK_SIZE) {
            int i_max = std::min(i0 + BLOCK_SIZE, N);
            int tid = omp_get_thread_num();
            float* local_dW = local_grad_weights[tid].data.data();
            float* local_db = local_grad_biases[tid].data.data();

            // --- dInput computation ---
            // For each input element (i, k): sum across output dimension j
            // of grad_output[i, j] * weight[k, j]. This is the "transposed GEMM".
            for (int k0 = 0; k0 < K; k0 += BLOCK_SIZE) {
                int k_max = std::min(k0 + BLOCK_SIZE, K);
                for (int j0 = 0; j0 < M; j0 += BLOCK_SIZE) {
                    int j_max = std::min(j0 + BLOCK_SIZE, M);
                    for (int i = i0; i < i_max; ++i) {
                        float* di_row = di + i * K;
                        const float* go_row = go + i * M;
                        for (int k = k0; k < k_max; ++k) {
                            const float* w_row = W + k * M;
                            float acc = 0.0f;
                            #ifdef __AVX2__
                            __m256 v_acc = _mm256_setzero_ps();
                            int j = j0;
                            for (; j + 31 < j_max; j += 32) {
                                __m256 go0 = _mm256_loadu_ps(go_row + j);
                                __m256 w0 = _mm256_loadu_ps(w_row + j);
                                v_acc = _mm256_fmadd_ps(go0, w0, v_acc);
                                
                                __m256 go1 = _mm256_loadu_ps(go_row + j + 8);
                                __m256 w1 = _mm256_loadu_ps(w_row + j + 8);
                                v_acc = _mm256_fmadd_ps(go1, w1, v_acc);
                                
                                __m256 go2 = _mm256_loadu_ps(go_row + j + 16);
                                __m256 w2 = _mm256_loadu_ps(w_row + j + 16);
                                v_acc = _mm256_fmadd_ps(go2, w2, v_acc);
                                
                                __m256 go3 = _mm256_loadu_ps(go_row + j + 24);
                                __m256 w3 = _mm256_loadu_ps(w_row + j + 24);
                                v_acc = _mm256_fmadd_ps(go3, w3, v_acc);
                            }
                            for (; j + 7 < j_max; j += 8) {
                                __m256 go0 = _mm256_loadu_ps(go_row + j);
                                __m256 w0 = _mm256_loadu_ps(w_row + j);
                                v_acc = _mm256_fmadd_ps(go0, w0, v_acc);
                            }
                            alignas(32) float acc_arr[8];
                            _mm256_store_ps(acc_arr, v_acc);
                            for (int a = 0; a < 8; ++a) acc += acc_arr[a];
                            for (; j < j_max; ++j) {
                                acc += go_row[j] * w_row[j];
                            }
                            #else
                            #pragma omp simd reduction(+:acc)
                            for (int j = j0; j < j_max; ++j) {
                                acc += go_row[j] * w_row[j];
                            }
                            #endif
                            di_row[k] += acc;
                        }
                    }
                }
            }

            // --- dWeights computation ---
            // dW[k, j] += input[i, k] * grad_output[i, j]  (outer product, accumulated)
            for (int k0 = 0; k0 < K; k0 += BLOCK_SIZE) {
                int k_max = std::min(k0 + BLOCK_SIZE, K);
                for (int j0 = 0; j0 < M; j0 += BLOCK_SIZE) {
                    int j_max = std::min(j0 + BLOCK_SIZE, M);
                    for (int k = k0; k < k_max; ++k) {
                        float* dw_row = local_dW + k * M;
                        for (int i = i0; i < i_max; ++i) {
                            float val = inp[i * K + k]; // inp^T
                            const float* go_row = go + i * M;
                            #ifdef __AVX2__
                            __m256 v_val = _mm256_set1_ps(val);
                            int j = j0;
                            for (; j + 7 < j_max; j += 8) {
                                __m256 h_v = _mm256_loadu_ps(dw_row + j);
                                __m256 g_v = _mm256_loadu_ps(go_row + j);
                                h_v = _mm256_fmadd_ps(v_val, g_v, h_v);
                                _mm256_storeu_ps(dw_row + j, h_v);
                            }
                            for (; j < j_max; ++j) {
                                dw_row[j] += val * go_row[j];
                            }
                            #else
                            #pragma omp simd
                            for (int j = j0; j < j_max; ++j) {
                                dw_row[j] += val * go_row[j];
                            }
                            #endif
                        }
                    }
                }
            }

            // --- dBias computation ---
            // dBias[j] = sum of grad_output[:, j] across samples in this thread's chunk
            for (int i = i0; i < i_max; ++i) {
                const float* go_row = go + i * M;
                #ifdef __AVX2__
                int j = 0;
                for (; j + 7 < M; j += 8) {
                    __m256 db_v = _mm256_loadu_ps(local_db + j);
                    __m256 go_v = _mm256_loadu_ps(go_row + j);
                    db_v = _mm256_add_ps(db_v, go_v);
                    _mm256_storeu_ps(local_db + j, db_v);
                }
                for (; j < M; ++j) {
                    local_db[j] += go_row[j];
                }
                #else
                #pragma omp simd
                for (int j = 0; j < M; ++j) {
                    local_db[j] += go_row[j];
                }
                #endif
            }
        }

        // Reduce thread-local gradients into the main gradient tensors
        for (int t = 0; t < max_threads; ++t) {
            const float* l_dW = local_grad_weights[t].data.data();
            const float* l_db = local_grad_biases[t].data.data();
            #pragma omp simd
            for (int i = 0; i < K * M; ++i) dW[i] += l_dW[i];
            #pragma omp simd
            for (int i = 0; i < M; ++i) db[i] += l_db[i];
        }
    }

    inline void update_weights(float lr) override {
        float* W  = weights.data.data(); float* dW = weights.grad.data();
        float* b  = biases.data.data();  float* db = biases.grad.data();
        #pragma omp simd
        for (int i = 0; i < weights.size(); ++i) W[i] -= lr * dW[i];
        #pragma omp simd
        for (int i = 0; i < biases.size();  ++i) b[i] -= lr * db[i];
        weights.zero_grad(); biases.zero_grad();
    }

    inline std::vector<Tensor*> get_parameters() override { return {&weights, &biases}; }
    inline std::string name() const override { return "Dense"; }
};

} // namespace CnnInCpp

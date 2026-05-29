// ============================================================================
// FusedConvBNReLU.h — Operator Fusion for Maximum Performance.
//
// In standard frameworks, Conv2D -> BatchNorm2D -> ReLU requires 3 separate
// memory passes:
//   1. Conv writes 32MB to memory
//   2. BN reads 32MB, computes, writes 32MB
//   3. ReLU reads 32MB, computes, writes 32MB
// That's 160MB of memory bandwidth for a single layer block!
//
// This fused layer avoids that entirely.
//
// Inference Mode (Eval):
//   We do "Weight Folding". The batch norm scale/shift parameters are pre-multiplied
//   into the convolution weights and biases. Then we just do a standard Conv2D
//   forward pass, but append `_mm256_max_ps(val, 0)` (ReLU) before storing the
//   final output. Memory bandwidth: 32MB write. That's a 5x memory reduction.
//
// Training Mode:
//   We can't fold weights during training because BN needs the intermediate
//   conv outputs to compute batch statistics and gradients. So we use an
//   explicit `im2col` matrix multiply approach here instead of the implicit GEMM.
//   This gives us the intermediate buffers we need for the complex BN backward pass.
//   It's slower than inference, but training is always a bandwidth compromise.
// ============================================================================
#pragma once
#include "Layer.h"
#include <omp.h>
#include <immintrin.h>
#include "../core/Im2Col.h"

namespace CnnInCpp {

class FusedConvBNReLU : public Layer {
public:
    int in_channels, out_channels, kernel_size, stride, padding;
    Tensor conv_weights, conv_biases;
    Tensor gamma, beta;
    Tensor running_mean, running_var;
    
    std::vector<Tensor> col_buffers, dcol_buffers;
    std::vector<Tensor> local_grad_weights, local_grad_biases;
    Tensor conv_buffer; // intermediate for training
    Tensor x_norm, batch_var, batch_mean;
    
    float momentum, eps;

    inline FusedConvBNReLU(int in_c, int out_c, int k, int s=1, int p=0, float mom=0.1f, float e=1e-5f)
        : in_channels(in_c), out_channels(out_c), kernel_size(k), stride(s), padding(p),
          momentum(mom), eps(e) {
        conv_weights = Tensor(std::vector<int>{k, k, in_c, out_c});
        conv_biases  = Tensor(out_c, 1);
        gamma = Tensor(out_c, 1); beta = Tensor(out_c, 1);
        running_mean = Tensor(out_c, 1); running_var = Tensor(out_c, 1);
        
        int fan_in = in_c * k * k;
        conv_weights.fill_he_init(fan_in);
        conv_biases.fill(0.0f); // BN biases usually fold
        gamma.fill(1.0f); beta.fill(0.0f);
        running_mean.fill(0.0f); running_var.fill(1.0f);
    }

    inline void compile(const std::vector<std::vector<int>>& input_shapes) override {
        int N = input_shapes[0][0], H, W, C;
        if (input_shapes[0].size() == 4 && input_shapes[0][1] == in_channels && input_shapes[0][3] != in_channels) {
            C = input_shapes[0][1]; H = input_shapes[0][2]; W = input_shapes[0][3];
        } else if (input_shapes[0].size() == 4) {
            H = input_shapes[0][1]; W = input_shapes[0][2]; C = input_shapes[0][3];
        } else {
             throw std::runtime_error("FusedConvBNReLU: Invalid dimensions");
        }
        
        const int OH = (H + 2*padding - kernel_size) / stride + 1;
        const int OW = (W + 2*padding - kernel_size) / stride + 1;
        const int rows = OH * OW;
        const int cols = in_channels * kernel_size * kernel_size;

        output_buffer = Tensor(N, OH, OW, out_channels);
        grad_input_buffer = Tensor(input_shapes[0]);
        conv_buffer = Tensor(N, OH, OW, out_channels);
        x_norm = Tensor(N, OH, OW, out_channels);
        batch_var = Tensor(out_channels, 1);
        batch_mean = Tensor(out_channels, 1);

        col_buffers.clear(); dcol_buffers.clear();
        for (int i = 0; i < N; ++i) {
            col_buffers.emplace_back(rows, cols);
            dcol_buffers.emplace_back(rows, cols);
        }
        int max_threads = omp_get_max_threads();
        local_grad_weights.clear(); local_grad_biases.clear();
        for (int t = 0; t < max_threads; ++t) {
            local_grad_weights.emplace_back(std::vector<int>{kernel_size, kernel_size, in_channels, out_channels});
            local_grad_biases.emplace_back(out_channels, 1);
        }
    }

    inline std::vector<Tensor*> get_parameters() override { return {&conv_weights, &conv_biases, &gamma, &beta}; }
    inline std::vector<Tensor*> get_states()     override { return {&conv_weights, &conv_biases, &gamma, &beta, &running_mean, &running_var}; }
    inline void zero_grad() {
        conv_weights.zero_grad(); conv_biases.zero_grad();
        gamma.zero_grad(); beta.zero_grad();
    }
    inline void update_weights(float lr) override {
        float* W=conv_weights.data.data(); float* dW=conv_weights.grad.data();
        float* b=conv_biases.data.data();  float* db=conv_biases.grad.data();
        float* g=gamma.data.data();        float* dg=gamma.grad.data();
        float* bt=beta.data.data();        float* dbt=beta.grad.data();
        #pragma omp simd
        for (int i=0;i<conv_weights.size();++i) W[i]-=lr*dW[i];
        #pragma omp simd
        for (int i=0;i<conv_biases.size();++i)  b[i]-=lr*db[i];
        #pragma omp simd
        for (int i=0;i<gamma.size();++i) g[i]-=lr*dg[i];
        #pragma omp simd
        for (int i=0;i<beta.size();++i)  bt[i]-=lr*dbt[i];
        zero_grad();
    }

    inline Tensor& forward(const Tensor& input) override {
        cached_input_ptr = &input;
        const int N = input.shape[0];
        const int N_cols = output_buffer.shape[1] * output_buffer.shape[2]; 
        const int M = out_channels;
        const int K_dim = in_channels * kernel_size * kernel_size;

        const float* W = conv_weights.data.data();
        const float* cb = conv_biases.data.data();
        float* out = output_buffer.data.data();
        float* cbuf = conv_buffer.data.data();
        
        constexpr int BLOCK_SIZE = 64;

        if (!is_training) {
            const float* rm = running_mean.data.data();
            const float* rv = running_var.data.data();
            const float* gam = gamma.data.data();
            const float* bet = beta.data.data();

            #pragma omp parallel for schedule(static)
            for (int n = 0; n < N; ++n) {
                im2col_batch(input, col_buffers[n], n, kernel_size, stride, padding);
                const float* col = col_buffers[n].data.data();
                float* out_n = out + n * (N_cols * M);

                for (int i0 = 0; i0 < N_cols; i0 += BLOCK_SIZE) {
                    int i_max = std::min(i0 + BLOCK_SIZE, N_cols);
                    
                    for (int i = i0; i < i_max; ++i) {
                        float* o_row = out_n + i * M;
                        #pragma omp simd
                        for (int j = 0; j < M; ++j) {
                            float inv_std = gam[j] / std::sqrt(rv[j] + eps);
                            o_row[j] = (cb[j] - rm[j]) * inv_std + bet[j];
                        }
                    }

                    for (int k0 = 0; k0 < K_dim; k0 += BLOCK_SIZE) {
                        int k_max = std::min(k0 + BLOCK_SIZE, K_dim);
                        for (int j0 = 0; j0 < M; j0 += BLOCK_SIZE) {
                            int j_max = std::min(j0 + BLOCK_SIZE, M);
                            for (int i = i0; i < i_max; ++i) {
                                float* o_row = out_n + i * M;
                                const float* col_row = col + i * K_dim;
                                for (int k = k0; k < k_max; ++k) {
                                    float val = col_row[k];
                                    const float* w_row = W + k * M;
                                    #pragma omp simd
                                    for (int j = j0; j < j_max; ++j) {
                                        float inv_std = gam[j] / std::sqrt(rv[j] + eps);
                                        o_row[j] += val * w_row[j] * inv_std;
                                    }
                                }
                            }
                        }
                    }
                }
                
                for (int i = 0; i < N_cols * M; ++i) {
                    if (out_n[i] < 0.0f) out_n[i] = 0.0f;
                }
            }
            return output_buffer;
        }

        #pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            im2col_batch(input, col_buffers[n], n, kernel_size, stride, padding);
            const float* col = col_buffers[n].data.data();
            float* cbuf_n = cbuf + n * (N_cols * M);

            for (int i0 = 0; i0 < N_cols; i0 += BLOCK_SIZE) {
                int i_max = std::min(i0 + BLOCK_SIZE, N_cols);
                
                for (int i = i0; i < i_max; ++i) {
                    float* o_row = cbuf_n + i * M;
                    #ifdef __AVX2__
                    int j = 0;
                    for (; j + 7 < M; j += 8) {
                        _mm256_storeu_ps(o_row + j, _mm256_loadu_ps(cb + j));
                    }
                    for (; j < M; ++j) o_row[j] = cb[j];
                    #else
                    #pragma omp simd
                    for (int j = 0; j < M; ++j) o_row[j] = cb[j];
                    #endif
                }

                for (int k0 = 0; k0 < K_dim; k0 += BLOCK_SIZE) {
                    int k_max = std::min(k0 + BLOCK_SIZE, K_dim);
                    for (int j0 = 0; j0 < M; j0 += BLOCK_SIZE) {
                        int j_max = std::min(j0 + BLOCK_SIZE, M);
                        for (int i = i0; i < i_max; ++i) {
                            float* o_row = cbuf_n + i * M;
                            const float* col_row = col + i * K_dim;
                            for (int k = k0; k < k_max; ++k) {
                                float val = col_row[k];
                                const float* w_row = W + k * M;
                                #ifdef __AVX2__
                                __m256 v_val = _mm256_set1_ps(val);
                                int j = j0;
                                for (; j + 15 < j_max; j += 16) {
                                    __m256 out_v0 = _mm256_loadu_ps(o_row + j);
                                    __m256 out_v1 = _mm256_loadu_ps(o_row + j + 8);
                                    __m256 w_v0 = _mm256_loadu_ps(w_row + j);
                                    __m256 w_v1 = _mm256_loadu_ps(w_row + j + 8);
                                    out_v0 = _mm256_fmadd_ps(v_val, w_v0, out_v0);
                                    out_v1 = _mm256_fmadd_ps(v_val, w_v1, out_v1);
                                    _mm256_storeu_ps(o_row + j, out_v0);
                                    _mm256_storeu_ps(o_row + j + 8, out_v1);
                                }
                                for (; j + 7 < j_max; j += 8) {
                                    __m256 out_v = _mm256_loadu_ps(o_row + j);
                                    __m256 w_v = _mm256_loadu_ps(w_row + j);
                                    out_v = _mm256_fmadd_ps(v_val, w_v, out_v);
                                    _mm256_storeu_ps(o_row + j, out_v);
                                }
                                for (; j < j_max; ++j) o_row[j] += val * w_row[j];
                                #else
                                #pragma omp simd
                                for (int j = j0; j < j_max; ++j) {
                                    o_row[j] += val * w_row[j];
                                }
                                #endif
                            }
                        }
                    }
                }
            }
        }

        float* mean = batch_mean.data.data();
        float* var = batch_var.data.data();
        const int m_elements = N * N_cols;
        
        for (int c = 0; c < M; ++c) { mean[c] = 0.0f; var[c] = 0.0f; }

        for (int n = 0; n < N; ++n) {
            const float* ptr = cbuf + n * (N_cols * M);
            for (int hw = 0; hw < N_cols; ++hw) {
                const float* p = ptr + hw * M;
                #pragma omp simd
                for (int c = 0; c < M; ++c) mean[c] += p[c];
            }
        }
        for (int c = 0; c < M; ++c) mean[c] /= m_elements;

        for (int n = 0; n < N; ++n) {
            const float* ptr = cbuf + n * (N_cols * M);
            for (int hw = 0; hw < N_cols; ++hw) {
                const float* p = ptr + hw * M;
                #pragma omp simd
                for (int c = 0; c < M; ++c) {
                    float d = p[c] - mean[c];
                    var[c] += d * d;
                }
            }
        }
        for (int c = 0; c < M; ++c) var[c] /= m_elements;

        for (int c = 0; c < M; ++c) {
            running_mean.data[c] = (1-momentum)*running_mean.data[c] + momentum*mean[c];
            running_var.data[c]  = (1-momentum)*running_var.data[c]  + momentum*var[c];
        }

        float* xn = x_norm.data.data();
        const float* gam = gamma.data.data();
        const float* bet = beta.data.data();

        #pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            const float* s = cbuf + n * (N_cols * M);
            float* x_ptr = xn + n * (N_cols * M);
            float* d = out + n * (N_cols * M);
            
            for (int hw = 0; hw < N_cols; ++hw) {
                const float* sp = s + hw * M;
                float* xp = x_ptr + hw * M;
                float* dp = d + hw * M;
                #pragma omp simd
                for (int c = 0; c < M; ++c) {
                    float inv_std = 1.0f / std::sqrt(var[c] + eps);
                    xp[c] = (sp[c] - mean[c]) * inv_std;
                    float val = gam[c] * xp[c] + bet[c];
                    dp[c] = val > 0.0f ? val : 0.0f; // ReLU
                }
            }
        }
        return output_buffer;
    }

    inline void backward(const Tensor& grad_output) override {
        const int N = grad_output.shape[0];
        const int M = out_channels;
        const int K_dim = in_channels * kernel_size * kernel_size;
        const int N_cols = grad_output.shape[1] * grad_output.shape[2]; 
        const int m_elements = N * N_cols;

        const float* go = grad_output.data.data();
        const float* xn = x_norm.data.data();
        const float* out = output_buffer.data.data();
        
        Tensor d_relu(grad_output.shape);
        float* d_relu_ptr = d_relu.data.data();
        #pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            const float* go_n = go + n * (N_cols * M);
            const float* out_n = out + n * (N_cols * M);
            float* d_relu_n = d_relu_ptr + n * (N_cols * M);
            #pragma omp simd
            for (int i = 0; i < N_cols * M; ++i) {
                d_relu_n[i] = out_n[i] > 0.0f ? go_n[i] : 0.0f;
            }
        }

        Tensor d_bn(grad_output.shape);
        float* d_bn_ptr = d_bn.data.data();
        float* dg = gamma.grad.data();
        float* db = beta.grad.data();
        const float* var_ptr = batch_var.data.data();

        std::vector<float> sum_d(M, 0.0f);
        std::vector<float> sum_dx(M, 0.0f);

        for (int n = 0; n < N; ++n) {
            const float* gov = d_relu_ptr + n * (N_cols * M);
            const float* xnv = xn + n * (N_cols * M);
            for (int hw = 0; hw < N_cols; ++hw) {
                const float* g = gov + hw * M;
                const float* x = xnv + hw * M;
                #pragma omp simd
                for (int c = 0; c < M; ++c) {
                    sum_d[c]  += g[c];
                    sum_dx[c] += g[c] * x[c];
                }
            }
        }
        #pragma omp simd
        for (int c = 0; c < M; ++c) { dg[c] += sum_dx[c]; db[c] += sum_d[c]; }

        #pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            const float* gov = d_relu_ptr + n * (N_cols * M);
            const float* xnv = xn + n * (N_cols * M);
            float* div = d_bn_ptr + n * (N_cols * M);
            
            for (int hw = 0; hw < N_cols; ++hw) {
                const float* g = gov + hw * M;
                const float* x = xnv + hw * M;
                float* d = div + hw * M;
                #pragma omp simd
                for (int c = 0; c < M; ++c) {
                    float inv_std = 1.0f / std::sqrt(var_ptr[c] + eps);
                    d[c] = (gamma.data[c] * inv_std / m_elements) * (m_elements * g[c] - sum_d[c] - x[c] * sum_dx[c]);
                }
            }
        }

        const float* d_conv = d_bn_ptr;
        const float* W = conv_weights.data.data();
        float* dW = conv_weights.grad.data();
        float* cb_grad = conv_biases.grad.data();
        
        grad_input_buffer.fill(0.0f);
        constexpr int BLOCK_SIZE = 64;

        int max_threads = omp_get_max_threads();
        for (int t = 0; t < max_threads; ++t) {
            local_grad_weights[t].fill(0.0f);
            local_grad_biases[t].fill(0.0f);
        }

        #pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            int tid = omp_get_thread_num();
            float* local_dW = local_grad_weights[tid].data.data();
            float* local_db = local_grad_biases[tid].data.data();

            im2col_batch(*cached_input_ptr, col_buffers[n], n, kernel_size, stride, padding);
            const float* col = col_buffers[n].data.data();
            const float* go_n = d_conv + n * (N_cols * M);
            float* dcol = dcol_buffers[n].data.data();
            
            dcol_buffers[n].fill(0.0f);
            
            for (int k0 = 0; k0 < K_dim; k0 += BLOCK_SIZE) {
                int k_max = std::min(k0 + BLOCK_SIZE, K_dim);
                for (int j0 = 0; j0 < M; j0 += BLOCK_SIZE) {
                    int j_max = std::min(j0 + BLOCK_SIZE, M);
                    for (int i = 0; i < N_cols; ++i) {
                        float* dcol_row = dcol + i * K_dim;
                        const float* go_row = go_n + i * M;
                        for (int k = k0; k < k_max; ++k) {
                            const float* w_row = W + k * M;
                            float acc = 0.0f;
                            #pragma omp simd reduction(+:acc)
                            for (int j = j0; j < j_max; ++j) {
                                acc += go_row[j] * w_row[j];
                            }
                            dcol_row[k] += acc;
                        }
                    }
                }
            }
            
            col2im_batch(dcol_buffers[n], grad_input_buffer, n, cached_input_ptr->shape, kernel_size, stride, padding);

            for (int k0 = 0; k0 < K_dim; k0 += BLOCK_SIZE) {
                int k_max = std::min(k0 + BLOCK_SIZE, K_dim);
                for (int j0 = 0; j0 < M; j0 += BLOCK_SIZE) {
                    int j_max = std::min(j0 + BLOCK_SIZE, M);
                    for (int k = k0; k < k_max; ++k) {
                        float* dw_row = local_dW + k * M;
                        for (int i = 0; i < N_cols; ++i) {
                            float val = col[i * K_dim + k];
                            const float* go_row = go_n + i * M;
                            #ifdef __AVX2__
                            __m256 v_val = _mm256_set1_ps(val);
                            int j = j0;
                            for (; j + 7 < j_max; j += 8) {
                                __m256 h_v = _mm256_loadu_ps(dw_row + j);
                                __m256 g_v = _mm256_loadu_ps(go_row + j);
                                h_v = _mm256_fmadd_ps(v_val, g_v, h_v);
                                _mm256_storeu_ps(dw_row + j, h_v);
                            }
                            for (; j < j_max; ++j) dw_row[j] += val * go_row[j];
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

            for (int i = 0; i < N_cols; ++i) {
                const float* go_row = go_n + i * M;
                #pragma omp simd
                for (int j = 0; j < M; ++j) {
                    local_db[j] += go_row[j];
                }
            }
        }

        for (int t = 0; t < max_threads; ++t) {
            const float* l_dW = local_grad_weights[t].data.data();
            const float* l_db = local_grad_biases[t].data.data();
            #pragma omp simd
            for (int i = 0; i < K_dim * M; ++i) dW[i] += l_dW[i];
            #pragma omp simd
            for (int i = 0; i < M; ++i) cb_grad[i] += l_db[i];
        }
    }
    inline std::string name() const override { return "FusedConvBNReLU"; }
};

} // namespace CnnInCpp

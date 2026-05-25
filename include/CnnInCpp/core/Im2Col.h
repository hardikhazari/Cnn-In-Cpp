// ============================================================================
// Im2Col.h — Image-to-Column transform for convolution.
//
// Convolution in neural networks is mathematically equivalent to a matrix
// multiply, but the input has to be rearranged first. The im2col transform
// "unrolls" each receptive field (the kernel_size × kernel_size × C patch the
// kernel slides over) into a single row of a 2D matrix. Once unrolled, the
// entire convolution becomes: output = col_matrix × weight_matrix — a standard
// GEMM that we can throw AVX2/FMA at.
//
// This file provides the im2col/col2im pair:
// - im2col_batch: extracts patches from one image in the batch into a column
//   matrix. Used in forward pass of FusedConvBNReLU (the explicit im2col path).
// - col2im_batch: the transpose operation — scatters gradient from the column
//   matrix back into the spatial input gradient. Used in backward pass.
//
// Note: Conv2D does NOT use im2col at all — it does an "implicit GEMM" where
// the multiply-accumulate happens directly from the input tensor without ever
// materializing the unrolled matrix. This saves a full buffer allocation and
// a memory copy. FusedConvBNReLU uses explicit im2col because the fused
// backward pass needs the column buffers for both gradient computation paths.
//
// Layout assumption: input is NHWC (batch, height, width, channels).
// The channel-last layout means consecutive channels are adjacent in memory,
// which is perfect for the innermost `for (c = 0; c < C; ++c)` loop — it
// becomes a contiguous memcpy-like operation that auto-vectorizes trivially.
// ============================================================================
#pragma once
#include "Tensor.h"
#include <vector>

namespace CnnInCpp {

// im2col_batch — Extract one image (batch index `b`) into column format.
//
// Output layout: col is (OH*OW) × (kernel_size² * C)
// Each row = one flattened receptive field patch.
//
// The bounds checking (iy>=0 && iy<H && ix>=0 && ix<W) handles padding:
// out-of-bounds positions get zero-filled, which is zero-padding by definition.
// The #pragma omp simd on the channel copy loop tells the compiler to vectorize
// what is essentially a small memcpy of C floats.
inline void im2col_batch(const Tensor& input, Tensor& col, int b, int kernel_size, int stride=1, int padding=0) {
    const int H  = input.shape[1], W  = input.shape[2], C  = input.shape[3];
    const int OH = (H + 2*padding - kernel_size) / stride + 1;
    const int OW = (W + 2*padding - kernel_size) / stride + 1;
    const int cols = kernel_size * kernel_size * C;

    // Raw pointer arithmetic — we precompute the base offset for batch `b`
    // once and then stride through the spatial dimensions. This avoids calling
    // Tensor::operator() which would recompute the 4D index every access.
    const float* src = input.data.data() + b * (H * W * C);
    float*       dst = col.data.data();

    for (int y = 0; y < OH; ++y) {
        for (int x = 0; x < OW; ++x) {
            int out_idx = y * OW + x;
            float* dst_row = dst + out_idx * cols;
            
            for (int ky = 0; ky < kernel_size; ++ky) {
                int iy = y*stride - padding + ky;
                for (int kx = 0; kx < kernel_size; ++kx) {
                    int ix = x*stride - padding + kx;
                    int patch_idx = (ky * kernel_size + kx) * C;
                    
                    if (iy>=0 && iy<H && ix>=0 && ix<W) {
                        const float* src_row = src + (iy * W + ix) * C;
                        #pragma omp simd
                        for (int c = 0; c < C; ++c) {
                            dst_row[patch_idx + c] = src_row[c];
                        }
                    } else {
                        // Zero-padding: out-of-bounds spatial positions → 0.0
                        #pragma omp simd
                        for (int c = 0; c < C; ++c) {
                            dst_row[patch_idx + c] = 0.0f;
                        }
                    }
                }
            }
        }
    }
}

// col2im_batch — The backward-pass counterpart of im2col.
//
// Scatters gradients from column format back into the spatial input gradient.
// This is an accumulation (+= not =) because multiple output positions may
// have read from the same input position (when stride < kernel_size, which is
// the common case). The += is what makes this the transpose of im2col.
//
// Note: this function is called per-batch-element inside an omp parallel for,
// so no additional parallelism is needed here.
inline void col2im_batch(const Tensor& col, Tensor& d_in, int b, const std::vector<int>& in_shape,
                     int kernel_size, int stride=1, int padding=0) {
    const int H  = in_shape[1], W  = in_shape[2], C  = in_shape[3];
    const int OH = (H + 2*padding - kernel_size) / stride + 1;
    const int OW = (W + 2*padding - kernel_size) / stride + 1;
    const int cols = kernel_size * kernel_size * C;

    const float* src = col.data.data();
    float*       dst = d_in.data.data() + b * (H * W * C);

    for (int y = 0; y < OH; ++y) {
        for (int x = 0; x < OW; ++x) {
            int out_idx = y * OW + x;
            const float* src_row = src + out_idx * cols;
            
            for (int ky = 0; ky < kernel_size; ++ky) {
                int iy = y*stride - padding + ky;
                for (int kx = 0; kx < kernel_size; ++kx) {
                    int ix = x*stride - padding + kx;
                    int patch_idx = (ky * kernel_size + kx) * C;
                    
                    if (iy>=0 && iy<H && ix>=0 && ix<W) {
                        float* dst_row = dst + (iy * W + ix) * C;
                        // += because overlapping patches both contribute
                        // gradient to the same input location
                        #pragma omp simd
                        for (int c = 0; c < C; ++c) {
                            dst_row[c] += src_row[patch_idx + c];
                        }
                    }
                }
            }
        }
    }
}

// Convenience wrapper — creates the column buffer and fills it for batch 0.
// Only used in simple/test paths; the real hot path in FusedConvBNReLU
// pre-allocates col_buffers[] during compile() and reuses them.
inline Tensor im2col(const Tensor& input, int kernel_size, int stride=1, int padding=0) {
    const int H  = input.shape[1], W  = input.shape[2], C = input.shape[3];
    const int OH = (H + 2*padding - kernel_size) / stride + 1;
    const int OW = (W + 2*padding - kernel_size) / stride + 1;
    Tensor col(OH * OW, kernel_size * kernel_size * C);
    im2col_batch(input, col, 0, kernel_size, stride, padding);
    return col;
}

} // namespace CnnInCpp

// ============================================================================
// Tensor.h — The fundamental data container for everything in CnnInCpp.
//
// Every weight matrix, every activation map, every gradient buffer is a Tensor.
// This is intentionally a simple, flat, contiguous-memory structure — not a
// general-purpose N-dimensional array with broadcasting or lazy evaluation.
// The design philosophy: keep the data layout dead simple so the hot-path code
// in Conv2D/Dense/etc. can reason about memory access patterns and use raw
// pointers + SIMD without fighting abstractions.
//
// Here's the thing nobody tells you about high-performance tensor libraries:
// the #1 performance killer isn't the math — it's accidental memory allocation.
// A single malloc() in a forward pass can cost more than the entire convolution
// if it triggers a kernel-mode page fault or contends on the heap lock across
// OpenMP threads. That's why NoInitAllocator exists below, and that's why the
// Model pre-allocates every buffer during compile() so the hot path is purely
// compute-bound.
//
// Layout: NHWC (batch, height, width, channels) for 4D tensors. This is
// channel-last, which is more SIMD-friendly than NCHW because consecutive
// channel values sit in adjacent memory addresses, letting us load 8 channels
// at once with a single _mm256_loadu_ps().
// ============================================================================
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>
#include <memory>

namespace CnnInCpp {

// ============================================================================
// NoInitAllocator — Why we skip zero-initialization and why it's safe.
//
// std::vector<float> normally zero-fills on resize(). For a 128-batch tensor
// of shape [128, 32, 32, 64], that's 128*32*32*64 = 8M floats = 32MB of
// memset(0) — pure waste when the very next operation writes every element
// anyway (e.g., forward() fills the output buffer completely).
//
// This allocator intercepts the default-construct path (the no-args construct()
// call) and does literally nothing, leaving memory in whatever state the OS
// gave us. This drops allocation time from ~200ms to ~1ms for large buffers.
//
// The safety contract: every buffer allocated with NoInitAllocator MUST be
// fully written before it's read. If you ever read from an uninitialized
// element, you'll get garbage (not a crash, but wrong results). This is
// enforced by design — forward() writes every output element, backward()
// writes every gradient element, and compile() is the only place that calls
// resize().
//
// Heads up — the gradients (.grad) are the ONE exception: they use
// resize(size, 0.0f) in require_grad() because gradients accumulate across
// backward passes and must start from zero. That explicit 0.0f argument
// bypasses our no-init trick because sizeof...(Args) > 0.
// ============================================================================
template <typename T>
struct NoInitAllocator : std::allocator<T> {
    template <typename U, typename... Args>
    void construct(U* p, Args&&... args) {
        if constexpr (sizeof...(Args) == 0) {
            // Do nothing! 
        } else {
            ::new(static_cast<void*>(p)) U(std::forward<Args>(args)...);
        }
    }
};

// ============================================================================
// QuantizedTensor — INT8 representation for inference-only workloads.
//
// Stores weights/activations as 8-bit integers with a scale+zero_point affine
// mapping: float_value ≈ scale * (int8_value - zero_point). This cuts memory
// by 4× and can enable integer SIMD paths (VNNI on recent Intel). Currently
// used for model export/inspection — the main compute path still uses FP32.
// ============================================================================
struct QuantizedTensor {
  std::vector<int> shape;
  std::vector<int8_t> data;
  float scale = 1.0f;
  int32_t zero_point = 0;
};

class Tensor {
public:
  std::vector<int> shape;
  // Note: both data and grad use NoInitAllocator — see the big comment above
  // for why this is safe and what the contract is.
  std::vector<float, NoInitAllocator<float>> data;
  std::vector<float, NoInitAllocator<float>> grad;

  inline Tensor() {}

  // 2D constructor (batch × features) — used by Dense, biases, labels
  inline Tensor(int rows, int cols) : shape({rows, cols}) {
    data.resize(rows * cols); // No 0.0f passed! Triggers instant allocation
  }

  // 4D constructor (N, C, H, W or N, H, W, C depending on context)
  inline Tensor(int n, int c, int h, int w) : shape({n, c, h, w}) {
    data.resize(n * c * h * w);
  }

  // Arbitrary-shape constructor — used for kernel weights etc.
  inline explicit Tensor(std::vector<int> s) : shape(std::move(s)) {
    int total = 1;
    for (int d : shape) total *= d;
    data.resize(total);
  }

  // Allocate gradient storage. This is the ONE place where we deliberately
  // zero-initialize via NoInitAllocator — passing 0.0f as the fill value
  // triggers the sizeof...(Args) > 0 path, so every gradient element starts
  // at zero. Gradients must be zero because they accumulate via += in backward.
  inline void require_grad() {
    if (grad.size() != data.size()) {
      grad.resize(data.size(), 0.0f); // Gradients MUST still be zeroed
    }
  }

  inline void randomize() { fill_random(-0.01f, 0.01f); }

  inline void fill_random(float min = -1.0f, float max = 1.0f) {
    // thread_local: each OpenMP thread gets its own RNG — no lock contention
    static thread_local std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dis(min, max);
    for (auto &v : data) v = dis(gen);
  }

  // He initialization — the standard for ReLU-family networks.
  // Scale = sqrt(2 / fan_in) keeps variance stable through deep networks.
  // Using a normal distribution (not uniform) because it more closely matches
  // the theoretical derivation from the Kaiming He et al. paper.
  inline void fill_he_init(int fan_in) {
    if (fan_in <= 0) return;
    static thread_local std::mt19937 gen(std::random_device{}());
    float limit = std::sqrt(2.0f / fan_in);
    std::normal_distribution<float> dis(0.0f, limit);
    for (auto &v : data) v = dis(gen);
  }

  inline void fill_constant(float value) { std::fill(data.begin(), data.end(), value); }
  inline void zero_grad() { if (!grad.empty()) std::fill(grad.begin(), grad.end(), 0.0f); }
  inline void fill(float v) { fill_constant(v); }

  inline int size() const { return (int)data.size(); }
  inline int dims() const { return (int)shape.size(); }

  // ---- Index accessors ----
  // Note: these are fine for cold-path code (setup, debugging, loss computation
  // over small class counts). But in hot inner loops (Conv2D, Dense GEMM), we
  // bypass these entirely and use raw pointer arithmetic with precomputed
  // strides. The reason: operator() calls get_index() which does a multiply
  // per dimension, and the compiler can't always prove that shape[] values are
  // loop-invariant, so it may reload them every iteration. Raw pointer walking
  // with precomputed offsets avoids this entirely.
  inline int get_index(int i, int j) const { return i * shape[1] + j; }
  inline int get_index(int n, int c, int h, int w) const {
    return n * (shape[1] * shape[2] * shape[3]) + c * (shape[2] * shape[3]) + h * shape[3] + w;
  }
  inline int get_index(const std::vector<int> &idx) const {
    int i = 0, mul = 1;
    for (int d = (int)shape.size() - 1; d >= 0; --d) {
      i += idx[d] * mul;
      mul *= shape[d];
    }
    return i;
  }

  inline float &operator()(int i, int j) { return data[get_index(i, j)]; }
  inline float operator()(int i, int j) const { return data[get_index(i, j)]; }
  inline float &operator()(int n, int c, int h, int w) { return data[get_index(n, c, h, w)]; }
  inline float operator()(int n, int c, int h, int w) const { return data[get_index(n, c, h, w)]; }

  // Raw pointer access — this is what the hot paths actually use.
  // data_ptr() gives you a float* you can stride through manually.
  inline float *data_ptr() { return data.data(); }
  inline const float *data_ptr() const { return data.data(); }
  inline float *grad_ptr() { return grad.data(); }
  inline const float *grad_ptr() const { return grad.data(); }

  // std::span views — zero-cost wrappers that carry the size for safety
  // in non-performance-critical paths (e.g., Adam optimizer iteration).
  inline std::span<float> view() { return {data.data(), data.size()}; }
  inline std::span<const float> view() const { return {data.data(), data.size()}; }

  // ---- INT8 Quantization ----
  // Affine quantization: maps [min, max] float range to [-128, 127] int8.
  // scale = (max - min) / 255, zero_point = round(-128 - min/scale).
  // This is the same scheme TensorRT/ONNX Runtime use for post-training
  // quantization. The #pragma omp simd hint lets the compiler auto-vectorize
  // the clamp+round loop.
  inline QuantizedTensor quantize() const {
    QuantizedTensor q;
    q.shape = shape;
    q.data.resize(data.size());
    if (data.empty()) return q;

    auto [min_it, max_it] = std::minmax_element(data.begin(), data.end());
    float mn = *min_it;
    float mx = *max_it;

    float sc = (mx - mn) / 255.0f;
    if (sc == 0.0f) sc = 1e-9f;

    int32_t zp = (int32_t)std::round(-128.0f - mn / sc);

    #pragma omp simd
    for (int i = 0; i < (int)data.size(); ++i) {
      int32_t qv = (int32_t)std::round(data[i] / sc) + zp;
      qv = std::max(-128, std::min(127, qv));
      q.data[i] = (int8_t)qv;
    }

    q.scale = sc;
    q.zero_point = zp;
    return q;
  }
};
} // namespace CnnInCpp
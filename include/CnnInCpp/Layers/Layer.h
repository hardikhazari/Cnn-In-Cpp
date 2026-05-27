// ============================================================================
// Layer.h — Abstract base class for every neural network layer in CnnInCpp.
//
// This is the interface that all layers (Conv2D, Dense, ReLU, etc.) implement.
// It defines the forward/backward contract, the buffer management scheme, and
// the graph connectivity for the DAG-based autograd system.
//
// Design trade-off: Yes, forward() and backward() are virtual. This means a
// vtable lookup on every layer call during the forward/backward pass. But
// here's why it doesn't matter: the virtual dispatch happens once per layer
// per batch, and a single Conv2D::forward() call does millions of FMAs. The
// vtable lookup is ~2ns vs. the ~2ms of actual compute. The abstraction cost
// is literally 0.0001% of runtime.
//
// What DOES matter for performance is the buffer management:
// - output_buffer: pre-allocated during compile(), reused every forward pass.
//   This is the #1 reason CnnInCpp is fast — no malloc in the hot path.
// - grad_input_buffer: pre-allocated gradient storage for single-input layers.
// - multi_grad_input_buffers: for multi-input layers (Concat, ElementwiseAdd)
//   that need to route gradients back to each parent separately.
// - cached_input_ptr: raw pointer to the input tensor from the last forward().
//   Needed by backward() to compute weight gradients (e.g., dW = input^T × dOutput).
//   This is a non-owning pointer — the input must outlive the backward call.
//
// The input_nodes/output_nodes/extra_input_nodes fields form the DAG:
// - input_nodes: parents in the graph (whose outputs feed into this layer)
// - output_nodes: children (who consume this layer's output)
// - extra_input_nodes: residual/skip connections that add to this layer's input
//   (used for ResNet-style architectures via Model::add_residual_link)
// ============================================================================
#pragma once
#include "../core/Tensor.h"
#include <vector>
#include <memory>
#include <stdexcept>
#include <string>

namespace CnnInCpp {

class Layer {
public:
    virtual ~Layer() = default;

    // compile() — Called once during Model::compile() to pre-allocate all buffers.
    // This is the "pay once, reuse forever" pattern: every tensor this layer will
    // ever need is allocated here, so forward()/backward() never touch the heap.
    // input_shapes: vector of shapes from parent layers (most layers have exactly 1).
    virtual void compile(const std::vector<std::vector<int>>& input_shapes) = 0;

    // ---- Pre-allocated buffers (set during compile, reused every pass) ----
    Tensor output_buffer;           // Output of forward() — written every call
    Tensor grad_input_buffer;       // Gradient w.r.t. input — written by backward()
    Tensor grad_output_buffer;      // Incoming gradient from child layers (accumulated)
    std::vector<Tensor> multi_grad_input_buffers; // Per-parent gradients for multi-input layers
    Tensor temp_add_buffer;         // Workspace for residual addition in Model::forward()
    
    // ---- DAG connectivity ----
    std::vector<Layer*> input_nodes;        // Parent layers (data flows in)
    std::vector<Layer*> output_nodes;       // Child layers (data flows out)
    std::vector<Layer*> extra_input_nodes;  // Residual/skip connection parents
    std::vector<const Tensor*> extra_inputs; // Pointers to residual inputs (set per forward)
    
    // ---- Cached state for backward pass ----
    const Tensor* cached_input_ptr = nullptr;  // Non-owning pointer to last forward() input
    std::vector<const Tensor*> cached_input_ptrs; // For multi-input forward
    bool   has_grad_cache = false;
    bool   is_training    = true;

    inline void add_extra_input_source(Layer* parent) {
        extra_input_nodes.push_back(parent);
    }

    // ---- Forward/backward interface ----
    // Default implementations throw — concrete layers override the ones they need.
    // Single-input layers override forward(const Tensor&).
    // Multi-input layers (Concat, ElementwiseAdd) override forward(vector<const Tensor*>).
    virtual Tensor& forward(const Tensor& input) {
        throw std::runtime_error("forward(Tensor) not implemented");
    }
    virtual void backward(const Tensor& grad) {
        throw std::runtime_error("backward(Tensor) not implemented");
    }
    virtual Tensor& forward(const std::vector<const Tensor*>& inputs) {
        if (inputs.size() == 1) {
            return forward(*inputs[0]);
        }
        throw std::runtime_error("Multi-input forward not implemented");
    }
    inline Tensor& forward(std::initializer_list<const Tensor*> inputs) {
        return forward(std::vector<const Tensor*>(inputs));
    }
    // backward_multi: for layers like Concat that need to produce separate
    // gradient tensors for each input parent. Default falls through to backward().
    virtual void backward_multi(const Tensor& grad) {
        backward(grad);
    }
    
    // ---- Training infrastructure (cold path — called once per step) ----
    virtual void update_weights(float lr) {}
    virtual void train() { is_training = true; }
    virtual void eval()  { is_training = false; }
    virtual std::vector<Tensor*> get_parameters() { return {}; }
    virtual std::vector<Tensor*> get_states()     { return get_parameters(); }
    virtual std::string name() const = 0;
};

} // namespace CnnInCpp

// ============================================================================
// Model.h — The Directed Acyclic Graph (DAG) Autograd Engine.
//
// CnnInCpp models are not simple lists of layers; they are general DAGs.
// Layers are nodes, and the outputs of one layer flow into the inputs of
// another. This structure supports complex architectures like ResNets
// (residual links) and Inception (parallel branches).
//
// Graph Compilation (`compile()`):
//   1. Runs Kahn's algorithm for topological sorting to establish a valid
//      execution order (parents always execute before children).
//   2. Pre-allocates ALL buffers for every layer in the graph by propagating
//      tensor shapes from input to output.
//   3. This is the cold path. It is slow, but it guarantees the hot path
//      (forward/backward) does zero memory allocation.
//
// Forward Pass:
//   Walks the topological order. If a layer has `extra_inputs` (residual links),
//   it uses its `temp_add_buffer` to sum them before passing to the layer.
//
// Backward Pass:
//   Walks the graph in REVERSE topological order. Gradients are accumulated
//   into `grad_output_buffer` from all children before a layer's `backward()`
//   is called. This is the essence of reverse-mode automatic differentiation.
//
// Memory Management (train vs eval):
//   During `eval()`, we drop all gradient buffers (`grad_input_buffer`, `grad_output_buffer`)
//   to cut memory usage in half. `train()` re-allocates them.
// ============================================================================
#pragma once
#include <vector>
#include <memory>
#include <unordered_map>
#include <queue>
#include <fstream>
#include <string>
#include "../Layers/Layer.h"
#include "Optimizer.h"
#include "Loss.h"
#include "../Layers/BatchNorm2D.h"     
#include "../Layers/FusedConvBNReLU.h" 

namespace CnnInCpp {

class Model {
public:
    std::vector<std::shared_ptr<Layer>> nodes;
    std::vector<Layer*> layers;

    inline void add_node(std::shared_ptr<Layer> n) {
        nodes.push_back(n);
    }

    inline void connect(std::shared_ptr<Layer> from, std::shared_ptr<Layer> to) {
        from->output_nodes.push_back(to.get());
        to->input_nodes.push_back(from.get());
    }

    // Fluent sequential builder: model << layerA << layerB
    inline Model& operator<<(std::shared_ptr<Layer> layer) {
        add_node(layer);
        if (nodes.size() > 1)
            connect(nodes[nodes.size() - 2], layer);
        return *this;
    }

    inline void build_graph() {
        layers.clear();
        std::unordered_map<Layer*, int> in_degree;
        std::queue<Layer*> q;

        for (auto& n : nodes) {
            in_degree[n.get()] = (int)n->input_nodes.size();
            if (in_degree[n.get()] == 0) q.push(n.get());
        }
        while (!q.empty()) {
            auto u = q.front(); q.pop();
            layers.push_back(u);
            for (auto& v : u->output_nodes)
                if (--in_degree[v] == 0) q.push(v);
        }
    }

    inline void add_residual_link(std::shared_ptr<Layer> parent, std::shared_ptr<Layer> child) {
        child->add_extra_input_source(parent.get());
    }

    inline void compile(const std::vector<int>& primary_input_shape) {
        if (layers.empty()) build_graph();
        for (auto& layer : layers) {
            std::vector<std::vector<int>> input_shapes;
            if (layer->input_nodes.empty()) {
                input_shapes.push_back(primary_input_shape);
            } else {
                for (auto& p : layer->input_nodes)
                    input_shapes.push_back(p->output_buffer.shape);
            }
            for (auto& p : layer->extra_input_nodes) {
                input_shapes.push_back(p->output_buffer.shape);
            }
            
            layer->compile(input_shapes);

            if (!layer->extra_input_nodes.empty()) {
                layer->temp_add_buffer = Tensor(input_shapes[0]);
            }
            layer->grad_output_buffer = Tensor(layer->output_buffer.shape);
        }
    }

    inline Tensor& forward(const Tensor& input) {
        if (layers.empty()) build_graph(); 
        
        for (auto& layer : layers) {
            std::vector<const Tensor*> ins;
            if (layer->input_nodes.empty()) {
                ins.push_back(&input);
            } else {
                for (auto& n : layer->input_nodes)
                    ins.push_back(&n->output_buffer);
            }
            
            layer->extra_inputs.clear();
            for (auto& en : layer->extra_input_nodes) {
                layer->extra_inputs.push_back(&en->output_buffer);
            }

            if (!layer->extra_inputs.empty() && !ins.empty()) {
                const Tensor* primary = ins[0];
                Tensor& comb = layer->temp_add_buffer;
                float* d = comb.data.data();
                const float* p = primary->data.data();
                int sz = comb.size();
                #pragma omp simd
                for (int i=0; i<sz; ++i) d[i] = p[i];

                for (const auto* ei : layer->extra_inputs) {
                    const float* s = ei->data.data();
                    #pragma omp simd
                    for (int i=0; i<sz; ++i) d[i] += s[i];
                }
                ins[0] = &comb;
            }
            
            layer->forward(ins);
        }
        return layers.back()->output_buffer;
    }

    inline void backward(const Tensor& grad_output) {
        // Clear all parent gradients first for pure additive accumulation
        for (auto& layer : layers) {
            layer->grad_output_buffer.fill(0.0f);
            for (Tensor* p : layer->get_parameters()) {
                p->zero_grad();
            }
        }
        
        float* dst = layers.back()->grad_output_buffer.data.data();
        const float* src = grad_output.data.data();
        int sz = grad_output.size();
        #pragma omp simd
        for (int k=0; k<sz; ++k) dst[k] = src[k];

        for (int i = (int)layers.size() - 1; i >= 0; --i) {
            auto& layer  = layers[i];
            
            layer->backward_multi(layer->grad_output_buffer);

            for (size_t j = 0; j < layer->input_nodes.size(); ++j) {
                auto& parent = layer->input_nodes[j];
                const Tensor& grad_to_pass = (layer->input_nodes.size() == 1) ? layer->grad_input_buffer : layer->multi_grad_input_buffers[j];

                float* pdst = parent->grad_output_buffer.data.data();
                const float* psrc = grad_to_pass.data.data();
                int psz = parent->grad_output_buffer.size();
                #pragma omp simd
                for (int k = 0; k < psz; ++k) pdst[k] += psrc[k];
            }

            for (size_t j = 0; j < layer->extra_input_nodes.size(); ++j) {
                auto& extra_parent = layer->extra_input_nodes[j];
                const Tensor& grad_to_pass = layer->grad_input_buffer;

                float* pdst = extra_parent->grad_output_buffer.data.data();
                const float* psrc = grad_to_pass.data.data();
                int psz = extra_parent->grad_output_buffer.size();
                #pragma omp simd
                for (int k = 0; k < psz; ++k) pdst[k] += psrc[k];
            }
        }
    }

    inline void eval() { 
        for (auto& n : nodes) n->eval(); 
        for (auto& l : layers) {
            l->eval(); 
            // NATIVE INFERENCE MEMORY OPTIMIZATION (PyTorch torch.no_grad() equivalent)
            l->grad_input_buffer.data.clear();  l->grad_input_buffer.data.shrink_to_fit();
            l->grad_output_buffer.data.clear(); l->grad_output_buffer.data.shrink_to_fit();
            for (auto& buf : l->multi_grad_input_buffers) {
                buf.data.clear(); buf.data.shrink_to_fit();
            }
            // Clear layer-specific training buffers safely
            if (auto bn = dynamic_cast<BatchNorm2D*>(l)) {
                bn->x_norm.data.clear(); bn->x_norm.data.shrink_to_fit();
            }
            if (auto fused = dynamic_cast<FusedConvBNReLU*>(l)) {
                fused->x_norm.data.clear(); fused->x_norm.data.shrink_to_fit();
            }
        }
    }

    inline void train() { 
        for (auto& n : nodes) n->train(); 
        for (auto& l : layers) {
            l->train();
            // AUTOMATIC RE-ALLOCATION FOR TRAINING
            if (l->grad_input_buffer.data.empty() && !l->grad_input_buffer.shape.empty()) {
                int sz = 1; for(int d: l->grad_input_buffer.shape) sz *= d;
                l->grad_input_buffer.data.resize(sz, 0.0f);
            }
            if (l->grad_output_buffer.data.empty() && !l->grad_output_buffer.shape.empty()) {
                int sz = 1; for(int d: l->grad_output_buffer.shape) sz *= d;
                l->grad_output_buffer.data.resize(sz, 0.0f);
            }
            for (auto& buf : l->multi_grad_input_buffers) {
                 if (buf.data.empty() && !buf.shape.empty()) {
                      int sz = 1; for(int d: buf.shape) sz *= d;
                      buf.data.resize(sz, 0.0f);
                 }
            }
            if (auto bn = dynamic_cast<BatchNorm2D*>(l)) {
                if(bn->x_norm.data.empty() && !bn->x_norm.shape.empty()) {
                    int sz = 1; for(int d: bn->x_norm.shape) sz *= d;
                    bn->x_norm.data.resize(sz, 0.0f);
                }
            }
            if (auto fused = dynamic_cast<FusedConvBNReLU*>(l)) {
                if(fused->x_norm.data.empty() && !fused->x_norm.shape.empty()) {
                    int sz = 1; for(int d: fused->x_norm.shape) sz *= d;
                    fused->x_norm.data.resize(sz, 0.0f);
                }
            }
            // Enable gradients on learnable parameters
            for (Tensor* p : l->get_parameters()) p->require_grad();
        }
    }

    inline void save(const std::string& filename) {
        std::string json = "{ \"layers\": [";
        for (size_t i = 0; i < layers.size(); ++i) {
            json += "{\"name\":\"" + layers[i]->name() + "\", \"params\": [";
            auto states = layers[i]->get_states();
            for (size_t j = 0; j < states.size(); ++j) {
                json += "{\"shape\": [";
                for (size_t k = 0; k < states[j]->shape.size(); ++k) {
                    json += std::to_string(states[j]->shape[k]) + (k == states[j]->shape.size() - 1 ? "" : ",");
                }
                json += "], \"size\":" + std::to_string(states[j]->size()) + "}" + (j == states.size() - 1 ? "" : ",");
            }
            json += std::string("]}") + (i == layers.size() - 1 ? "" : ",");
        }
        json += "] }";

        std::ofstream file(filename, std::ios::binary);
        uint64_t head_sz = json.size();
        file.write(reinterpret_cast<const char*>(&head_sz), sizeof(head_sz));
        file.write(json.data(), head_sz);

        for (auto& layer : layers)
            for (Tensor* s : layer->get_states())
                file.write(reinterpret_cast<const char*>(s->data.data()),
                           s->data.size() * sizeof(float));
    }

    inline void load(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) return;

        uint64_t head_sz;
        file.read(reinterpret_cast<char*>(&head_sz), sizeof(head_sz));
        std::string json(head_sz, ' ');
        file.read(&json[0], head_sz);
        // Simple JSON header for metadata - we rely on the binary order for now, 
        // but the header is there for future validation/inspection.

        for (auto& layer : layers)
            for (Tensor* s : layer->get_states())
                file.read(reinterpret_cast<char*>(s->data.data()),
                          s->data.size() * sizeof(float));
    }
};

} // namespace CnnInCpp

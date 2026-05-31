// ============================================================================
// Adam.h — Adaptive Moment Estimation optimizer.
//
// Adam is the industry standard for deep learning. Unlike SGD which uses a
// single global learning rate, Adam keeps a running average of the first
// moment (mean) and second moment (uncentered variance) of the gradients for
// EVERY individual parameter.
//
// This means every parameter gets its own adaptive learning rate. Parameters
// with sparse/infrequent gradients get boosted, while parameters with large/
// noisy gradients get dampened.
//
// Memory cost: 2× the parameter size (we have to store m and v for every weight).
//
// We use `#pragma omp simd` in the update loop because it's a perfectly mapped
// element-wise operation.
// ============================================================================
#pragma once
#include "Optimizer.h"
#include <unordered_map>
#include <cmath>

namespace CnnInCpp {

class Adam : public Optimizer {
public:
    float lr, b1, b2, eps;
    int   t = 0;
    std::unordered_map<Tensor*, Tensor> m, v;

    float weight_decay;

    inline Adam(float l=0.001f, float b1_=0.9f, float b2_=0.999f, float e=1e-8f, float wd=0.0f)
        : lr(l), b1(b1_), b2(b2_), eps(e), weight_decay(wd) {}

    inline void step(std::vector<std::shared_ptr<Layer>>& layers) override {
        ++t;
        const float bc1 = 1.0f - std::pow(b1, t);
        const float bc2 = 1.0f - std::pow(b2, t);
        
        float total_norm_sq = 0.0f;
        for (auto& layer : layers) {
            for (Tensor* p : layer->get_parameters()) {
                auto pd = p->data_ptr();
                auto gd = p->grad_ptr();
                int sz = p->size();
                
                if (weight_decay > 0.0f) {
                    #pragma omp simd
                    for (int i=0; i<sz; ++i) {
                        gd[i] += weight_decay * pd[i];
                    }
                }
                
                float local_sum = 0.0f;
                #pragma omp simd reduction(+:local_sum)
                for (int i=0; i<sz; ++i) {
                    local_sum += gd[i] * gd[i];
                }
                total_norm_sq += local_sum;
            }
        }
        float total_norm = std::sqrt(total_norm_sq);
        float threshold = 1.0f;
        float scale = (total_norm > threshold) ? (threshold / total_norm) : 1.0f;

        for (auto& layer : layers) {
            for (Tensor* p : layer->get_parameters()) {
                if (m.find(p) == m.end()) {
                    m[p]=Tensor(p->shape); m[p].fill(0.0f);
                    v[p]=Tensor(p->shape); v[p].fill(0.0f);
                }
                auto pd = p->view();
                auto gd = p->grad_ptr();
                auto md = m[p].view();
                auto vd = v[p].view();
                int sz = (int)pd.size();

                #pragma omp simd
                for (int i=0; i<sz; ++i) {
                    float scaled_grad = gd[i] * scale;
                    md[i] = b1*md[i] + (1-b1)*scaled_grad;
                    vd[i] = b2*vd[i] + (1-b2)*scaled_grad*scaled_grad;
                    float mh = md[i]/bc1, vh = vd[i]/bc2;
                    pd[i] -= lr * mh / (std::sqrt(vh) + eps);
                }
            }
        }
    }
};

} // namespace CnnInCpp

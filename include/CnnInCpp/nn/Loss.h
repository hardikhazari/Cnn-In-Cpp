// ============================================================================
// Loss.h — Loss Functions (CrossEntropy, MSE).
//
// These functions evaluate how far off the model's predictions are from the
// true labels, returning a single scalar float (the loss).
// They also compute the initial gradient (`grad_output`) that gets fed into
// the `backward()` method of the final layer in the network to start the
// backpropagation chain.
// ============================================================================
#pragma once
#include "../core/Tensor.h"
#include <cmath>
#include <limits>
#include <algorithm>

namespace CnnInCpp {

class Loss {
public:
    virtual ~Loss() = default;
    virtual float  forward (const Tensor& p, const Tensor& t) = 0;
    virtual Tensor backward(const Tensor& p, const Tensor& t) = 0;
};

class MSELoss : public Loss {
public:
    inline float forward(const Tensor& p, const Tensor& t) override {
        float loss=0.0f; const int N=p.shape[0];
        const float* pd=p.data.data(); const float* td=t.data.data();
        #pragma omp parallel for simd reduction(+:loss)
        for (int i=0;i<p.size();++i) { float d=pd[i]-td[i]; loss+=d*d; }
        return loss/N;
    }
    inline Tensor backward(const Tensor& p, const Tensor& t) override {
        const int N=p.shape[0]; Tensor d(p.shape);
        const float* pd=p.data.data(); const float* td=t.data.data(); float* di=d.data.data();
        #pragma omp simd
        for (int i=0;i<p.size();++i) di[i]=2.0f*(pd[i]-td[i])/N;
        return d;
    }
};

class CrossEntropyLoss : public Loss {
public:
    inline float forward(const Tensor& p, const Tensor& t) override {
        const int N=p.shape[0], C=p.shape[1]; float loss=0.0f;
        #pragma omp parallel for reduction(+:loss)
        for (int i=0; i<N; ++i) {
            float mx = -std::numeric_limits<float>::infinity();
            for (int j=0; j<C; ++j) {
                float clamped_p = std::clamp(p(i,j), -100.0f, 100.0f);
                mx = std::max(mx, clamped_p);
            }
            
            float sum_exp = 0.0f;
            for (int j=0; j<C; ++j) {
                float clamped_p = std::clamp(p(i,j), -100.0f, 100.0f);
                sum_exp += std::exp(clamped_p - mx);
            }
            float log_sum_exp = mx + std::log(sum_exp);
            
            for (int j=0; j<C; ++j) {
                float clamped_p = std::clamp(p(i,j), -100.0f, 100.0f);
                if (t(i,j) > 0.0f) {
                    loss -= t(i,j) * (clamped_p - log_sum_exp);
                }
            }
        }
        return loss/N;
    }
    inline Tensor backward(const Tensor& p, const Tensor& t) override {
        const int N=p.shape[0], C=p.shape[1]; Tensor d(p.shape);
        #pragma omp parallel for
        for (int i=0;i<N;++i) {
            float mx=-std::numeric_limits<float>::infinity();
            for (int j=0;j<C;++j) {
                float clamped_p = std::clamp(p(i,j), -100.0f, 100.0f);
                mx=std::max(mx,clamped_p);
            }
            float se=0; 
            for (int j=0;j<C;++j) {
                float clamped_p = std::clamp(p(i,j), -100.0f, 100.0f);
                se+=std::exp(clamped_p-mx);
            }
            for (int j=0;j<C;++j) {
                float clamped_p = std::clamp(p(i,j), -100.0f, 100.0f);
                d(i,j)=(std::exp(clamped_p-mx)/se-t(i,j))/N;
            }
        }
        return d;
    }
};

} // namespace CnnInCpp

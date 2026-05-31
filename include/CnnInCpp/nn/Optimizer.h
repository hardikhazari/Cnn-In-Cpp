// ============================================================================
// Optimizer.h — Abstract base class for optimizers.
//
// Simple interface: call step() with a parameter tensor, and the optimizer
// updates the tensor's data in place based on its grad.
// ============================================================================
#pragma once
#include <vector>
#include <memory>
#include "../Layers/Layer.h"

namespace CnnInCpp {

class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void step(std::vector<std::shared_ptr<Layer>>& layers) = 0;
};

class SGD : public Optimizer {
public:
    float lr;
    inline SGD(float l=0.01f) : lr(l) {}
    inline void step(std::vector<std::shared_ptr<Layer>>& layers) override {
        for (auto& l : layers) l->update_weights(lr);
    }
};

} // namespace CnnInCpp

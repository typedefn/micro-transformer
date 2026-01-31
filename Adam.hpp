#pragma once
#include "Common.hpp"

class Adam {
  private:
    float beta1, beta2, epsilon;
    std::vector<float> m, v;
  public:
    Adam(int param_count, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f)
      : beta1(beta1), beta2(beta2), epsilon(epsilon) { 
        m.assign(param_count, 0.0f);
        v.assign(param_count, 0.0f);
      }

    void update(std::vector<float>& params, const std::vector<float>& grads, float lr, int t, float wd = 0.0f, float grad_scale = 1.0f) {
      for (size_t i = 0; i < params.size(); ++i) {
        float g = grads[i] * grad_scale; // Scale gradient by (clipper_scale / batch_size)

        params[i] *= (1.0f - lr * wd); // Weight decay
        m[i] = beta1 * m[i] + (1.0f - beta1) * g;
        v[i] = beta2 * v[i] + (1.0f - beta2) * (g * g);

        float m_hat = m[i] / (1.0f - std::pow(beta1, t));
        float v_hat = v[i] / (1.0f - std::pow(beta2, t));
        params[i] -= lr * (m_hat / (std::sqrt(v_hat) + epsilon));
      }
    }

    void save() {
      if (!output_file.is_open()) throw std::runtime_error("Adam Save: File not open");
      
      // Save 1st moment vector
      for (float val : m) output_file << val << " ";
      
      // Save 2nd moment vector
      for (float val : v) output_file << val << " ";
    }

    void load() {
      if (!inputFile.is_open()) throw std::runtime_error("Adam Load: File not open");

      // Load 1st moment vector
      for (auto &val : m) inputFile >> val;

      // Load 2nd moment vector
      for (auto &val : v) inputFile >> val;
    }
};

#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <map>
#include <random>
#include <exception>
#include <fstream>
#include <sstream>
#include <string>
#include <limits> 
#include <cassert>
#include <signal.h>
#include <chrono>
#include <mutex>
#include <atomic>
#include <utility>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

static unsigned int seed = 1;

std::ofstream output_file;
std::ifstream inputFile;
bool verbose = false;
bool sig = false;
std::string filename = "model_last.data";

#include "Logger.hpp"
#include "Adam.hpp"
#define BLOCK_SIZE 256
#include "Mat.hpp"

// Update signature to accept an optional scale factor (default 1.0)
void he_init(std::vector<float>& weights, int fan_in, float scale_factor = 1.0f) {
    // Standard deviation for ReLU networks
    float std_dev = std::sqrt(2.0f / (float)fan_in);
    
    // Apply the additional scaling (important for deep residual networks)
    std_dev *= scale_factor;

    static std::mt19937 gen(seed);
    std::normal_distribution<float> distribution(0.0f, std_dev);

    for (auto& weight : weights) {
        weight = distribution(gen);
    }
}

static std::mt19937& get_rng() {
    static std::random_device rd;
    static std::mt19937 gen(seed);//rd()); // Seed with hardware entropy once
    return gen;
}

void normal_init(std::vector<float>& weights, float std_dev = 0.02f) {
    std::normal_distribution<float> distribution(0.0f, std_dev);
    auto& gen = get_rng();
    for (auto& weight : weights) {
      weight = distribution(gen);
    }
}

static std::vector<float> pe(int pos, int dim) {
    if (dim <= 0) {
        throw std::runtime_error("Position encoding dim must be greater than 0");
    }

    std::vector<float> pe(dim);
    
    // Standard geometric sequence for frequencies: 10000^(-2i/dim)
    // We compute the scalar divisor once per pair.
    float div_term_scalar = -std::log(10000.0f) / (float)dim;

    for (int i = 0; i < dim; i += 2) {
        // Calculate frequency: exp(i * -log(10000) / dim)
        // This is mathematically equivalent to 1 / (10000^(2i/dim))
        float div_term = std::exp(i * div_term_scalar);

        // Even indices: Sine
        pe[i] = std::sin(pos * div_term);

        // Odd indices: Cosine
        // Check bounds in case dim is odd
        if (i + 1 < dim) {
            pe[i + 1] = std::cos(pos * div_term);
        }
    }

    return pe;
}

class GlobalClipper {
public:
    double total_sum_sq = 0.0;

    // Accumulate sum of squares from a vector
    void accumulate(const std::vector<float>& grad) {
        double layer_sum = 0.0;
        #pragma omp parallel for reduction(+:layer_sum)
        for (size_t i = 0; i < grad.size(); ++i) {
            layer_sum += grad[i] * grad[i];
        }
        total_sum_sq += layer_sum;
    }

    // Calculate the multiplier to scale all gradients
    float get_scale_factor(float max_norm) {
        float global_norm = std::sqrt(total_sum_sq);
        // If global norm is 100 and max is 1.0, we scale by 1/100
        return (global_norm > max_norm) ? (max_norm / global_norm) : 1.0f;
    }

    void reset() {
        total_sum_sq = 0.0f;
    }

    float get_global_scale(float max_norm) {
      float total_norm = std::sqrt(total_sum_sq);
      float clip_coef = 1.0f;
      if (total_norm > max_norm) {
        clip_coef = max_norm / (total_norm + 1e-6f);
      }
      return clip_coef;
    }
};


struct DataBatch {
  Mat<float> inputs; // (batchSize * seqLength, embeddingLength)
  Mat<float> targets; // (batchSize * seqLength, vocabSize)
  std::vector<int> token_ids;
  std::vector<int> target_ids;
  int valid_seq_len;
};

void clipGradients(std::vector<float>& gradients, float max_norm = 1.0f) {
    float total_norm = 0.0f;
    for (const auto& grad : gradients) {
        total_norm += grad * grad;
    }
    total_norm = std::sqrt(total_norm);

    if (total_norm > max_norm) {
        float scale = max_norm / (total_norm + 1e-6f);
        for (auto& grad : gradients) {
            grad *= scale;
        }
    }
}


#pragma once
#include "Common.hpp"

class FeedForwardNetwork {
  Adam optimizer_w1, optimizer_b1, optimizer_w2, optimizer_b2;
  bool dropout_enabled;
  float p;
  Mat<float> hidden_local;
  Mat<float> hidden_local_transpose;
  Mat<float> weights2_local_transpose;
  Mat<float> hidden_local_activated;
  Mat<float> hidden_local_activated_tmp;
  Mat<float> mask_local_scaled;
  Mat<float> hidden_local_transpose_mul_gradient;
  Mat<float> input_local_transpose;
  Mat<float> input_local_to_hidden;
  Mat<float> weights1_local_transpose;
  Mat<float> pre_active_hidden_local;
  Mat<float> output_local;
  Mat<float> mask_local;
  Mat<float> dbias2_local;
  Mat<float> dbias1_local;
  Mat<float> dweights2_local;
  Mat<float> dweights1_local;
  float weight_decay;

  public:
  // Define a cache to hold all intermediate values
  struct FFNCache {
    Mat<float> input;
    Mat<float> pre_active_hidden;
    Mat<float> hidden;
    Mat<float> output;
    Mat<float> mask;
  };

  FeedForwardNetwork(int inputDim, int hiddenDim, int outputDim, const std::string & id, int decoder_layers, int maxTotalTokens, float weight_decay)
  : inputDim(inputDim), hiddenDim(hiddenDim), outputDim(outputDim),
    optimizer_w1(inputDim * hiddenDim), optimizer_b1(hiddenDim),
    optimizer_w2(hiddenDim * outputDim), optimizer_b2(outputDim), id(id),
    weights1(inputDim, hiddenDim),
    dweights1(inputDim, hiddenDim),
    bias1(1, hiddenDim),
    dbias1(1, hiddenDim),
    weights2(hiddenDim, outputDim),
    dweights2(hiddenDim, outputDim),
    bias2(1, outputDim),
    dbias2(1, outputDim),
    p(0),
    dropout_enabled(false),
    weight_decay(weight_decay)
  {

    // 1. Initial Weight Setup
    float gpt_scale = 0.02f / sqrt(2.0f * (float) decoder_layers);

    normal_init(weights1.raw(), 0.02f);
    normal_init(weights2.raw(), gpt_scale);
    std::fill(bias1.raw().begin(), bias1.raw().end(), 0.0f);
    std::fill(bias2.raw().begin(), bias2.raw().end(), 0.0f);
    
     
    // 2. Static Cache Allocation (N x Dim)
    cache.input.assign(maxTotalTokens, inputDim);
    cache.pre_active_hidden.assign(maxTotalTokens, hiddenDim);
    cache.hidden.assign(maxTotalTokens, hiddenDim);
    cache.mask.assign(maxTotalTokens, hiddenDim);
    cache.output.assign(maxTotalTokens, outputDim);

    // 3. Computation Locals (N-dependent)
    pre_active_hidden_local.assign(maxTotalTokens, hiddenDim); 
    mask_local_scaled.assign(maxTotalTokens, hiddenDim);
    hidden_local.assign(maxTotalTokens, hiddenDim);
    hidden_local_activated.assign(maxTotalTokens, hiddenDim);
    hidden_local_activated_tmp.assign(maxTotalTokens, hiddenDim);
    output_local.assign(maxTotalTokens, outputDim);
    mask_local.assign(maxTotalTokens, hiddenDim);

    // 4. Transpose Locals (Dim x N)
    hidden_local_transpose.assign(hiddenDim, maxTotalTokens);
    input_local_transpose.assign(inputDim, maxTotalTokens);

    // 5. Weight-related Locals (Static Dims)
    weights2_local_transpose.assign(outputDim, hiddenDim);
    weights1_local_transpose.assign(hiddenDim, inputDim);
    input_local_to_hidden.assign(inputDim, hiddenDim);
    dbias2_local.assign(1, outputDim);
    dbias1_local.assign(1, hiddenDim);
    dweights1_local.assign(inputDim, hiddenDim);
    dweights2_local.assign(hiddenDim, outputDim);
    hidden_local_transpose_mul_gradient.assign(hiddenDim, outputDim);
    weights1.to_gpu();
    weights2.to_gpu();
    bias1.to_gpu();
    bias2.to_gpu();
    reset();
  }
  // Forward pass
  void forward(Mat<float>& input, Mat<float> & output) {
    cache.input.copy_from(input);
    input.mul(weights1, cache.pre_active_hidden);
    cache.pre_active_hidden.add(bias1, cache.pre_active_hidden);
    cache.pre_active_hidden.gelu(hidden_local);
    // Create Mask
    if (dropout_enabled) {
      Mat<float>::create_dropout_mask(hidden_local.get_rows(), hidden_local.get_cols(), p, cache.mask);
      // Hidden to output
      float scale_factor = 1.0f / (1.0f - p); 
      hidden_local.element_wise_product(cache.mask, mask_local);
      mask_local.scale(scale_factor, mask_local_scaled);
      mask_local_scaled.mul(weights2, cache.output);
      cache.output.add(bias2, cache.output);
    } else {
      hidden_local.mul(weights2, cache.output);
      cache.output.add(bias2, cache.output);           
    }
    output.copy_from(cache.output);
  }

  // Backward pass (backpropagation)
  // upstream gradient dims 1 x outputDim
  // output matrix is inputDim x inputDim
  void backward(Mat<float> & upstream_gradient, Mat<float> & output) {
    int N = upstream_gradient.get_rows();
    // dW2 = Hidden^T * Upstream_Gradient
    cache.hidden.transpose(hidden_local_transpose);
    hidden_local_transpose.mul(upstream_gradient, dweights2_local);
    // Correct accumulation: Add the current batch gradient to the persistent dweights2
    dweights2.add(dweights2_local, dweights2); 

    // db2 = sum_rows(Upstream_Gradient)
    upstream_gradient.sum_rows(dbias2_local); 
    // FIX: Add the current batch bias gradient to the persistent dbias2 buffer
    dbias2.add(dbias2_local, dbias2); 

    // Propagate gradient to Hidden layer: dHidden = Upstream_Gradient * W2^T
    weights2.transpose(weights2_local_transpose);
    upstream_gradient.mul(weights2_local_transpose, hidden_local_activated);

    // --- Step 2: Backprop through Dropout and GELU ---

    if (dropout_enabled) {
      float scale_factor = 1.0f / (1.0f - p); 
      hidden_local_activated.element_wise_product(cache.mask, hidden_local_activated_tmp);
      hidden_local_activated_tmp.scale(scale_factor, hidden_local_activated);
    }

    // dPreActive = dHidden * GELU_derivative(PreActiveHidden)
    cache.pre_active_hidden.gelu_backward(hidden_local_activated, pre_active_hidden_local);

    // --- Step 3: Gradients for the First Layer (Linear 1) ---

    // dW1 = Input^T * dPreActive
    cache.input.transpose(input_local_transpose);
    input_local_transpose.mul(pre_active_hidden_local, dweights1_local);
    // Correct accumulation: Add to the persistent dweights1 buffer
    dweights1.add(dweights1_local, dweights1);

    // db1 = sum_rows(dPreActive)
    pre_active_hidden_local.sum_rows(dbias1_local); 
    // FIX: Add to the persistent dbias1 buffer
    dbias1.add(dbias1_local, dbias1);

    // Final Output: dInput = dPreActive * W1^T (to be passed to preceding Attention layer)
    weights1.transpose(weights1_local_transpose);
    pre_active_hidden_local.mul(weights1_local_transpose, output);
  }

  void accumulate_gradients(GlobalClipper& clipper) {
    clipper.accumulate(dweights1.raw());
    clipper.accumulate(dweights2.raw());
    clipper.accumulate(dbias1.raw());
    clipper.accumulate(dbias2.raw());
  }

  void dW_scale(float scale_val) {
    if (scale_val == 1.0f) return;

    // Scale gradients for the first linear layer (Input -> Hidden)
    dweights1.scale(scale_val, dweights1);
    dbias1.scale(scale_val, dbias1);

    // Scale gradients for the second linear layer (Hidden -> Output)
    dweights2.scale(scale_val, dweights2);
    dbias2.scale(scale_val, dbias2);
  }
  void clipGrads(float scale) {
    clipGradients(dweights1.raw(), scale);
    clipGradients(dbias1.raw(), scale);
    clipGradients(dweights2.raw(), scale);
    clipGradients(dbias2.raw(), scale);
    dweights1.to_gpu(); dbias1.to_gpu();
    dweights2.to_gpu(); dbias2.to_gpu();
  }

  void update_weights(float learningRate, float scale, int current_t) {
/*
    if (scale != 1.0f) {
      dweights1.scale(scale, dweights1);
      dbias1.scale(scale, dbias1);
      dweights2.scale(scale, dweights2);
      dbias2.scale(scale, dbias2);
    }
*/
    std::vector<float>& temp_dweights1 = dweights1.raw();
    std::vector<float>& temp_dweights2 = dweights2.raw();
    std::vector<float>& temp_dbias1 = dbias1.raw();
    std::vector<float>& temp_dbias2 = dbias2.raw();

    std::vector<float>& temp_weights1 = weights1.raw();
    std::vector<float>& temp_weights2 = weights2.raw();
    std::vector<float>& temp_bias1 = bias1.raw();
    std::vector<float>& temp_bias2 = bias2.raw();

    optimizer_w1.update(temp_weights1, temp_dweights1, learningRate, current_t, weight_decay, scale);
    optimizer_b1.update(temp_bias1, temp_dbias1, learningRate, current_t, 0.0f, scale);
    optimizer_w2.update(temp_weights2, temp_dweights2, learningRate, current_t, weight_decay, scale);
    optimizer_b2.update(temp_bias2, temp_dbias2, learningRate, current_t, 0.0f, scale);

    weights1.to_gpu();
    weights2.to_gpu();
    bias1.to_gpu();
    bias2.to_gpu();
  }

  void enable_dropout(bool enable, float p) {
    dropout_enabled = enable;
    this->p = p;
  }

  void reset() {
    dweights1.reset();
    dweights2.reset();
    dbias2.reset();
    dbias1.reset();
  }

  void save() {
    if (!output_file.is_open()) {
      throw std::runtime_error("Cannot save ANN weights");
    }
    for(float w: weights1.raw()) {
      output_file << w << " ";
    }
    for(float w: bias1.raw()) {
      output_file << w << " ";
    }
    for(float w: weights2.raw()) {
      output_file << w << " ";
    }
    for(float w: bias2.raw()) {
      output_file << w << " ";
    }
    optimizer_w1.save();
    optimizer_b1.save();
    optimizer_w2.save();
    optimizer_b2.save();
  }

  void load() {
    if (!inputFile.is_open()) {
      std::stringstream ss;
      ss << "Unable to open file";
      throw std::runtime_error(ss.str());
    }

    for (auto & w: weights1.raw()) {
      inputFile >> w;
    }  

    for (auto & w: bias1.raw()) {
      inputFile >> w;
    }  

    for (auto & w: weights2.raw()) {
      inputFile >> w;
    }  

    for (auto & w: bias2.raw()) {
      inputFile >> w;
    }  

    weights1.to_gpu();
    weights2.to_gpu();
    bias1.to_gpu();
    bias2.to_gpu();
    optimizer_w1.load();
    optimizer_b1.load();
    optimizer_w2.load();
    optimizer_b2.load();
  }
  private:
  int inputDim;
  int hiddenDim;
  int outputDim;
  Mat<float> weights1;
  Mat<float> bias1;
  Mat<float> weights2;
  Mat<float> bias2;
  Mat<float> dweights2;
  Mat<float> dweights1;
  Mat<float> dbias2;
  Mat<float> dbias1;
  std::string id;
  public:
  FFNCache cache;
};

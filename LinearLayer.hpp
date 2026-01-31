#include "Common.hpp"

class LinearLayer {
  float weight_decay;
  public:
    struct LinearCache {
      Mat<float> input;
    };

    LinearLayer(int inputDim, int hiddenDim, int maxTotalTokens, float weight_decay, int decoder_layers)
      : inputDim(inputDim), hiddenDim(hiddenDim),
      optimizer_w1(inputDim * hiddenDim), optimizer_b1(hiddenDim), weight_decay(weight_decay) {

        // Persistent Parameters (Static)
        bias1.assign(1, hiddenDim);
        dbias1.assign(1, hiddenDim);
        weights1.assign(inputDim, hiddenDim);
        dweights1.assign(inputDim, hiddenDim);
        float res_scale = 0.01f / (float)std::sqrt((float)inputDim);
        // Initialize weights and biases
        he_init(weights1.raw(), inputDim, res_scale);
        he_init(bias1.raw(), hiddenDim, 0.0f);

        weights1.to_gpu();
        bias1.to_gpu();

        // STATIC ALLOCATION for maxTotalTokens (N)
        // Caches and Outputs (Shape: N x Dim)
        cache.input.assign(maxTotalTokens, inputDim);
        output_local.assign(maxTotalTokens, hiddenDim);
        input_local_d_gradient.assign(inputDim, hiddenDim);
        dweights1_local.assign(inputDim, hiddenDim);
        dbias1_local.assign(1, hiddenDim);

        // Transposed Locals (Shape: Dim x N)
        input_local_transpose.assign(inputDim, maxTotalTokens);
        weights1_local_transpose.assign(hiddenDim, inputDim);
        cache.input.reset();
        output_local.reset();
        input_local_d_gradient.reset();
        dweights1_local.reset();
        dbias1_local.reset();
        reset();
      }

    // Forward pass
    void forward(Mat<float>& input, Mat<float> & out) {
      int N = input.get_rows();
      cache.input.copy_from(input);
      cache.input.mul(weights1, output_local);
      output_local.add(bias1, out); 
    }

    // Backward pass (backpropagation)
    // upstream_gradient dimensions are 1 x hiddenDim
    // out matrix dimensions are 1 x inputDim
    void backward(Mat<float>& upstream_gradient, Mat<float> & out) {
      int N = upstream_gradient.get_rows();

      // --- Step 1: Weight Gradients (dW = X^T * dO) ---
      cache.input.transpose(input_local_transpose);
      // Calculate the gradient for this specific batch
      input_local_transpose.mul(upstream_gradient, input_local_d_gradient);
      // Accumulate directly into the persistent dweights1 buffer
      dweights1.add(input_local_d_gradient, dweights1);     
      // --- Step 2: Bias Gradients (db = sum(dO)) ---
      upstream_gradient.sum_rows(dbias1_local);
      // FIX: Directly accumulate the batch sum into the master dbias1 buffer
      dbias1.add(dbias1_local, dbias1);

      // --- Step 3: Input Gradients (dX = dO * W^T) ---
      weights1.transpose(weights1_local_transpose);
      upstream_gradient.mul(weights1_local_transpose, out);  
    }


    void accumulate_gradients(GlobalClipper& clipper) {
      clipper.accumulate(dweights1.raw());
      clipper.accumulate(dbias1.raw());
      dweights1.to_gpu(); dbias1.to_gpu();
    }

    void dW_scale(float scale_val) {
      if (scale_val == 1.0f) return;
      dweights1.scale(scale_val, dweights1);
      dbias1.scale(scale_val, dbias1);
    }

    void clipGrads(float scale) {
      clipGradients(dweights1.raw(), scale);
      clipGradients(dbias1.raw(), scale);
      dweights1.to_gpu(); dbias1.to_gpu();
    }

    void update_weights(float learningRate, float scale, int current_t) {
      if (scale != 1.0f) {
        dweights1.scale(scale, dweights1);
        dbias1.scale(scale, dbias1);
      }
      std::vector<float>& temp_dweights1 = dweights1.raw();
      std::vector<float>& temp_dbias1 = dbias1.raw();
      std::vector<float>& temp_weights1 = weights1.raw();
      std::vector<float>& temp_bias1 = bias1.raw();
      optimizer_w1.update(temp_weights1, temp_dweights1, learningRate, current_t, weight_decay, 1.0f);
      optimizer_b1.update(temp_bias1, temp_dbias1, learningRate, current_t, 0, 1.0f);

      bias1.to_gpu();
      weights1.to_gpu();
    }
    void reset() {
      dweights1.reset();
      dbias1.reset();
    }

    void save() {
      if (!output_file.is_open()) {
        throw std::runtime_error("Cannot save linear weights");
      }
      for(float w: weights1.raw()) {
        output_file << w << " ";
      }
      for(float w: bias1.raw()) {
        output_file << w << " ";
      }
      optimizer_w1.save();
      optimizer_b1.save();
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
      weights1.to_gpu();
      bias1.to_gpu();
      optimizer_w1.load();
      optimizer_b1.load();
    }

  private:
    int inputDim;
    int hiddenDim;
    Mat<float> weights1;
    Mat<float> bias1;
    Mat<float> dweights1;
    Mat<float> dbias1;
    Mat<float> pre_activation;  
    Mat<float> input_local_transpose;
    Mat<float> input_local_d_gradient;
    Mat<float> dweights1_local;
    Mat<float> dbias1_local;
    Mat<float> weights1_local_transpose;
    Mat<float> output_local;
    Adam optimizer_w1, optimizer_b1;
    LinearCache cache;
};

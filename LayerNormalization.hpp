#pragma once
#include "Common.hpp"

class LayerNormalization {
    int featureSize;
    int seqLength; // Used for initialization, but we allow dynamic resizing
    Adam optimizer_gamma, optimizer_beta;
    float epsilon;
    
    // Parameters (Trainable) - Shape (1, featureSize)
    Mat<float> gamma;
    Mat<float> beta;
    Mat<float> dgamma;
    Mat<float> dbeta;
    Mat<float> dgamma_local; // Helpers for accumulation if needed
    Mat<float> dbeta_local;

    // Cache for Backward Pass
    Mat<float> cache_input; // Stores X
    Mat<float> cache_mean;  // Stores Mean per row
    Mat<float> cache_var;   // Stores Var per row
    Mat<float> d_input;     // Stores dx
    Mat<float> output;      // Stores Result

    std::string id;
    float weight_decay;
public:
    LayerNormalization(int featureSize, int maxSeqBatchSize, const std::string & id, float weight_decay):
        featureSize(featureSize), seqLength(maxSeqBatchSize),
        optimizer_gamma(featureSize), optimizer_beta(featureSize),
        epsilon(1e-5f), id(id),
        gamma(1, featureSize), beta(1, featureSize),
        dgamma(1, featureSize), dbeta(1, featureSize),
        dgamma_local(1, featureSize), dbeta_local(1, featureSize),
        weight_decay(weight_decay)
    {
        // Initialize Gamma to 1.0, Beta to 0.0
        // We can use a temporary CPU vector to set this up
        std::vector<float> h_gamma(featureSize, 1.0f);
        std::vector<float> h_beta(featureSize, 0.0f);
        
        gamma.set_raw(h_gamma);
        beta.set_raw(h_beta);
        // Pre-allocate caches for max size (optional, or resize dynamically)
        cache_input.assign(maxSeqBatchSize, featureSize);
        output.assign(maxSeqBatchSize, featureSize);
        cache_mean.assign(maxSeqBatchSize, 1);
        cache_var.assign(maxSeqBatchSize, 1);
        d_input.assign(maxSeqBatchSize, featureSize);
        reset();
    }

    // --- FORWARD PASS (GPU) ---
    // Takes GPU Mat, returns GPU Mat (view of internal output)
    Mat<float>& forward(const Mat<float> & inputs) {
        int rows = inputs.get_rows();
        int cols = inputs.get_cols(); // Should match featureSize
        // 2. Copy Input to Cache (Deep copy needed because input might be overwritten)
        // Actually, we can just copy to cache_input and operate on it.
        hipError_t err = hipMemcpy(cache_input.d_data, inputs.d_data, rows * cols * sizeof(float), hipMemcpyDeviceToDevice);

        if (err != hipSuccess) {
          std::stringstream ss;
          ss << "hipMemCpy failed: " << hipGetErrorString(err);
          throw std::runtime_error(ss.str());
        }

        // 3. Compute Mean and Variance
        // Launch 1 block per row
        kLayerNormStats<<<rows, BLOCK_SIZE>>>(cache_input.d_data, cache_mean.d_data, cache_var.d_data, rows, cols);

        // 4. Normalize and Scale
        int total = rows * cols;
        int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
        kLayerNormForward<<<blocks, BLOCK_SIZE>>>(
            cache_input.d_data, output.d_data, 
            cache_mean.d_data, cache_var.d_data, 
            gamma.d_data, beta.d_data, 
            rows, cols, epsilon
        );
        output.dirty();
        return output;
    }

    // Overload to keep compatibility with std::vector signature for now, 
    // BUT this effectively just wraps the Mat version. 
    std::vector<float> forward(const std::vector<float> & h_inputs) {
        // Wrap input in temporary Mat
        int rows = h_inputs.size() / featureSize;
        Mat<float> input_mat(rows, featureSize);
        hipError_t err = hipMemcpy(input_mat.d_data, h_inputs.data(), h_inputs.size() * sizeof(float), hipMemcpyHostToDevice);

        if (err != hipSuccess) {
          std::stringstream ss;
          ss << "hipMemcpy failed: " << hipGetErrorString(err);
          throw std::runtime_error(ss.str());
        }
        
        Mat<float>& out_mat = forward(input_mat);
        
        // Return CPU vector
        return out_mat.raw();
    }

    // --- BACKWARD PASS (GPU) ---
    Mat<float>& backward(Mat<float> & upstream_gradient) {
        int rows = upstream_gradient.get_rows();
        int cols = upstream_gradient.get_cols();
        
        // 1. Calculate Gradients for Gamma and Beta
        // We use one thread per feature column to sum down rows. 
        // Note: For huge batch sizes, this naive atomic/loop might be slow. 
        // Ideally use reduction. But for Batch=8/64, simple loop is fine.
        int threads = 256;
        int blocks = (cols + threads - 1) / threads;
        
        // Reset gradients first? No, we overwrite in the kernel usually, 
        // but if we accumulate over steps we need logic. 
        // Assuming 'backward' is called once per step per layer instance.
        
        kLayerNormBackwardParams<<<blocks, threads>>>(
            upstream_gradient.d_data, cache_input.d_data, 
            cache_mean.d_data, cache_var.d_data,
            dgamma_local.d_data, dbeta_local.d_data,
            rows, cols, epsilon
        );

        // Accumulate local gradients into master gradients
        dgamma.add(dgamma_local, dgamma); // Accumulate
        dbeta.add(dbeta_local, dbeta);

        // 2. Calculate Gradient w.r.t Input
        // One block per row
        kLayerNormBackwardInput<<<rows, BLOCK_SIZE>>>(
            upstream_gradient.d_data, cache_input.d_data, 
            cache_mean.d_data, cache_var.d_data,
            gamma.d_data, d_input.d_data,
            rows, cols, epsilon
        );

        d_input.dirty();
        return d_input;
    }

    // Compatibility Wrapper
    std::vector<float> backward(const std::vector<float> & h_grad) {
        int rows = h_grad.size() / featureSize;
        Mat<float> grad_mat(rows, featureSize);
        hipError_t err = hipMemcpy(grad_mat.d_data, h_grad.data(), h_grad.size() * sizeof(float), hipMemcpyHostToDevice);
        if (err != hipSuccess) {
          std::stringstream ss;
          ss << "hipMemcpy failed: " << hipGetErrorString(err);
          throw std::runtime_error(ss.str());
        }

        Mat<float>& out_mat = backward(grad_mat);
        return out_mat.raw();
    }

    void accumulate_gradients(GlobalClipper& clipper) {
      clipper.accumulate(dgamma.raw());
      clipper.accumulate(dbeta.raw());
    }

    void dW_scale(float scale_val) {
      if (scale_val == 1.0f) return;

      // Scale the master gradient matrices for trainable parameters
      dgamma.scale(scale_val, dgamma);
      dbeta.scale(scale_val, dbeta);
    }
    void clipGrads(float scale) {
      clipGradients(dgamma.raw(), scale);
      clipGradients(dbeta.raw(), scale);
      dgamma.to_gpu();
      dbeta.to_gpu();
    }
    void update_weights(float learningRate, float scale, int current_t) {
        std::vector<float>&  v_gamma = gamma.raw();
        std::vector<float>&  v_dgamma = dgamma.raw();
        std::vector<float>&  v_beta = beta.raw();
        std::vector<float>&  v_dbeta = dbeta.raw();
        optimizer_gamma.update(v_gamma, v_dgamma, learningRate, current_t, 0.0f, scale);
        optimizer_beta.update(v_beta, v_dbeta, learningRate, current_t, 0.0f, scale);

        gamma.to_gpu();
        beta.to_gpu();
    }

    void reset() {
        dgamma.reset();
        dbeta.reset();
        dgamma_local.reset(); dbeta_local.reset();
    }

    void save() { 
        if (!output_file.is_open()) throw std::runtime_error("Save: File not open");
        for(float w: gamma.raw()) output_file << w << " ";
        for(float w: beta.raw()) output_file << w << " ";
        optimizer_gamma.save();
        optimizer_beta.save();
    }
    
    void load() {
        std::vector<float> v_gamma(featureSize), v_beta(featureSize);
        if (!inputFile.is_open()) throw std::runtime_error("Load: File not open");
        for (auto & w: v_gamma) inputFile >> w;
        for (auto & w: v_beta) inputFile >> w;
        gamma.set_raw(v_gamma);
        beta.set_raw(v_beta);
        optimizer_gamma.load();
        optimizer_beta.load();
    }
};

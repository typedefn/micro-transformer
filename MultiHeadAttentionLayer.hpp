#pragma once
#include "Common.hpp"
#include "AttentionLayer.hpp"

class MultiHeadAttentionLayer {
  private:
    int num_heads;
    int embeddingLength;
    int seqSize;
    int batchSize;
    int head_dim; // embeddingLength / num_heads
    float weight_decay;

    // Each head is a complete, independent AttentionLayer
    // The final output projection layer
    Mat<float> W_o;
    Mat<float> dW_o;
    Adam optimizer_o;
    std::vector<Mat<float>> head_outputs;
    std::vector<Mat<float>> head_d_inputs;
    std::vector<Mat<float>> d_head_outputs;

    // Caches for backpropagation
    std::string id;
    Mat<float> concatenated_output_mat;
    Mat<float> final_output;
    Mat<float> concatenated_transposed_mat;
    Mat<float> W_o_transposed_mat;
    Mat<float> d_concatenated_mat;
    Mat<float> dW_o_temp;
    Mat<float> dW_o_accumulator_temp; 

  public:
    std::vector<AttentionLayer> heads;
    MultiHeadAttentionLayer(int seqSize, int batchSize, int embeddingLength, int num_heads, const std::string& id, int decoder_layers, float weight_decay)
        : seqSize(seqSize), batchSize(batchSize), embeddingLength(embeddingLength), num_heads(num_heads),
          optimizer_o(embeddingLength * embeddingLength), id(id), weight_decay(weight_decay) {
        
        assert(embeddingLength % num_heads == 0);
        this->head_dim = embeddingLength / num_heads;

       heads.reserve(num_heads);
       head_outputs.reserve(num_heads);
       head_d_inputs.reserve(num_heads);
       d_head_outputs.reserve(num_heads); 

        // Create an independent AttentionLayer for each head.
        // NOTE: The AttentionLayer needs to work on a smaller `head_dim`
        // instead of the full `embeddingLength`.
        int total_tokens = batchSize * seqSize;
        for (int i = 0; i < num_heads; ++i) {
            heads.emplace_back(seqSize, batchSize, embeddingLength, head_dim, id + "_head_" + std::to_string(i), decoder_layers, weight_decay);
            head_outputs.emplace_back(total_tokens, head_dim);
            head_d_inputs.emplace_back(total_tokens, embeddingLength);
            d_head_outputs.emplace_back(total_tokens, head_dim);
        }

        concatenated_output_mat.assign(total_tokens, embeddingLength);
        final_output.assign(total_tokens, embeddingLength); 
        // Initialize W_o, optimizer_o, etc.
        W_o.assign(embeddingLength, embeddingLength);
        float gpt_scale = 0.02f / sqrt(2.0f * (float) decoder_layers);
        normal_init(W_o.raw(), gpt_scale);
        W_o.to_gpu();
        dW_o.assign(embeddingLength, embeddingLength);
        dW_o_temp.assign(embeddingLength, embeddingLength);
        dW_o_accumulator_temp.assign(embeddingLength, embeddingLength);
        size_t rowSize = batchSize * seqSize;
        concatenated_transposed_mat.assign(embeddingLength, total_tokens);
        W_o_transposed_mat.assign(embeddingLength, embeddingLength);
        d_concatenated_mat.assign(total_tokens, embeddingLength);
        reset();
    }

    Mat<float> & forward(Mat<float>& inputs, int valid_seq_len) {
      int total_tokens = batchSize * seqSize;

      #pragma omp parallel for
      for (int h = 0; h < num_heads; ++h) {
        heads[h].forward(inputs, head_outputs[h], valid_seq_len);
      }

      hipError_t err = hipDeviceSynchronize(); 

      if (err != hipSuccess) {
        std::stringstream ss;
        ss << "hipDeviceSynchronize failed: " << hipGetErrorString(err);
        throw std::runtime_error(ss.str());
      }
 
      Mat<float>::concatenate_heads(head_outputs, concatenated_output_mat);
      concatenated_output_mat.mul(W_o, final_output);
      return final_output;
    } 

    void backward(Mat<float>& upstream_gradients, Mat<float>& dx_final_mat) {
      // 1. Calculate dW_o = concatenated_transposed_mat * upstream_gradients
      concatenated_output_mat.transpose(concatenated_transposed_mat);
      concatenated_transposed_mat.mul(upstream_gradients, dW_o_temp);

      // 2. Accumulate directly into dW_o
      // Since Mat::add(a, b, result) is implemented, we can use dW_o as the result
      dW_o.add(dW_o_temp, dW_o); 

      // 3. d_concatenated = upstream_gradients * W_o^T
      W_o.transpose(W_o_transposed_mat);
      upstream_gradients.mul(W_o_transposed_mat, d_concatenated_mat);

      // 4. Split and backprop through heads
      Mat<float>::split_gradients(d_concatenated_mat, d_head_outputs);

      #pragma omp parallel for
      for (int h = 0; h < num_heads; ++h) {
        heads[h].backward(d_head_outputs[h], head_d_inputs[h]);
      }

      hipError_t err = hipDeviceSynchronize();
 
      if (err != hipSuccess) {
        std::stringstream ss;
        ss << "hipDeviceSynchronize failed: " << hipGetErrorString(err);
        throw std::runtime_error(ss.str());
      }

      // 5. Final input gradient: Sum the head_d_inputs
      dx_final_mat.reset(); 
      for (int h = 0; h < num_heads; ++h) {
        dx_final_mat.add(head_d_inputs[h], dx_final_mat);
      }
    }

    void accumulate_gradients(GlobalClipper& clipper) {
      clipper.accumulate(dW_o.raw());
      for (auto& head : heads) {
        head.accumulate_gradients(clipper);
      }
    }

    void dW_scale(float scale_val) {
      if (scale_val == 1.0f) return;
      dW_o.scale(scale_val, dW_o);
      for (auto& head : heads) {
        head.dW_scale(scale_val);
      }
    }

    void clipGrads(float scale) {
      clipGradients(dW_o.raw(), scale);
      dW_o.to_gpu();
      for (auto& head : heads) {
        head.clipGrads(scale);
      }
    }

    void update_weights(float learningRate, float scale, int current_t) {
/*
      if (scale != 1.0f) {
        dW_o.scale(scale, dW_o);
      }
*/
      std::vector<float>& tmp_dW_o = dW_o.raw();
      std::vector<float>& tmp_W_o = W_o.raw();
      optimizer_o.update(tmp_W_o, tmp_dW_o, learningRate, current_t, weight_decay, scale);
      dW_o.to_gpu();
      W_o.to_gpu();

      for (auto& head : heads) {
        head.update_weights(learningRate, scale, current_t);
      }
    }

    void print() {
      W_o.print("MultiHeadAttention Weights:");
/*      for (auto & head : heads) {
        head.print();
      } 
*/
    }

    void reset() {
      dW_o.reset();
      for (auto& head : heads) {
        head.reset();
      }
    }

    void load() {
      int size = embeddingLength * embeddingLength;
      W_o.raw().assign(size, 0);
      if (!inputFile.is_open()) {
        std::stringstream ss;
        ss << "Unable to open file";
        throw std::runtime_error(ss.str());
      }
    
      for (auto & w: W_o.raw()) {
        inputFile >> w;
      }  
    
      for (auto& head : heads) {
        head.load();
      }
      W_o.to_gpu();
    
      optimizer_o.load();
    }

    void save() {
      if (!output_file.is_open()) {
        throw std::runtime_error("Cannot save attention weights");
      }
      for(float w: W_o.raw()) {
        output_file << w << " ";
      }

      for (auto& head : heads) {
        head.save();
      }
      optimizer_o.save();
    }
};

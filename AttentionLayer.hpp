#pragma once
#include "Common.hpp"

class AttentionLayer {
  public:

    struct AttentionCache {
      Mat<float> queries, keys, values;
      Mat<float> attention_scores;
      Mat<float> inputs;
    };
    AttentionCache cache;
  private:

    Adam optimizer_q, optimizer_k, optimizer_v;
    int seqSize;
    int batchSize;
    int embeddingLength;
    int head_dim;
    float weight_decay;

    Mat<float> W_q, W_k, W_v;
    Mat<float> dW_q, dW_k, dW_v;
    std::string id;

    // Local Buffers
    Mat<float> queries_local;
    Mat<float> keys_local;
    Mat<float> values_local;

    Mat<float> attention_score_local;
    Mat<float> attention_score_local_scaled;
    Mat<float> attention_score_softmaxed;

    Mat<float> d_values_local;
    Mat<float> d_scores_scaled;

    Mat<float> dW_q_local, dW_k_local, dW_v_local;
    Mat<float> dW_q_local_sum, dW_k_local_sum, dW_v_local_sum;

    Mat<float> inputs_transposed;
    Mat<float> W_q_transposed, W_k_transposed, W_v_transposed;
    Mat<float> tmp_q, tmp_k, tmp_v;
    Mat<float> dx;
  public:

    AttentionLayer(int seqSize, int batchSize, int embeddingLength, int head_dim, const std::string & id, int decoder_layers, float weight_decay)
      : seqSize(seqSize), batchSize(batchSize), embeddingLength(embeddingLength), head_dim(head_dim),
      optimizer_q(embeddingLength * head_dim),
      optimizer_k(embeddingLength * head_dim),
      optimizer_v(embeddingLength * head_dim),
      id(id), 
      W_q(embeddingLength, head_dim),
      W_k(embeddingLength, head_dim),
      W_v(embeddingLength, head_dim),
      dW_q(embeddingLength, head_dim),
      dW_k(embeddingLength, head_dim),
      dW_v(embeddingLength, head_dim),
      weight_decay(weight_decay)
  {

    // The Standard Transformer Initialization (GPT-2 style)
    float base_sigma = 0.01f / (float)std::sqrt((float)embeddingLength);
    float res_sigma = base_sigma / (float)std::sqrt(2.0f * decoder_layers);
    // Q and K are kept at base_sigma to ensure initial scores are balanced
    he_init(W_q.raw(), embeddingLength, base_sigma);
    he_init(W_k.raw(), embeddingLength, base_sigma);
    // V is scaled down because it directly impacts the magnitude of the residual signal
    he_init(W_v.raw(), embeddingLength, res_sigma);
    size_t rowSize = batchSize * seqSize;

    // --- Vectors (Size: Batch*Seq, HeadDim) ---
    cache.inputs.assign(rowSize, embeddingLength);
    cache.queries.assign(rowSize, head_dim);
    cache.keys.assign(rowSize, head_dim);
    cache.values.assign(rowSize, head_dim);
    queries_local.assign(rowSize, head_dim);
    keys_local.assign(rowSize, head_dim);
    values_local.assign(rowSize, head_dim);

    // --- Score Matrices (Size: Batch*Seq, Seq) ---
    // Note: We use 'seqSize' as cols, NOT 'rowSize'. This is the strided optimization.
    cache.attention_scores.assign(rowSize, seqSize);
    attention_score_local.assign(rowSize, seqSize);
    attention_score_local_scaled.assign(rowSize, seqSize);
    attention_score_softmaxed.assign(rowSize, seqSize);
    d_scores_scaled.assign(rowSize, seqSize);

    // --- Gradients ---
    d_values_local.assign(rowSize, head_dim);

    inputs_transposed.assign(embeddingLength, rowSize);

    dW_q_local.assign(embeddingLength, head_dim);
    dW_k_local.assign(embeddingLength, head_dim);
    dW_v_local.assign(embeddingLength, head_dim);

    dW_q_local_sum.assign(embeddingLength, head_dim);
    dW_k_local_sum.assign(embeddingLength, head_dim);
    dW_v_local_sum.assign(embeddingLength, head_dim);

    W_q_transposed.assign(head_dim, embeddingLength);
    W_k_transposed.assign(head_dim, embeddingLength);
    W_v_transposed.assign(head_dim, embeddingLength);

    tmp_q.assign(rowSize, embeddingLength);
    tmp_k.assign(rowSize, embeddingLength);
    tmp_v.assign(rowSize, embeddingLength);
    dx.assign(rowSize, embeddingLength);
    W_q.to_gpu();
    W_k.to_gpu();
    W_v.to_gpu();
  }

    void forward(Mat<float>& inputs, Mat<float> & outputs, int valid_seq_len) {
      // 1. Calculate Batch Size dynamically (Handles Inference vs Training)
      int stride_q = seqSize * head_dim;
      int stride_k = seqSize * head_dim;
      int stride_v = seqSize * head_dim;
      int stride_score = seqSize * seqSize;
      int stride_out = seqSize * head_dim;

      cache.inputs.copy_from(inputs);
      cache.inputs.mul(W_q, queries_local);
      cache.inputs.mul(W_k, keys_local);
      cache.inputs.mul(W_v, values_local);

      // 2. Calculate Attention Scores: Q * K^T
      // Uses Batched Stride: [B, S, H] * [B, H, S] -> [B, S, S]
      float scale = 1.0f/std::sqrt((float)head_dim);
      queries_local.batched_mul_transpose(
          keys_local, 
          attention_score_local, 
          batchSize, 
          stride_q, stride_k, stride_score
          );
      // 3. Mask and Scale
      attention_score_local.scale_mask(scale, valid_seq_len, seqSize, attention_score_local_scaled);

      // 4. Softmax
      attention_score_local_scaled.softmax(attention_score_softmaxed);
      // Cache for backward
      cache.queries.copy_from(queries_local);
      cache.keys.copy_from(keys_local);
      cache.values.copy_from(values_local);
      cache.attention_scores.copy_from(attention_score_softmaxed);

      // 5. Context: Scores * V
      // Uses Batched Stride: [B, S, S] * [B, S, H] -> [B, S, H]
      cache.attention_scores.batched_mul(
          cache.values, 
          outputs, 
          batchSize, 
          stride_score, stride_v, stride_out
          );
    }

    void backward(Mat<float>& upstream_gradients, Mat<float>& outputs) {
      int stride_score = seqSize * seqSize;
      int stride_dO = seqSize * head_dim; 
      int stride_V = seqSize * head_dim;
      int stride_Q = seqSize * head_dim;
      int stride_K = seqSize * head_dim;
      // 1. Calculate dV = S^T * dO
      // Uses batched_mul_lhs_transpose
      cache.attention_scores.batched_mul_lhs_transpose(
          upstream_gradients, 
          d_values_local, 
          batchSize, 
          stride_score, stride_dO, stride_V
          );

      // 2. Calculate dS = dO * V^T
      // Uses batched_mul_transpose
      upstream_gradients.batched_mul_transpose(
          cache.values, 
          attention_score_local, // reused buffer for dS
          batchSize, 
          stride_dO, stride_V, stride_score
          );

      // Softmax Gradient
      attention_score_local.softmax_backward(cache.attention_scores, d_scores_scaled);

      float scale = 1.0f / std::sqrt((float)head_dim);
      d_scores_scaled.scale(scale, d_scores_scaled);
      // 3. Calculate dQ = dS * K
      d_scores_scaled.batched_mul(
          cache.keys, 
          queries_local, // reusing queries_local buffer for dQ
          batchSize, 
          stride_score, stride_K, stride_Q
          );

      // 4. Calculate dK = dS^T * Q
      d_scores_scaled.batched_mul_lhs_transpose(
          cache.queries, 
          keys_local, // reusing keys_local buffer for dK
          batchSize, 
          stride_score, stride_Q, stride_K
          );

      // --- Standard Backprop for Weights (Global Aggregation) ---
      // The previous ops calculated per-token gradients (dQ, dK, dV).
      // Now we project back to embedding space and accumulate weight gradients.
      // This part is NOT batched-stride; it's just large matrix multiplication,
      // which correctly sums gradients across the batch naturally.

      cache.inputs.transpose(inputs_transposed);

      inputs_transposed.mul(queries_local, dW_q_local); // X^T * dQ
      inputs_transposed.mul(keys_local, dW_k_local);    // X^T * dK
      inputs_transposed.mul(d_values_local, dW_v_local);// X^T * dV
                                                        // Accumulate these into persistent buffers for the optimizer
      dW_q.add(dW_q_local, dW_q);
      dW_k.add(dW_k_local, dW_k);
      dW_v.add(dW_v_local, dW_v);

      // --- INPUT GRADIENT (dX) ---
      // Propagate error back to the preceding layer
      W_q.transpose(W_q_transposed);
      W_k.transpose(W_k_transposed);
      W_v.transpose(W_v_transposed);

      queries_local.mul(W_q_transposed, tmp_q);  // dX += dQ * Wq^T
      keys_local.mul(W_k_transposed, tmp_k);     // dX += dK * Wk^T
      d_values_local.mul(W_v_transposed, tmp_v); // dX += dV * Wv^T

      // Sum all contributions for the final input gradient
      tmp_q.add(tmp_k, dx);
      dx.add(tmp_v, outputs);
    }

    void accumulate_gradients(GlobalClipper& clipper) {
      clipper.accumulate(dW_q.raw());
      clipper.accumulate(dW_k.raw());
      clipper.accumulate(dW_v.raw());
    }
    void print() {
      attention_score_local.print("ATTENTION SCORE LOCAL:");
    }
    void dW_scale(float scale_val) {
      if (scale_val == 1.0f) return;
      dW_q.scale(scale_val, dW_q);
      dW_k.scale(scale_val, dW_k);
      dW_v.scale(scale_val, dW_v);
    }

    void clipGrads(float scale) {
      clipGradients(dW_q.raw(), scale);
      clipGradients(dW_k.raw(), scale);
      clipGradients(dW_v.raw(), scale);
      dW_q.to_gpu(); dW_k.to_gpu(); dW_v.to_gpu();
    }

    void update_weights(float learningRate, float scale, int current_t) {
      if (scale != 1.0f) {
        dW_q.scale(scale, dW_q);
        dW_k.scale(scale, dW_k);
        dW_v.scale(scale, dW_v);
      }
      optimizer_q.update(W_q.raw(), dW_q.raw(), learningRate, current_t, weight_decay, 1.0f);
      optimizer_k.update(W_k.raw(), dW_k.raw(), learningRate, current_t, weight_decay, 1.0f);
      optimizer_v.update(W_v.raw(), dW_v.raw(), learningRate, current_t, weight_decay, 1.0f);
      W_q.to_gpu(); W_k.to_gpu(); W_v.to_gpu();
    }

    void reset() {
      dW_q.reset(); dW_k.reset(); dW_v.reset();
    }

    void save() {
      if (!output_file.is_open()) throw std::runtime_error("Error save attn");
      for(float w: W_q.raw()) output_file << w << " ";
      for(float w: W_k.raw()) output_file << w << " ";
      for(float w: W_v.raw()) output_file << w << " ";
      optimizer_q.save(); optimizer_k.save(); optimizer_v.save();
    }

    void load() {
      if (!inputFile.is_open()) throw std::runtime_error("Error load attn");
      for(auto & w: W_q.raw()) inputFile >> w;
      for(auto & w: W_k.raw()) inputFile >> w;
      for(auto & w: W_v.raw()) inputFile >> w;
      W_q.to_gpu(); W_k.to_gpu(); W_v.to_gpu();
      optimizer_q.load(); optimizer_k.load(); optimizer_v.load();
    }
};

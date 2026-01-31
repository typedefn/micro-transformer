#pragma once
#include "Common.hpp"
#include "MultiHeadAttentionLayer.hpp"
#include "AttentionLayer.hpp"
#include "LayerNormalization.hpp"
#include "FeedForwardNetwork.hpp"
#include "LinearLayer.hpp"

class DecoderBlock {
    Mat<float> ffn_outputs;

    Mat<float> d_residual_1; 
    Mat<float> d_residual_1_from_ffn;
    Mat<float> d_input_from_attention;
    Mat<float> final_d_input_mat;
    Mat<float> residual_1;
    Mat<float> final_output;
    Mat<float> ff_out;
    Mat<float> d_attn_out;

  public:
    int embeddingLength;
    int seqLength;
    int num_heads;
    int total_tokens;

    MultiHeadAttentionLayer attentionLayer;
    LayerNormalization layerNorm1;
    FeedForwardNetwork ffn; 
    LayerNormalization layerNorm2;

    // We need to store the intermediate outputs for the backward pass
   
    DecoderBlock(int seqLength, int batchSize, int embeddingLength, int num_heads, const std::string & id, int decoder_layers, float weight_decay)
      : embeddingLength(embeddingLength),
      seqLength(seqLength),
      total_tokens(seqLength * batchSize),
      attentionLayer(seqLength, batchSize, embeddingLength, num_heads, id + "_mha_1", decoder_layers, weight_decay),
      layerNorm1(embeddingLength, seqLength * batchSize, id + "_1", weight_decay),
      ffn(embeddingLength, embeddingLength * 4, embeddingLength, id + "_1", decoder_layers, seqLength * batchSize, weight_decay),
      layerNorm2(embeddingLength, seqLength * batchSize, id + "_2", weight_decay) {
        ffn_outputs.assign(total_tokens, embeddingLength);
        d_residual_1.assign(total_tokens, embeddingLength);
        d_residual_1_from_ffn.assign(total_tokens, embeddingLength);
        d_input_from_attention.assign(total_tokens, embeddingLength);
        final_d_input_mat.assign(total_tokens, embeddingLength);
        residual_1.assign(total_tokens, embeddingLength);
        final_output.assign(total_tokens, embeddingLength);
        ff_out.assign(total_tokens, embeddingLength);
        d_attn_out.assign(total_tokens, embeddingLength);
      }

    Mat<float>& forward(const Mat<float>& input, int valid_seq_len) {
      // 1. Normalize the input FIRST
      Mat<float> & norm_input_1 = layerNorm1.forward(input);
      // 2. Pass the NORMALIZED input to the attention layer
      Mat<float> & attention_output = attentionLayer.forward(norm_input_1, valid_seq_len);
      // 3. Add the attention output to the ORIGINAL input (residual connection)
      input.add(attention_output, residual_1);
      // --- Second Sub-layer: Feed-Forward Network ---
      // 4. Normalize the output of the first sub-layer
      Mat<float> & norm_input_2 = layerNorm2.forward(residual_1);
      ffn.forward(norm_input_2, ff_out);
      // 6. Add the FFN output to the input of this sub-layer (residual_1)
      residual_1.add(ff_out, final_output);
      return final_output;
    }

    Mat<float> & backward(Mat<float>& upstream_gradient) {
      // --- Backprop through Second Sub-layer (FFN) ---
      // The upstream gradient flows to two places: the FFN output and the skip connection
      ffn.backward(upstream_gradient, ffn_outputs);
      // Backprop through the second LayerNorm
      Mat<float>& d_ln2_out = layerNorm2.backward(ffn_outputs);

      // Combine the gradients for residual_1
      upstream_gradient.add(d_ln2_out, d_residual_1);
      // --- Backprop through First Sub-layer (Attention) ---

      // Backprop through the Attention layer
      attentionLayer.backward(d_residual_1, d_attn_out);

      // Backprop through the first LayerNorm
      Mat<float>& d_ln1_out = layerNorm1.backward(d_attn_out);

      d_residual_1.add(d_ln1_out, final_d_input_mat);
      final_d_input_mat.dirty();

      return final_d_input_mat;
    }

    void reset() {
      // Force release of cache memory to prevent capacity creep
      layerNorm1.reset();
      attentionLayer.reset();
      layerNorm2.reset();
      ffn.reset();
    }

    void save() {
      layerNorm1.save();
      attentionLayer.save();
      layerNorm2.save();
      ffn.save();
    } 
    void print_attention() {
      attentionLayer.print();
    }
    void accumulate_gradients(GlobalClipper& clipper) {
      attentionLayer.accumulate_gradients(clipper);
      layerNorm1.accumulate_gradients(clipper);
      ffn.accumulate_gradients(clipper);
      layerNorm2.accumulate_gradients(clipper);
    }

    void dW_scale(float scale_val) {
      attentionLayer.dW_scale(scale_val);
      layerNorm1.dW_scale(scale_val); // If LN has trainable params
      ffn.dW_scale(scale_val);
      layerNorm2.dW_scale(scale_val);
    }

    void clipGrads(float scale) {
      layerNorm1.clipGrads(scale);
      attentionLayer.clipGrads(scale);
      layerNorm2.clipGrads(scale);
      ffn.clipGrads(scale);
    }
    void update_weights(float learningRate, float scale, int current_t) {
      layerNorm1.update_weights(learningRate, scale, current_t);
      attentionLayer.update_weights(learningRate, scale, current_t);
      layerNorm2.update_weights(learningRate, scale, current_t);
      ffn.update_weights(learningRate, scale, current_t);
    } 

    void enable_dropout(bool enable, float p) {
      ffn.enable_dropout(enable, p);
    }

    void load() {
      layerNorm1.load();
      attentionLayer.load();
      layerNorm2.load();
      ffn.load();
    }
};

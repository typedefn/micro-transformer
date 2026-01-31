#include "Common.hpp"
#include "DecoderBlock.hpp"

void signalCallbackHandler(int signal) {
  LOG << "Caught signal " << signal;
  sig = true;
}

static std::vector<std::string> createTokenIds(std::map<char, int> & char_to_id, std::vector<float> & vindex, const std::string & filename) {
  std::ifstream inputFile(filename);
  std::vector<std::string> lines;

  if (!inputFile.is_open()) {
    std::stringstream ss;
    ss << "Unable to open file";
    throw std::runtime_error(ss.str());
  }
  //Start ID from the next available number, not always 1
  int id = (char_to_id.empty()) ? 1 : (vindex.size()); 

  std::string line;
  while (std::getline(inputFile, line)) {
    lines.push_back(line);
    for (char ch : line) {
      if (char_to_id.find(ch) == char_to_id.end()) {
        char_to_id[ch] = id++;
        vindex.push_back(id);
      } 
    }
    // Manually add newline to vocab.
    char newline = '\n';
    if (char_to_id.find(newline) == char_to_id.end()) {
      char_to_id[newline] = id++;
      vindex.push_back(id);
    }
  } 
  inputFile.close();

  return lines;
}

class Transformer {
  Mat<float> linear_logits;
  Mat<float> probs_mat;
  std::map<char, int> char_to_id;
  int warmup_epochs;
  int batchSize;
  int embeddingLength;
  std::vector<DataBatch> data;
  float label_smoothing;
  float learningRate;
  int seqLength; 
  std::mt19937 gen;
  std::vector<float> tokenEmbeddings;
  bool train;
  bool load_weights;
  int epochs;
  float initialLearningRate;
  float decayRate;
  float temperature;
  int steps_per_epoch;
  int warmup_steps;
  std::vector<float> vindex;
  std::vector<std::string> training_lines;
  std::vector<std::string> validation_lines;
  int vocabSize;
  LinearLayer linearLayer;
  bool inference;
  std::vector<DecoderBlock> decoder_blocks;
  LayerNormalization finalLayerNorm;
  int decoder_layers;
  float weight_decay;
  void save_embeddings(const std::vector<float>& tokenEmbeddings) {
      if (!output_file.is_open()) {
          throw std::runtime_error("Cannot save token embeddings");
      }
      for (float w : tokenEmbeddings) {
          output_file << w << " ";
      }
      LOG << "Token embeddings saved.";
  }

  void load_embeddings(std::vector<float>& tokenEmbeddings) {
      if (!inputFile.is_open()) {
          throw std::runtime_error("Unable to open token_embeddings.data");
      }
      for (auto& w : tokenEmbeddings) {
          inputFile >> w;
      }
      LOG << "Token embeddings loaded.";
  }
  int num_heads;
  float dropout;
  int global_step;
  float embedding_scale;
  public:
  
  Transformer(
      int embeddingLength, int seqLength, int batchSize, int num_heads,
      int decoder_layers, bool train, float weight_decay, int warmup_epochs,
      float label_smoothing
    ):
    gen(seed),
    linearLayer(embeddingLength, vindex.size(), seqLength * batchSize, weight_decay, decoder_layers),
    finalLayerNorm(embeddingLength, seqLength * batchSize, "final", weight_decay),
    embeddingLength(embeddingLength),
    seqLength(seqLength),
    batchSize(batchSize),
    num_heads(num_heads),
    decoder_layers(decoder_layers),
    train(train),
    weight_decay(weight_decay),
    warmup_epochs(warmup_epochs),
    label_smoothing(label_smoothing)
  {
    global_step = 0;
    load_weights = false;
    inference = false;
    epochs = 1000;
    initialLearningRate = 0.0001f;
    learningRate = initialLearningRate;
    decayRate = 0.9f;
    dropout = 0.2f;
    temperature = 0.7f; // A value > 1.0 encourages creativity. Values between 0.8 and 2.0.
    // Place holder for unknown token, index 0 of the vocab index contains -1, unknown token.
    inputFile.open(filename); 
    vindex.push_back(-1);
    if (!train) {
      load_vocab_mapping(char_to_id, vindex);
    } else {
      training_lines = createTokenIds(char_to_id, vindex, "corpus.txt"); 
      validation_lines = createTokenIds(char_to_id, vindex, "valid.txt"); 
    }

    vocabSize = vindex.size();
    this->linearLayer = LinearLayer(embeddingLength, vocabSize, seqLength * batchSize, weight_decay, decoder_layers);
    embedding_scale = std::sqrt((float)embeddingLength);
  }
  ~Transformer() {
    inputFile.close();
  }
  std::vector<float> generate_inputs(std::string line) {
    std::vector<float> finputs(seqLength, 0);
    if (line.size() >= seqLength) {
      line = line.substr(line.size() - seqLength);
    }
    for (int i = 0; i < line.size(); ++i) {
      char c = (line.at(i));
      int c_to_id = -1;
      if (char_to_id.find(c) != char_to_id.end()) {
        c_to_id = char_to_id.at(c);
      }
      finputs.at(i) = (float)(c_to_id);
    } 
    return finputs;
  }
  
  // Transformer.cpp modification
  DataBatch createBatch(const std::vector<int> & ids, const std::vector<int> & start_indices) {
    DataBatch current_batch;
    int total_tokens = batchSize * seqLength;
    current_batch.inputs.assign(total_tokens, embeddingLength);
    current_batch.targets.assign(total_tokens, vocabSize);

    // Track the actual usable length for masking
    current_batch.valid_seq_len = seqLength; 

    std::vector<float> h_inputs(total_tokens * embeddingLength, 0.0f);
    std::vector<float> h_targets(total_tokens * vocabSize, 0.0f);

    for (int b = 0; b < batchSize; ++b) {
      int start_idx = start_indices[b];
      // Refined inner loop logic
      for (int j = 0; j < seqLength; ++j) {
        int batch_token_offset = b * seqLength + j;

        if (start_idx + j < ids.size() - 1) {
          int tokenIndex = ids[start_idx + j];
          int targetId = ids[start_idx + j + 1];

          current_batch.token_ids.push_back(tokenIndex);
          current_batch.target_ids.push_back(targetId);

          std::vector<float> pez = pe(j, embeddingLength);
          for (int k = 0; k < embeddingLength; ++k) {
            h_inputs[batch_token_offset * embeddingLength + k] = 
              pez[k] + (tokenEmbeddings[tokenIndex * embeddingLength + k] * embedding_scale);
          }
          // Just store targetId in a vector, avoid One-Hot if possible
          float off_value = label_smoothing / vocabSize;
          float on_value = 1.0f - label_smoothing + off_value;

          // When filling h_targets:
          std::fill(h_targets.begin() + batch_token_offset * vocabSize, 
              h_targets.begin() + (batch_token_offset + 1) * vocabSize, 
              off_value);
          h_targets[batch_token_offset * vocabSize + targetId] = on_value;
        } else {
          // Explicitly handle padding for sequences that are too short
          // Fill h_inputs[batch_token_offset] with 0s (already done by zero-init)
        }
      }
    }

    current_batch.inputs.set_raw(h_inputs);
    current_batch.targets.set_raw(h_targets);

    return current_batch;
  }

  Mat<float>& forward(DataBatch & current_batch)  {
    const Mat<float>* x = &current_batch.inputs; 

    for (int i = 0; i < decoder_blocks.size(); ++i) {
      x = &decoder_blocks[i].forward(*x, current_batch.valid_seq_len); 
    }

    Mat<float>& final_result = const_cast<Mat<float>&>(*x);
    final_result.dirty();
    return final_result;
  }

  void main(int argc, char **argv) {
    bool dropout_enabled = false;

    for (int i = 0; i < argc; ++i) {
      std::string arg(argv[i]);
      if (arg == "--train") {
        train = true;
      } else if (arg == "--load") {
        load_weights = true;
        LOG << "loading model = " << filename;
      } else if (arg == "--epochs") {
        if (i + 1 < argc) {
          epochs = atoi(argv[i+1]);
        }
      } else if(arg == "--dropout") {
        if (i + 1 < argc) {
          dropout = atof(argv[i+1]);
          dropout_enabled = true;
        }
      } else if(arg == "--initial-learning-rate") {
        if (i + 1 < argc) {
          learningRate = initialLearningRate = atof(argv[i+1]);
        }
      } else if (arg == "--inference") {
        load_weights = true;
        inference = true; 
      } else if (arg == "--temperature") {
        if (i + 1 < argc) {
          temperature = atof(argv[i+1]);
        }
      }
    }  
    linear_logits.assign(seqLength * batchSize, vindex.size());
    probs_mat.assign(seqLength * batchSize, vindex.size());

    for (int i = 0; i < decoder_layers; ++i) {
      decoder_blocks.emplace_back(seqLength, batchSize, embeddingLength, num_heads, std::to_string(i), decoder_layers, weight_decay);
    }

    tokenEmbeddings.assign(embeddingLength * vocabSize, 0.0f);

    he_init(tokenEmbeddings, vocabSize, 0.01f);
    //normal_init(tokenEmbeddings, 0.02f);

    std::vector<int> inputIds;
    std::vector<int> validationInputIds;
    std::vector<float> d_tokenEmbeddings(tokenEmbeddings.size(), 0.0f);
    Adam optimizer_embeddings(tokenEmbeddings.size());

    for (const auto & line : training_lines) {
      for (int i = 0; i < line.size(); ++i) {
        char c = (line.at(i));
        int c_to_id = char_to_id.at(c);
        inputIds.push_back(c_to_id);
      }
      inputIds.push_back(char_to_id.at('\n'));
    }    

    training_lines.clear();
    training_lines.shrink_to_fit(); 

    for (const auto & line : validation_lines) {
      for (int i = 0; i < line.size(); ++i) {
        char c = (line.at(i));
        int c_to_id = char_to_id.at(c);
        validationInputIds.push_back(c_to_id);
      }
    }    
    validation_lines.clear();
    validation_lines.shrink_to_fit();


    LOG << "inputIds.size = " << inputIds.size() << " tokenEmbeddings.size = " << tokenEmbeddings.size() << " vocab.size = " << vocabSize << " validationInput.size = " << validationInputIds.size() 
        << " batchSize = " << batchSize << " num_heads = " << num_heads << " seqLength = " << seqLength << " warmup_epochs = " << warmup_epochs << " label_smoothing = " << label_smoothing
        << " embeddingLength = " << embeddingLength << " decoder_layers = " << decoder_blocks.size() << " temperature = " << temperature << " weight_decay = " << weight_decay
        << " initialLearningRate = " << initialLearningRate;
    if (dropout_enabled) {
      LOG << " dropout = " << dropout;
      for (auto & d : decoder_blocks) {
        d.enable_dropout(train, dropout);
      }
    } 
    if (load_weights) {
      if (train) {
        std::map<char, int> dummy_map;
        std::vector<float> dummy_vec;
        // This function consumes the vocab section and advances the file pointer
        load_vocab_mapping(dummy_map, dummy_vec); 
      }
      for (auto & d : decoder_blocks) {
        d.load();
      }
      load_embeddings(tokenEmbeddings); 
      linearLayer.load();
      finalLayerNorm.load();
    }

    if (train) {
      GlobalClipper clipper;
 
      auto training_start_time = std::chrono::high_resolution_clock::now();
      size_t training_checkpoint = 0; 

      steps_per_epoch = inputIds.size() / (batchSize * seqLength);
      if (steps_per_epoch < 1) steps_per_epoch = 1;
      warmup_steps = warmup_epochs * steps_per_epoch;
      LOG << "START OF TRAINING: "
          << "steps_per_epoch = " << steps_per_epoch;
      float best_validation_loss = std::numeric_limits<float>::max();
      float gradient_scale = 1.0f / (float)(batchSize * seqLength);
      for (int epoch = 0; epoch < epochs; ++epoch) {
        DataBatch current_batch;
        float current_norm = 0;
        float global_scale = 0;
        int batch_idx = 0;
        float total_epoch_loss = 0.0f;
        float total_validation_loss = 0.0f;
        int validation_batch_count = 0;
        // The main training loop now iterates for a fixed number of steps
        for (int step = 0; step < steps_per_epoch; ++step) {
          for (auto & d : decoder_blocks) {
            d.reset();
          }     

          finalLayerNorm.reset();
          linearLayer.reset();
          std::fill(d_tokenEmbeddings.begin(), d_tokenEmbeddings.end(), 0.0f); 
          global_step++;
          // --- LEARNING RATE SCHEDULING ---
          float min_lr = initialLearningRate * 0.1f;
          int total_training_steps = 200 * steps_per_epoch;
          if (global_step < warmup_steps) {
            // Linear warmup
            learningRate = initialLearningRate * (float)global_step / (float)warmup_steps;
          } else {
            // Cosine decay
            float progress = (float)(global_step - warmup_steps) / (float)(total_training_steps - warmup_steps);
            progress = std::min(1.0f, progress);
            float cos_out = 0.5f * (1.0f + std::cos(M_PI * progress));
            learningRate = min_lr + (initialLearningRate - min_lr) * cos_out;
          }
          // Ensure learning rate doesn't go below a minimum value
          learningRate = std::max(learningRate, 1e-8f);
          // --- ON-THE-FLY BATCH GENERATION ---

          // 1. Generate Random Start Indices for this Batch
          std::vector<int> batch_indices;
          for(int b = 0; b < batchSize; ++b) {
            int rnd = gen() % (inputIds.size() - seqLength - 1);
            batch_indices.push_back(rnd);
          }
          // 2. Create Batch using these random indices
          current_batch = createBatch(inputIds, batch_indices);
          // --- END OF BATCH GENERATION ---
          float sequence_loss = 0.0;
          Mat<float>& block_output = forward(current_batch);
          Mat<float>& final_norm_output = finalLayerNorm.forward(block_output);
          linearLayer.forward(final_norm_output, linear_logits);
          // 3. Extract logits and Compute Loss
          linear_logits.softmax(probs_mat);
          total_epoch_loss += Mat<float>::softCrossEntropyLoss(probs_mat, current_batch.targets, batchSize, seqLength);//current_batch.token_ids.size());

          Mat<float>::computeLogitGradients(probs_mat, current_batch.targets, linear_logits);
          linear_logits.scale(gradient_scale, linear_logits);
          // --- BACKWARD PASS ---
          linearLayer.backward(linear_logits, final_norm_output);
          // Backprop through the final LayerNorm BEFORE the decoders
          Mat<float>& d_input = finalLayerNorm.backward(final_norm_output);

          for (int i = decoder_blocks.size() - 1;i >= 0; --i) {
            d_input = decoder_blocks[i].backward(d_input);
          }
          // We iterate linearly through the gradient vector (d_input) 
          // and map it to the recorded token ID (current_batch.token_ids)
          std::vector<float>& d_input_vec = d_input.raw(); // Gets the underlying vector
          int total_tokens_in_batch = current_batch.token_ids.size();

          for (int i = 0; i < total_tokens_in_batch; ++i) {
            int tokenIndex = current_batch.token_ids[i];

            // Safety check for vocab bounds
            if (tokenIndex < 0 || tokenIndex >= vocabSize) continue;

            for (int k = 0; k < embeddingLength; ++k) {
              d_tokenEmbeddings[tokenIndex * embeddingLength + k] += 
                d_input_vec[i * embeddingLength + k] * embedding_scale;
            }
          }
          float max_norm = 1.0f;
 
          clipper.reset();
          clipper.accumulate(d_tokenEmbeddings);
          linearLayer.accumulate_gradients(clipper);
          finalLayerNorm.accumulate_gradients(clipper);
          for (auto & block : decoder_blocks) {
            block.accumulate_gradients(clipper);
          }
          current_norm = std::sqrt(clipper.total_sum_sq);
          float total_tokens = (float)(batchSize * seqLength);
          global_scale = clipper.get_global_scale(1.0f);
          for (auto & d : d_tokenEmbeddings)  d *= global_scale;

          optimizer_embeddings.update(tokenEmbeddings, d_tokenEmbeddings, learningRate, global_step, weight_decay, 1.0f);
          linearLayer.update_weights(learningRate, global_scale, global_step);
          finalLayerNorm.update_weights(learningRate, global_scale, global_step);
          // Update and reset the embeddings 
          for (auto & d : decoder_blocks) {
            d.update_weights(learningRate, global_scale, global_step);
          }

          if (Mat<float>::enable_arena) {
            if (epoch == 0 && step == 0) {
              // STEP 0 SPECIAL CASE:
              // During the very first step, layers might resize their internal matrices 
              // (e.g., LinearLayer resizing cache.input). 
              // We MUST allow this to happen and persist.
              // We capture the "High Water Mark" here. 
              // (Note: This includes Step 0's temporary garbage, but only 1 step's worth).
              training_checkpoint = global_arena.get_offset();
            } else {
              global_arena.set_offset(training_checkpoint);
            }
          }
          if (sig) {
            sig = false;
            output_file.open(filename);
            save_header();
            for (auto & d : decoder_blocks) {
              d.save();
            }     
            save_embeddings(tokenEmbeddings);
            linearLayer.save();
            finalLayerNorm.save();
            output_file.close();
            LOG << "Save Done!";
          }
        } // End of one training step.


        if (dropout_enabled) {
          for (auto & d : decoder_blocks) {
            d.enable_dropout(false, 0.0f);
          }
        }

        size_t val_start_checkpoint = global_arena.get_offset();
        // Calc validation loss here.
        int val_total_tokens = validationInputIds.size();
        // Stride = How many tokens we consume per step (e.g. 64 * 32)
        int stride = seqLength * batchSize; 

        for (int val_start = 0; val_start < val_total_tokens - stride; val_start += stride) {
          validation_batch_count++;

          // 1. Generate Sequential Indices for this Batch
          // If batchSize=2, seqLength=64, and val_start=0:
          // We want indices: [0, 64]
          std::vector<int> val_indices;
          for (int b = 0; b < batchSize; ++b) {
            val_indices.push_back(val_start + (b * seqLength));
          }

          // 2. Create Batch
          DataBatch validation_batch = createBatch(validationInputIds, val_indices);

          // --- END OF BATCH GENERATION ---

          float validation_loss = 0.0f;
          Mat<float> & block_output = forward(validation_batch);
          Mat<float> & final_norm_output = finalLayerNorm.forward(block_output);
          // 1. Load the ENTIRE batch into the linear layer at once
          // 2. Forward pass (Computes logits for all tokens in parallel)
          linearLayer.forward(final_norm_output, linear_logits);

          // 3. Compute Softmax correctly (Row-wise) using the GPU
          linear_logits.softmax(probs_mat);

          // 4. Compute Loss across all tokens
          validation_loss = Mat<float>::softCrossEntropyLoss(
              probs_mat, 
              validation_batch.targets, 
              batchSize, //validation_batch.token_ids.size()
              seqLength
              );

          // Accumulate average loss (validation_loss is already averaged by softCrossEntropyLoss)
          total_validation_loss += validation_loss;

          if (Mat<float>::enable_arena) {
            global_arena.set_offset(val_start_checkpoint);
          }
        }

        if (dropout_enabled) {
          for (auto & d : decoder_blocks) {
            d.enable_dropout(true, dropout);
          }
        }
        if (validation_batch_count > 0) {
          if (verbose) {
            for(int i = 0; i < decoder_blocks.size(); ++i) {
              for (int h = 0; h < num_heads; ++h) {
                LOG << "\n[DEBUG] Block " << i << ", Head " << h << " Attention Scores";
                decoder_blocks[i].attentionLayer.heads[h].cache.attention_scores.print("Softmax Scores");
              }
              decoder_blocks[i].print_attention();
            }
            debug_print_batch(current_batch);
          }
          float current_val_loss = total_validation_loss / validation_batch_count;
          LOG << "validation loss = " << current_val_loss
            << " training loss = " << total_epoch_loss / ((float)steps_per_epoch) 
            << " epoch = " << epoch << "/" << epochs 
            << " learningRate " << learningRate;
          LOG << "global_step = " << global_step << " normalized_norm = " << current_norm  << " scale = " << global_scale;
          if (current_val_loss < best_validation_loss) {
            best_validation_loss = current_val_loss;
            LOG << ">>> New Best Model Found! (Loss: " << best_validation_loss << "). Saving to model_best.data ...";

            output_file.open("model_best.data"); // distinct filename
            if (output_file.is_open()) {
              save_header();
              for (auto & d : decoder_blocks) {
                d.save();
              }
              save_embeddings(tokenEmbeddings); 
              linearLayer.save();
              finalLayerNorm.save();
              output_file.close();
            } else {
              LOG << "Error: Could not open model_best.data for writing.";
            }
          }
        }        
      }

      output_file.open(filename);
      save_header();
      for (auto & d : decoder_blocks) {
        d.save();
      }
      save_embeddings(tokenEmbeddings); 
      linearLayer.save();
      finalLayerNorm.save();
      output_file.close();

      auto training_end_time = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::seconds>(training_end_time - training_start_time);
      
      int hours = duration.count() / 3600;
      int minutes = (duration.count() % 3600) / 60;
      int seconds = duration.count() % 60;

      LOG << "--------------------------------------------------";
      LOG << "Total Training Time: " << hours << "h " << minutes << "m " << seconds << "s";
      LOG << "--------------------------------------------------";
    } else if (inference) {
      generate();
    }
  }

  void generate() {
    std::string line; 
    int num_tokens_to_generate = 200; 

    // 1. Capture the Arena state BEFORE the loop
    size_t generation_checkpoint = 0;
    if (Mat<float>::enable_arena) {
        generation_checkpoint = global_arena.get_offset();
    }

    while (true) {
        std::cout << "\n> " << std::flush;
        if (!std::getline(std::cin, line)) break;

        // DEBUG: Print what the model sees
        std::vector<float> debug_ids = generate_inputs(line);
        std::cout << "[DEBUG] Input Tokens: ";
        for(float f : debug_ids) {
            if(f != 0) std::cout << (int)f << " ";
        }
        std::cout << std::endl;

        for (int i = 0; i < num_tokens_to_generate; ++i) {
            // 1. Allocate & Explicitly Zero (Paranoia Mode)
            Mat<float> inference_input_mat(seqLength, embeddingLength);
            std::vector<float> h_buffer(seqLength * embeddingLength, 0.0f); // Zero init

            std::vector<float> current_ids = generate_inputs(line);
            int valid_len = std::min((int)line.size(), seqLength);

            // 2. Fill Buffer
            for (int t = 0; t < valid_len; ++t) {
                int id = (int)current_ids[t];
                if (id < 0 || id >= vocabSize) id = 0;

                std::vector<float> pez = pe(t, embeddingLength);
                for (int k = 0; k < embeddingLength; ++k) {
                    // Combine PE + Scaled Embedding
                    h_buffer[t * embeddingLength + k] = (pez[k]) + (tokenEmbeddings[id * embeddingLength + k] * embedding_scale);
                }
            }
            
            // 3. Upload
            inference_input_mat.set_raw(h_buffer);

            // 4. Forward
            Mat<float>* x = &inference_input_mat;
            for (auto& block : decoder_blocks) {
                x = &block.forward(*x, valid_len);
            }

            Mat<float> & output = finalLayerNorm.forward(*x);
            linearLayer.forward(output, linear_logits);
            
            float temp_scale = 1.0f / temperature;
            linear_logits.scale(temp_scale, linear_logits); 
            linear_logits.softmax(probs_mat); 
            std::vector<float> predictions = probs_mat.raw(); 

            // 5. Sample
            int last_pos = valid_len - 1;
            std::vector<float> last_token_probs(vocabSize);
            std::copy(predictions.begin() + (last_pos * vocabSize), 
                      predictions.begin() + ((last_pos + 1) * vocabSize), 
                      last_token_probs.begin());

            std::discrete_distribution<> dist(last_token_probs.begin(), last_token_probs.end());
            int predicted_id = dist(gen);
            if (predicted_id == 0) break;

            char predicted_char = '?';
            for (const auto & pair : char_to_id) {
              if (pair.second == predicted_id) {
                predicted_char = pair.first;
                break;
              }
            }
            std::cout << predicted_char << std::flush;
            line += predicted_char;
            // 2. RESET ARENA AFTER EACH TOKEN
            if (Mat<float>::enable_arena) {
                global_arena.set_offset(generation_checkpoint);
            }
        }
    }
  }

  void save_header() {
    if (!output_file.is_open()) {
      throw std::runtime_error("Cannot save vocab");
    }
    // 1. Save the size
    output_file << char_to_id.size() << "\n";
    // 2. Save the pairs. We cast char to int to handle newlines/special chars safely
    for (auto const& [key, val] : char_to_id) {
      output_file << val << " " << (int)key << "\n";
    }
    output_file << global_step << " \n";
  }
  bool load_vocab_mapping(std::map<char, int> & char_to_id, std::vector<float> & vindex) {
    if (!inputFile.is_open()) return false; // File doesn't exist yet

    int size;
    inputFile >> size;

    char_to_id.clear();
    vindex.clear();
    vindex.push_back(-1); // Preserve original index 0 placeholder

    int id, val;
    for (int i = 0; i < size; ++i) {
      inputFile >> id >> val;
      char c = (char)val;
      char_to_id[c] = id;
      vindex.push_back(id); 
    }
    inputFile >> global_step;
    return true;
  }


  void debug_print_batch(const DataBatch& batch, int num_samples = 2) {
    // 1. Create a reverse map (ID -> Char) for decoding
    std::map<int, char> id_to_char;
    for (const auto& pair : char_to_id) {
      id_to_char[pair.second] = pair.first;
    }

    std::cout << "\n=== DEBUG: Batch Content Decoded (" << num_samples << " samples) ===" << std::endl;

    // 2. Iterate through the requested number of batch rows
    for (int b = 0; b < std::min(num_samples, batchSize); ++b) {
      std::string input_str = "";
      std::string target_str = "";

      for (int t = 0; t < seqLength; ++t) {
        // Calculate flat index
        int idx = b * seqLength + t;

        // Decode Input
        if (idx < batch.token_ids.size()) {
          int in_id = batch.token_ids[idx];
          if (id_to_char.count(in_id)) {
            char c = id_to_char[in_id];
            // Handle newlines for cleaner printing
            if (c == '\n') input_str += "\\n"; 
            else input_str += c;
          } else {
            input_str += "?"; // Unknown token
          }
        }

        // Decode Target (Only if you applied Step 1 & 2)
        if (idx < batch.target_ids.size()) {
          int tgt_id = batch.target_ids[idx];
          if (id_to_char.count(tgt_id)) {
            char c = id_to_char[tgt_id];
            if (c == '\n') target_str += "\\n";
            else target_str += c;
          } else {
            target_str += "?";
          }
        }
      }

      std::cout << "[Batch " << b << "]\n";
      std::cout << "  IN:  \"" << input_str << "\"\n";
      std::cout << "  TGT: \"" << target_str << "\"\n";
      std::cout << "--------------------------------------------------" << std::endl;
    }
  }
};

template <typename T>
bool Mat<T>::enable_arena = true;

int main(int argc, char**argv) {
  if (Mat<float>::enable_arena) {
    global_arena.init(10ULL * 1024 * 1024 * 1024); 
  }
  signal(SIGHUP, signalCallbackHandler);
  int embeddingLength = 128;
  int seqLength = 64;
  int batchSize = 32;
  int num_heads = 4;
  int decoder_layers = 2;
  int warmup_epochs = 10;
  float weight_decay = 0.2f;
  float label_smoothing = 0.1f;
  bool train = false;
  
  output_file.precision(std::numeric_limits<float>::max_digits10);

  for (int i = 0; i < argc; ++i) {
    std::string arg(argv[i]);
    if(arg == "--seed") {
      if (i + 1 < argc) {
        seed = atoi(argv[i+1]);
      }
    } else if(arg == "--seq-length") {
      if (i + 1 < argc) {
        seqLength = atoi(argv[i+1]);
      }
    } else if (arg == "--decoder-layers") {
      if (i + 1 < argc) {
        decoder_layers = atoi(argv[i+1]);
      }
    } else if (arg == "--embedding-length") {
      if (i + 1 < argc) {
        embeddingLength = atoi(argv[i+1]);
      }
    } else if (arg == "--num-heads") {
      if (i + 1 < argc) {
        num_heads = atoi(argv[i+1]);
      }
    } else if (arg == "--batch-size") {
      if (i + 1 < argc) {
        batchSize = atoi(argv[i+1]);
      }
    } else if (arg == "--train") {
        train = true;
    } else if (arg == "--load") {
      if (i + 1 < argc) {
        filename = std::string(argv[i+1]);
      }
    } else if (arg == "--weight-decay") {
      if (i + 1 < argc) {
        weight_decay = atof(argv[i+1]); 
      }
    } else if (arg == "--verbose") {
      verbose = true;
    } else if (arg == "--warmup-epochs") {
      if (i + 1 < argc) {
        warmup_epochs = atoi(argv[i+1]);
      }
    } else if (arg == "--label-smoothing") {
      if (i + 1 < argc) {
        label_smoothing = atof(argv[i+1]); 
      }
    } 
  }
  Transformer t(embeddingLength, seqLength, batchSize, num_heads, decoder_layers, train, weight_decay, warmup_epochs, label_smoothing);

  t.main(argc, argv);
  
  if (Mat<float>::enable_arena) {
    global_arena.free_all();
  }
  destroy_rb_handle();
  return 0;
}


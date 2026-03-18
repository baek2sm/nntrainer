// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   mt5_causallm.h
 * @date   18 Mar 2026
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note   Please refer to the following code:
 *  https://github.com/huggingface/transformers/blob/main/src/transformers/models/mt5/modeling_mt5.py
 *
 * @note   mT5 is an encoder-decoder model. Key architectural differences:
 *         - Encoder: bidirectional self-attention (is_causal=false)
 *         - Decoder: causal self-attention + cross-attention
 *         - No RoPE; uses relative position bias (NYI, use_rope=false)
 *         - Gated GeLU FFN (GeGLU)
 *         - RMSNorm (T5LayerNorm)
 *
 *         This implementation uses TWO separate nntrainer models:
 *         1. encoder_model: processes source tokens, outputs hidden states
 *         2. model (decoder): takes decoder tokens + pre-computed cross-attn K/V
 *
 *         The cross-attention K/V are computed once by running the encoder, then
 *         projecting the encoder output through per-layer K/V FC weights stored
 *         in a separate "kv_projection_model".
 */

#ifndef __MT5_CAUSAL_LM_H__
#define __MT5_CAUSAL_LM_H__

#include <causal_lm.h>

namespace causallm {

/**
 * @brief MT5Transformer class
 * @note  Base class for mT5 encoder-decoder architecture.
 *        Manages encoder model construction and encoder-specific parameters.
 */
class MT5Transformer : virtual public Transformer {

public:
  static constexpr const char *architectures = "MT5Transformer";

  MT5Transformer(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(sanitizeConfig(cfg), generation_cfg, nntr_cfg) {}

  virtual ~MT5Transformer() = default;

protected:
  static json &sanitizeConfig(json &cfg);

  /**
   * @brief Setup parameters for MT5 model
   */
  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;

  /**
   * @brief Construct decoder model (overrides base class)
   */
  void constructModel() override;

  /**
   * @brief Construct encoder model
   */
  void constructEncoder();

  /**
   * @brief Construct the KV projection model
   * @note  This model projects encoder output to cross-attention K/V
   *        for all decoder layers. Run once after encoder to get K/V.
   */
  void constructKVProjectionModel();

  /**
   * @brief Create encoder self-attention block
   */
  std::vector<LayerHandle>
  createEncoderBlock(const int layer_id, std::string input_name);

  /**
   * @brief Create decoder block (self-attn + cross-attn + FFN)
   */
  std::vector<LayerHandle>
  createTransformerDecoderBlock(const int layer_id,
                                std::string input_name) override;

  /**
   * @brief Create self-attention (no RoPE for T5)
   */
  std::vector<LayerHandle> createAttention(const int layer_id, int seq_len,
                                           int n_heads, int head_dim,
                                           std::string query_name,
                                           std::string key_name,
                                           std::string value_name) override;

  /**
   * @brief Create cross-attention block
   * @param layer_id  Decoder layer index
   * @param query_name  Name of the layer providing Q input (decoder hidden)
   * @param cross_k_input  Name of the input layer for pre-computed K
   * @param cross_v_input  Name of the input layer for pre-computed V
   */
  std::vector<LayerHandle>
  createCrossAttention(const int layer_id, std::string query_name,
                       std::string cross_k_input, std::string cross_v_input);

  /**
   * @brief Create GeGLU feed-forward block
   */
  std::vector<LayerHandle> createMlp(const int layer_id, int dim,
                                     int hidden_dim,
                                     std::string input_name) override;

  void registerCustomLayers() override;

  /** Encoder model */
  ModelHandle encoder_model;
  /** KV projection model: projects encoder output to cross-attn K/V */
  ModelHandle kv_proj_model;

  /** Encoder-specific parameters */
  int NUM_ENCODER_LAYERS;
  int NUM_DECODER_LAYERS;
  int D_KV; /**< per-head key/value dimension */

  /** Buffers for cross-attention K/V (per decoder layer) */
  std::vector<float *> cross_k_buffers;
  std::vector<float *> cross_v_buffers;
  float *encoder_output_buffer = nullptr;
  unsigned int encoder_seq_len = 0;
};

/**
 * @brief MT5CausalLM class
 * @note  Encoder-decoder model for conditional generation.
 *        First model in nntrainer CausalLM to use cross-attention.
 */
class MT5CausalLM : public CausalLM, public MT5Transformer {

public:
  static constexpr const char *architectures = "MT5ForConditionalGeneration";

  MT5CausalLM(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(sanitizeConfig(cfg), generation_cfg, nntr_cfg,
                ModelType::CAUSALLM),
    CausalLM(sanitizeConfig(cfg), generation_cfg, nntr_cfg),
    MT5Transformer(sanitizeConfig(cfg), generation_cfg, nntr_cfg) {}

  virtual ~MT5CausalLM();

  /**
   * @brief Initialize both encoder and decoder models
   */
  void initialize() override;

  /**
   * @brief Load weights for both encoder and decoder
   */
  void load_weight(const std::string &weight_path) override;

  /**
   * @brief Run encoder-decoder generation
   */
  void run(const WSTR prompt, bool do_sample = false,
           const WSTR system_prompt = "",
           const WSTR tail_prompt = "") override;

  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override {
    CausalLM::setupParameters(cfg, generation_cfg, nntr_cfg);
    MT5Transformer::setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  void registerCustomLayers() override;

private:
  /**
   * @brief Run encoder on source tokens and compute cross-attn K/V
   * @param input_ids  Source token IDs
   * @param input_len  Number of source tokens
   */
  void runEncoder(const std::vector<int64_t> &input_ids,
                  unsigned int input_len);
};

} // namespace causallm

#endif /* __MT5_CAUSAL_LM_H__ */

// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   gemma3_causallm.h
 * @date   24 Dec 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __GEMMA3_CAUSAL_LM_H__
#define __GEMMA3_CAUSAL_LM_H__ __GEMMA3_CAUSAL_LM_H__

#include <causal_lm.h>

namespace causallm {

/**
 * @brief Gemma3CausalLM class
 */
class Gemma3CausalLM : public CausalLM {

public:
  static constexpr const char *architectures = "Gemma3ForCausalLM";

  Gemma3CausalLM(json &cfg, json &generation_cfg, json &nntr_cfg) :
    CausalLM(sanitizeConfig(cfg), sanitizeGenerationConfig(generation_cfg, cfg),
             nntr_cfg) {
    if (cfg.contains("layer_types")) {
      layer_types = cfg["layer_types"].get<std::vector<std::string>>();
    }
    EMBEDDING_SCALE = std::sqrt(static_cast<float>(cfg["hidden_size"]));
  }

  virtual ~Gemma3CausalLM() = default;

private:
  static json &sanitizeConfig(json &cfg);
  static json &sanitizeGenerationConfig(json &gen_cfg, const json &cfg);

  std::vector<std::string> layer_types;

public:
  std::vector<LayerHandle> createAttention(const int layer_id, int seq_len,
                                           int n_heads, int head_dim,
                                           std::string query_name,
                                           std::string key_name,
                                           std::string value_name) override;

  std::vector<LayerHandle>
  createTransformerDecoderBlock(const int layer_id, std::string input_name);

  virtual void setupParameters(json &cfg, json &generation_cfg,
                               json &nntr_cfg) override;

  std::vector<LayerHandle> createMlp(const int layer_id, int dim,
                                     int hidden_dim,
                                     std::string input_name) override;

  void registerCustomLayers() override;
};
} // namespace causallm

#endif /* __GEMMA3_CAUSAL_LM_H__ */

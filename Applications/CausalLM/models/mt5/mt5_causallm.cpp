// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   mt5_causallm.cpp
 * @date   12 March 2026
 * @brief  This defines an mT5 decoder path for CausalLM runtime.
 * @see    https://github.com/nntrainer/nntrainer
 * @author nntrainer contributors
 * @bug    No known bugs except for NYI items
 */

#include <mt5_causallm.h>

#include <algorithm>

#include <llm_util.hpp>

namespace causallm {

json &MT5Transformer::sanitizeConfig(json &cfg) {
  if (!cfg.contains("hidden_size") && cfg.contains("d_model")) {
    cfg["hidden_size"] = cfg["d_model"];
  }

  if (!cfg.contains("intermediate_size") && cfg.contains("d_ff")) {
    cfg["intermediate_size"] = cfg["d_ff"];
  }

  if (!cfg.contains("num_hidden_layers")) {
    if (cfg.contains("num_decoder_layers")) {
      cfg["num_hidden_layers"] = cfg["num_decoder_layers"];
    } else if (cfg.contains("num_layers")) {
      cfg["num_hidden_layers"] = cfg["num_layers"];
    }
  }

  if (!cfg.contains("num_attention_heads") && cfg.contains("num_heads")) {
    cfg["num_attention_heads"] = cfg["num_heads"];
  }

  if (!cfg.contains("num_key_value_heads") &&
      cfg.contains("num_attention_heads")) {
    cfg["num_key_value_heads"] = cfg["num_attention_heads"];
  }

  if (!cfg.contains("head_dim")) {
    if (cfg.contains("d_kv")) {
      cfg["head_dim"] = cfg["d_kv"];
    } else if (cfg.contains("hidden_size") &&
               cfg.contains("num_attention_heads")) {
      auto hidden = cfg["hidden_size"].get<int>();
      auto heads = std::max(1, cfg["num_attention_heads"].get<int>());
      cfg["head_dim"] = hidden / heads;
    }
  }

  if (!cfg.contains("max_position_embeddings")) {
    if (cfg.contains("n_positions")) {
      cfg["max_position_embeddings"] = cfg["n_positions"];
    } else {
      cfg["max_position_embeddings"] = 2048;
    }
  }

  if (!cfg.contains("rope_theta")) {
    cfg["rope_theta"] = 1000000U;
  }

  if (!cfg.contains("rms_norm_eps")) {
    if (cfg.contains("layer_norm_epsilon")) {
      cfg["rms_norm_eps"] = cfg["layer_norm_epsilon"];
    } else {
      cfg["rms_norm_eps"] = 1e-6f;
    }
  }

  if (!cfg.contains("tie_word_embeddings")) {
    cfg["tie_word_embeddings"] = true;
  }

  if (!cfg.contains("is_causal")) {
    cfg["is_causal"] = true;
  }

  if (!cfg.contains("bos_token_id")) {
    if (cfg.contains("decoder_start_token_id") &&
        !cfg["decoder_start_token_id"].is_null()) {
      cfg["bos_token_id"] = cfg["decoder_start_token_id"];
    } else if (cfg.contains("pad_token_id") && !cfg["pad_token_id"].is_null()) {
      cfg["bos_token_id"] = cfg["pad_token_id"];
    } else {
      cfg["bos_token_id"] = 0;
    }
  }

  if (!cfg.contains("eos_token_id")) {
    cfg["eos_token_id"] = 1;
  }

  return cfg;
}

json &MT5Transformer::sanitizeGenerationConfig(json &gen_cfg,
                                               const json &cfg) {
  if (!gen_cfg.contains("eos_token_id") || gen_cfg["eos_token_id"].is_null()) {
    if (cfg.contains("eos_token_id")) {
      gen_cfg["eos_token_id"] = cfg["eos_token_id"];
    }
  } else if (gen_cfg["eos_token_id"].is_number()) {
    gen_cfg["eos_token_id"] =
      std::vector<unsigned int>{gen_cfg["eos_token_id"].get<unsigned int>()};
  }

  if (!gen_cfg.contains("bos_token_id") || gen_cfg["bos_token_id"].is_null()) {
    if (cfg.contains("decoder_start_token_id") &&
        !cfg["decoder_start_token_id"].is_null()) {
      gen_cfg["bos_token_id"] = cfg["decoder_start_token_id"];
    } else if (cfg.contains("bos_token_id") && !cfg["bos_token_id"].is_null()) {
      gen_cfg["bos_token_id"] = cfg["bos_token_id"];
    } else if (cfg.contains("pad_token_id") && !cfg["pad_token_id"].is_null()) {
      gen_cfg["bos_token_id"] = cfg["pad_token_id"];
    } else {
      gen_cfg["bos_token_id"] = 0;
    }
  }

  return gen_cfg;
}

std::vector<LayerHandle> MT5Transformer::createMlp(const int layer_id, int dim,
                                                   int hidden_dim,
                                                   std::string input_name) {
  std::vector<LayerHandle> layers;

  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_gate"),
     withKey("unit", hidden_dim), withKey("disable_bias", "true"),
     withKey("input_layers", input_name),
     withKey("weight_initializer", "ones")}));

  layers.push_back(createLayer(
    "activation",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_gate_gelu"),
     withKey("activation", "tanh_gelu"),
     withKey("input_layers",
             "layer" + std::to_string(layer_id) + "_ffn_gate")}));

  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_up"),
     withKey("unit", hidden_dim), withKey("disable_bias", "true"),
     withKey("input_layers", input_name),
     withKey("weight_initializer", "ones")}));

  layers.push_back(createLayer(
    "multiply",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_geglu"),
     withKey("input_layers", "layer" + std::to_string(layer_id) +
                               "_ffn_gate_gelu,layer" +
                               std::to_string(layer_id) + "_ffn_up")}));

  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_down"),
     withKey("unit", dim), withKey("disable_bias", "true"),
     withKey("input_layers", "layer" + std::to_string(layer_id) + "_ffn_geglu"),
     withKey("weight_initializer", "ones")}));

  return layers;
}

} // namespace causallm

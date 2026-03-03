// SPDX-License-Identifier: Apache-2.0
#include "mt5_transformer.h"

#include <layer.h>
#include <model.h>

#include <algorithm>
#include <cmath>

namespace causallm {

MT5Transformer::MT5Transformer(json &cfg, json &generation_cfg,
                               json &nntr_cfg) :
  Transformer(sanitizeConfig(cfg),
              sanitizeGenerationConfig(generation_cfg, cfg), nntr_cfg) {}

void MT5Transformer::setupParameters(json &cfg, json &generation_cfg,
                                     json &nntr_cfg) {
  Transformer::setupParameters(cfg, generation_cfg, nntr_cfg);

  is_gated_act = cfg.value("is_gated_act", false);
  layer_norm_epsilon = cfg.value("layer_norm_epsilon", 1e-6f);
  dense_act_fn = cfg.value("dense_act_fn", "relu");
  d_ff = cfg.value("d_ff", 2048);
  d_kv = cfg.value("d_kv", 64);
}

std::vector<LayerHandle>
MT5Transformer::createTransformerDecoderBlock(const int layer_id,
                                              std::string input_name) {
  std::vector<LayerHandle> layers;
  layers = Transformer::createTransformerDecoderBlock(layer_id, input_name);
  return layers;
}

std::vector<LayerHandle>
MT5Transformer::createAttention(const int layer_id, int seq_len, int n_heads,
                                int head_dim, std::string query_name,
                                std::string key_name, std::string value_name) {
  std::vector<LayerHandle> layers;
  layers = Transformer::createAttention(layer_id, seq_len, n_heads, head_dim,
                                        query_name, key_name, value_name);
  return layers;
}

std::vector<LayerHandle> MT5Transformer::createMlp(const int layer_id, int dim,
                                                   int hidden_dim,
                                                   std::string input_name) {
  std::vector<LayerHandle> layers;
  layers = Transformer::createMlp(layer_id, dim, hidden_dim, input_name);
  return layers;
}

void MT5Transformer::registerCustomLayers() {
  Transformer::registerCustomLayers();
}

json &MT5Transformer::sanitizeConfig(json &cfg) {
  if (cfg.contains("d_model") && !cfg["d_model"].is_null()) {
    cfg["hidden_size"] = cfg["d_model"];
  }

  if (cfg.contains("num_layers") && !cfg["num_layers"].is_null()) {
    cfg["num_hidden_layers"] = cfg["num_layers"];
  }

  if (cfg.contains("num_heads") && !cfg["num_heads"].is_null()) {
    cfg["num_attention_heads"] = cfg["num_heads"];
  }

  if (cfg.contains("d_ff") && !cfg["d_ff"].is_null()) {
    cfg["intermediate_size"] = cfg["d_ff"];
  }

  if (cfg.contains("layer_norm_epsilon") &&
      !cfg["layer_norm_epsilon"].is_null()) {
    cfg["rms_norm_eps"] = cfg["layer_norm_epsilon"];
  }

  if (!cfg.contains("head_dim") || cfg["head_dim"].is_null()) {
    if (cfg.contains("d_kv") && !cfg["d_kv"].is_null()) {
      cfg["head_dim"] = cfg["d_kv"];
    } else if (cfg.contains("hidden_size") &&
               cfg.contains("num_attention_heads") &&
               !cfg["hidden_size"].is_null() &&
               !cfg["num_attention_heads"].is_null()) {
      cfg["head_dim"] =
        cfg["hidden_size"].get<int>() / cfg["num_attention_heads"].get<int>();
    }
  }

  if (!cfg.contains("num_key_value_heads") ||
      cfg["num_key_value_heads"].is_null()) {
    if (cfg.contains("num_attention_heads") &&
        !cfg["num_attention_heads"].is_null()) {
      cfg["num_key_value_heads"] = cfg["num_attention_heads"];
    }
  }

  if (!cfg.contains("max_position_embeddings") ||
      cfg["max_position_embeddings"].is_null()) {
    cfg["max_position_embeddings"] = 512;
  }

  if (!cfg.contains("rope_theta") || cfg["rope_theta"].is_null()) {
    cfg["rope_theta"] = 10000;
  }

  if (!cfg.contains("is_causal") || cfg["is_causal"].is_null()) {
    cfg["is_causal"] = false;
  }

  if (!cfg.contains("use_bidirectional_attention") ||
      cfg["use_bidirectional_attention"].is_null()) {
    cfg["use_bidirectional_attention"] = true;
  }

  return cfg;
}

json &MT5Transformer::sanitizeGenerationConfig(json &gen_cfg, const json &cfg) {
  return gen_cfg;
}

void MT5Transformer::constructModel() { Transformer::constructModel(); }

} // namespace causallm

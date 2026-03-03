// SPDX-License-Identifier: Apache-2.0
#ifndef __MT5_TRANSFORMER_H__
#define __MT5_TRANSFORMER_H__

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#define WSTR std::wstring
#define WCHAR_P wchar_t *
#else
#define WIN_EXPORT
#define WSTR std::string
#define WCHAR_P std::string &
#endif

#include "../transformer.h"

namespace causallm {

class MT5Transformer : virtual public Transformer {
public:
  static constexpr const char *architectures = "MT5Transformer";
  MT5Transformer(json &cfg, json &generation_cfg, json &nntr_cfg);
  virtual ~MT5Transformer() = default;
  static json &sanitizeConfig(json &cfg);
  static json &sanitizeGenerationConfig(json &gen_cfg, const json &cfg);

protected:
  virtual void setupParameters(json &cfg, json &generation_cfg,
                               json &nntr_cfg) override;
  virtual std::vector<LayerHandle>
  createTransformerDecoderBlock(const int layer_id,
                                std::string input_name) override;
  virtual std::vector<LayerHandle>
  createAttention(const int layer_id, int seq_len, int n_heads, int head_dim,
                  std::string query_name, std::string key_name,
                  std::string value_name) override;
  virtual std::vector<LayerHandle> createMlp(const int layer_id, int dim,
                                             int hidden_dim,
                                             std::string input_name) override;
  virtual void registerCustomLayers() override;
  virtual void constructModel() override;

  // MT5-specific parameters
  bool is_gated_act;
  float layer_norm_epsilon;
  std::string dense_act_fn;
  int d_ff;
  int d_kv;
};

} // namespace causallm

#endif /* __MT5_TRANSFORMER_H__ */

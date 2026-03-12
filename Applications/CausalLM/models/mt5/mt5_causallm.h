// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   mt5_causallm.h
 * @date   12 March 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author nntrainer contributors
 * @bug    No known bugs except for NYI items
 * @brief  This defines an mT5 decoder path for CausalLM runtime.
 */

#ifndef __MT5_CAUSAL_LM_H__
#define __MT5_CAUSAL_LM_H__

#include <causal_lm.h>

namespace causallm {

/**
 * @brief MT5Transformer class
 */
class MT5Transformer : virtual public Transformer {

public:
  static constexpr const char *architectures = "MT5Transformer";

  MT5Transformer(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(sanitizeConfig(cfg),
                sanitizeGenerationConfig(generation_cfg, cfg), nntr_cfg) {}

  virtual ~MT5Transformer() = default;

protected:
  static json &sanitizeConfig(json &cfg);
  static json &sanitizeGenerationConfig(json &gen_cfg, const json &cfg);

public:
  std::vector<LayerHandle> createMlp(const int layer_id, int dim, int hidden_dim,
                                     std::string input_name) override;
};

/**
 * @brief MT5CausalLM class
 */
class MT5CausalLM : public CausalLM, public MT5Transformer {

public:
  static constexpr const char *architectures = "MT5ForConditionalGeneration";

  MT5CausalLM(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(sanitizeConfig(cfg),
                sanitizeGenerationConfig(generation_cfg, cfg), nntr_cfg,
                ModelType::CAUSALLM),
    CausalLM(sanitizeConfig(cfg), sanitizeGenerationConfig(generation_cfg, cfg),
             nntr_cfg),
    MT5Transformer(sanitizeConfig(cfg),
                   sanitizeGenerationConfig(generation_cfg, cfg), nntr_cfg) {}

  virtual ~MT5CausalLM() = default;
};

} // namespace causallm

#endif /* __MT5_CAUSAL_LM_H__ */

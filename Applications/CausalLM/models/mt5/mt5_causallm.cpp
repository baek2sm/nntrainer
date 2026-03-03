// SPDX-License-Identifier: Apache-2.0
#include "mt5_causallm.h"

namespace causallm {

MT5CausalLM::MT5CausalLM(json &cfg, json &generation_cfg, json &nntr_cfg) :
  Transformer(cfg, generation_cfg, nntr_cfg, ModelType::CAUSALLM),
  MT5Transformer(cfg, generation_cfg, nntr_cfg),
  CausalLM(cfg, generation_cfg, nntr_cfg) {}

void MT5CausalLM::setupParameters(json &cfg, json &generation_cfg,
                                  json &nntr_cfg) {
  CausalLM::setupParameters(cfg, generation_cfg, nntr_cfg);
  MT5Transformer::setupParameters(cfg, generation_cfg, nntr_cfg);
}

void MT5CausalLM::constructModel() { CausalLM::constructModel(); }

void MT5CausalLM::registerCustomLayers() {
  CausalLM::registerCustomLayers();
  MT5Transformer::registerCustomLayers();
}

} // namespace causallm

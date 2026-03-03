// SPDX-License-Identifier: Apache-2.0
#ifndef __MT5_CAUSAL_LM_H__
#define __MT5_CAUSAL_LM_H__

#include "../causal_lm.h"
#include "mt5_transformer.h"

namespace causallm {

class MT5CausalLM : public CausalLM, virtual public MT5Transformer {
public:
  static constexpr const char *architectures = "MT5ForCausalLM";
  MT5CausalLM(json &cfg, json &generation_cfg, json &nntr_cfg);
  virtual ~MT5CausalLM() = default;

  virtual void setupParameters(json &cfg, json &generation_cfg,
                               json &nntr_cfg) override;
  virtual void constructModel() override;
  virtual void registerCustomLayers() override;

private:
  // MT5-specific parameters for CausalLM
};

} // namespace causallm

#endif /* __MT5_CAUSAL_LM_H__ */

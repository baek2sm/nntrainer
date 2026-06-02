// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 SeungBaek Hong <sb92.hong@samsung.com>
// @file lfm2_vl_projector.h
// @brief LFM2.5-VL multi_modal_projector: pixel_unshuffle + FC + GELU + FC

#ifndef __LFM2_VL_PROJECTOR_H__
#define __LFM2_VL_PROJECTOR_H__

#include <vjepa_projector.h>

namespace causallm {

class Lfm2VlProjector : public VjepaProjector {
public:
  static constexpr const char *architectures = "Lfm2VlProjector";

  Lfm2VlProjector(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg, ModelType::MODEL),
    VjepaProjector(cfg, generation_cfg, nntr_cfg) {
    setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  ~Lfm2VlProjector() override = default;

  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;
  std::pair<Tensor, Tensor> constructModel() override;

private:
  unsigned int FC1_DIM = 2048; ///< hidden dim of first FC layer
};

} // namespace causallm

#endif /* __LFM2_VL_PROJECTOR_H__ */

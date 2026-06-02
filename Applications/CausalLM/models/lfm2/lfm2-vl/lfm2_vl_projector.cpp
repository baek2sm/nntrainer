// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 SeungBaek Hong <sb92.hong@samsung.com>
// @file lfm2_vl_projector.cpp
// @brief LFM2.5-VL multi_modal_projector: pixel_unshuffle(factor=2) + FC1 + GELU + FC2

#include "lfm2_vl_projector.h"
#include <llm_util.hpp>

namespace causallm {

void Lfm2VlProjector::setupParameters(json &cfg, json &generation_cfg,
                                      json &nntr_cfg) {
  VjepaProjector::setupParameters(cfg, generation_cfg, nntr_cfg);
  FC1_DIM = cfg.value("merger_hidden_1", 2048u);
}

std::pair<Tensor, Tensor> Lfm2VlProjector::constructModel() {
  // Input: after pixel_unshuffle [B, 1, OUTPUT_TOKENS, INPUT_DIM]
  Tensor input({BATCH_SIZE, 1, OUTPUT_TOKENS, INPUT_DIM}, "input0");
  Tensor h = input;

  LayerHandle fc1(createLayer(
    "fully_connected",
    {withKey("name", "proj_fc1"), withKey("unit", std::to_string(FC1_DIM)),
     withKey("disable_bias", "false")}));
  h = fc1(h);

  LayerHandle act(createLayer("activation", {withKey("name", "proj_gelu"),
                                             withKey("activation", "gelu")}));
  h = act(h);

  LayerHandle fc2(createLayer(
    "fully_connected",
    {withKey("name", "proj_fc2"), withKey("unit", std::to_string(TEXT_DIM)),
     withKey("disable_bias", "false")}));
  h = fc2(h);

  return {input, h};
}

} // namespace causallm

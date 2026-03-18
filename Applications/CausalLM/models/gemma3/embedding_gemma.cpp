// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   embedding_gemma.cpp
 * @date   11 Jan 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This file defines Gemma3 Embedding model
 */

#include <embedding_gemma.h>
#include <iostream>

namespace causallm {

void EmbeddingGemma::setupParameters(json &cfg, json &generation_cfg,
                                     json &nntr_cfg) {
  std::cout << "[DEBUG] EmbeddingGemma::setupParameters start" << std::endl;
  Gemma3Transformer::setupParameters(cfg, generation_cfg, nntr_cfg);
  SentenceTransformer::setupParameters(cfg, generation_cfg, nntr_cfg);
  std::cout << "[DEBUG] EmbeddingGemma::setupParameters end" << std::endl;
}

void EmbeddingGemma::registerCustomLayers() {
  std::cout << "[DEBUG] EmbeddingGemma::registerCustomLayers start" << std::endl;
  SentenceTransformer::registerCustomLayers();
  Gemma3Transformer::registerCustomLayers();
  std::cout << "[DEBUG] EmbeddingGemma::registerCustomLayers end" << std::endl;
}

} // namespace causallm

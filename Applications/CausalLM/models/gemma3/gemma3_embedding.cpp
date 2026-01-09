// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   gemma3_embedding.cpp
 * @date   09 Jan 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This file defines Gemma3 Embedding model
 */

#include <gemma3_embedding.h>

namespace causallm {

void Gemma3Embedding::registerCustomLayers() {
  Embedding::registerCustomLayers();
  Gemma3Transformer::registerCustomLayers();
}

} // namespace causallm

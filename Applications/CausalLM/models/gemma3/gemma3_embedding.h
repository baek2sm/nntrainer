// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   gemma3_embedding.h
 * @date   09 Jan 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note   This gemma3_embedding.h constructs a class for Gemma3-based Embedding
 * model.
 */

#ifndef __GEMMA3_EMBEDDING_H__
#define __GEMMA3_EMBEDDING_H__

#include <embedding.h>
#include <gemma3_causallm.h>

namespace causallm {

/**
 * @brief Gemma3Embedding Class
 */
class Gemma3Embedding : public Embedding, public Gemma3Transformer {

public:
  static constexpr const char *architectures = "Gemma3Embedding";

  /**
   * @brief Construct a new Gemma3Embedding object
   * @param cfg Configuration for the model
   * @param generation_cfg Configuration for generation
   * @param nntr_cfg Configuration for nntrainer
   */
  Gemma3Embedding(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg, ModelType::EMBEDDING),
    Embedding(cfg, generation_cfg, nntr_cfg),
    Gemma3Transformer(cfg, generation_cfg, nntr_cfg) {}

  /**
   * @brief Destroy the Gemma3Embedding object
   */
  virtual ~Gemma3Embedding() = default;

  /**
   * @brief register CustomLayers
   */
  void registerCustomLayers() override;

protected:
  /**
   * @brief Setup the parameters for the Embedding model
   */
  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override {
    Embedding::setupParameters(cfg, generation_cfg, nntr_cfg);
    Gemma3Transformer::setupParameters(cfg, generation_cfg, nntr_cfg);
    
    // Explicitly set Embedding's scale correctly as it has its own Copy of Transformer
    Embedding::EMBEDDING_SCALE = std::sqrt(Embedding::DIM);
  }
};

} // namespace causallm

#endif // __GEMMA3_EMBEDDING_H__

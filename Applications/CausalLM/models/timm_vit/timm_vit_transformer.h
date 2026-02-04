// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   timm_vit_transformer.h
 * @date   28 Jan 2026
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note   This timm_vit_transformer.h constructs a class for timm ViT model
 * compatible with the PyTorch timm library. It supports Vision Transformer
 * models with patch embedding and transformer blocks.
 */

#ifndef __TIMM_VIT_TRANSFORMER_H__
#define __TIMM_VIT_TRANSFORMER_H__

#include <transformer.h>

namespace causallm {

/**
 * @brief TimmViTTransformer class
 */
class TimmViTTransformer : virtual public Transformer {

public:
  static constexpr const char *architectures = "TimmViT";

  TimmViTTransformer(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg, ModelType::EMBEDDING) {}

  virtual ~TimmViTTransformer() = default;

public:
  std::vector<LayerHandle> createPatchEmbed();
  std::vector<LayerHandle> createAttention(const int layer_id,
                                           const std::string &input_name);
  std::vector<LayerHandle> createMlp(const int layer_id,
                                     const std::string &input_name);
  std::vector<LayerHandle>
  createTransformerBlock(const int layer_id, const std::string &input_name);

protected:
  void constructModel() override;

  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;

  virtual std::vector<LayerHandle>
  createTransformerDecoderBlock(const int layer_id, std::string input_name);

  void registerCustomLayers() override;

  /**
   * @brief Run the model (override for ViT specific behavior)
   */
  void run(const WSTR prompt, bool do_sample = false,
           const WSTR system_prompt = "", const WSTR tail_prompt = "") override;

  /**
   * @brief Initialize (override to skip compile/initialize for TimmViT)
   */
  void initialize() override;
};

} // namespace causallm

#endif /* __TIMM_VIT_TRANSFORMER_H__ */

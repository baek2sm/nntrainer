// SPDX-License-Identifier: Apache-2.0
/**
 * @file   lfm2_vl_model.h
 * @date   04 June 2026
 * @brief  LFM2-VL multimodal model: SigLIP2 vision + connector + LFM2 LM.
 *
 *         Pipeline:
 *           preprocess image -> SigLIP2 ViT -> pixel-unshuffle x2
 *           -> MLP connector (3072->2048->1024) -> splice at image_token_id=396
 *           -> Lfm2CausalLM::run_with_embeddings -> generate tokens.
 *
 *         The V-JEPA/VoRA path in Lfm2CausalLM is preserved unchanged.
 * @author SeungBaek Hong <baek2sm@naver.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __LFM2_VL_MODEL_H__
#define __LFM2_VL_MODEL_H__

#include <lfm2_causallm.h>
#include <lfm2_vl_connector.h>
#include <lfm2_vl_vision_transformer.h>

#include <memory>
#include <string>
#include <vector>

namespace causallm {

/**
 * @brief LFM2-VL multimodal model (LiquidAI LFM2-VL-450M architecture).
 *
 * Owns:
 *   - Lfm2VlVisionTransformer  (SigLIP2 ViT encoder)
 *   - Lfm2VlConnector          (pixel-unshuffle + MLP)
 *   - Lfm2CausalLM             (LFM2 decoder LM, via composition)
 *
 * Config structure expected in config.json:
 * {
 *   "architectures": ["Lfm2VlForConditionalGeneration"],
 *   "model_type": "lfm2_vl",
 *   "text_config": { ...LFM2 LM config... },
 *   "vision_config": { ...SigLIP2 ViT config... },
 *   "image_token_id": 396,
 *   "projector_hidden_size": 2048,
 *   "downsample_factor": 2
 * }
 */
class Lfm2VlForConditionalGeneration {
public:
  static constexpr const char *architectures =
    "Lfm2VlForConditionalGeneration";
  static constexpr int DEFAULT_IMAGE_TOKEN_ID = 396;

  /**
   * @brief Construct the VL model from config / nntr_config.
   *
   * @param cfg            Full model config.json (contains text_config +
   *                       vision_config + top-level VL fields).
   * @param generation_cfg generation_config.json (may be empty object).
   * @param nntr_cfg       nntr_config.json (model_file_name, vision paths,
   *                       connector paths, etc.).
   */
  Lfm2VlForConditionalGeneration(json &cfg, json &generation_cfg,
                                 json &nntr_cfg);

  /**
   * @brief Initialize both sub-models (build graphs, compile).
   */
  void initialize();

  /**
   * @brief Load weights for ViT, connector, and LM.
   *
   * nntr_cfg keys used:
   *   "model_file_name"       -> LM weight binary
   *   "vision_model_file"     -> ViT weight binary (optional; skipped if absent)
   *   "connector_model_file"  -> connector weight binary (optional; skipped if absent)
   */
  void load_weight(const std::string &base_path);

  /**
   * @brief Preprocess image from a raw float tensor file (stb-style).
   *
   * The file must be a flat FP32 binary: [3, IMAGE_SIZE, IMAGE_SIZE].
   * Returns a flat FP32 vector suitable for passing to the ViT.
   *
   * @param image_tensor_path Path to the preprocessed image tensor binary.
   * @return FP32 pixel values [3 * H * W].
   */
  std::vector<float> loadImageTensor(const std::string &image_tensor_path);

  /**
   * @brief Run the full VL pipeline: image + text prompt -> generate tokens.
   *
   * Steps:
   *  1. Load + encode image with ViT (if image_tensor_path is non-empty).
   *  2. Pixel-unshuffle + connector MLP -> image_embeds [n_img_tokens, 1024].
   *  3. Tokenize prompt with LM's tokenizer (via lookupEmbedding).
   *  4. Splice image_embeds at image_token_id positions.
   *  5. Call lm_->run_with_embeddings.
   *
   * @param image_tensor_path  Path to preprocessed image tensor (FP32 binary).
   *                           Pass empty string to skip vision encoding.
   * @param prompt             Text prompt (may contain <image> placeholder).
   * @param do_sample          Sampling flag forwarded to LM.
   * @param log_output         Print generated tokens.
   */
  void run(const std::string &image_tensor_path, const std::string &prompt,
           bool do_sample = false, bool log_output = true);

  /**
   * @brief Get the generated token IDs from the last run() call.
   */
  const std::vector<unsigned int> &getGeneratedIds() const;

private:
  json cfg_;
  json generation_cfg_;
  json nntr_cfg_;

  int image_token_id_{DEFAULT_IMAGE_TOKEN_ID};
  unsigned int downsample_factor_{2};
  unsigned int projector_hidden_size_{2048};

  std::unique_ptr<Lfm2VlVisionTransformer> vit_;
  std::unique_ptr<Lfm2VlConnector> connector_;
  std::unique_ptr<Lfm2CausalLM> lm_;

  bool initialized_{false};

  /**
   * @brief Extract text_config and vision_config from the top-level config.
   *
   * LFM2-VL config.json nests the LM config under "text_config" and the
   * ViT config under "vision_config".  This helper flattens them and returns
   * two json objects that the respective sub-model constructors expect.
   */
  static std::pair<json, json> splitConfig(const json &cfg);
};

} // namespace causallm

#endif // __LFM2_VL_MODEL_H__

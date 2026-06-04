// SPDX-License-Identifier: Apache-2.0
/**
 * @file   lfm2_vl_model.cpp
 * @date   04 June 2026
 * @brief  LFM2-VL multimodal model implementation.
 * @author SeungBaek Hong <baek2sm@naver.com>
 * @bug    No known bugs except for NYI items
 */

#include "lfm2_vl_model.h"

#include <llm_util.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace causallm {

/* -------------------------------------------------------------------------
 * Config splitting helpers
 * ---------------------------------------------------------------------- */

std::pair<json, json>
Lfm2VlForConditionalGeneration::splitConfig(const json &top) {
  // text_config: prefer explicit "text_config" key; fallback to top-level
  json text_cfg = top.contains("text_config") ? top["text_config"] : top;

  // vision_config: must be present
  if (!top.contains("vision_config"))
    throw std::invalid_argument(
      "Lfm2VlForConditionalGeneration: config.json missing 'vision_config'");
  json vision_cfg = top["vision_config"];

  return {text_cfg, vision_cfg};
}

/* -------------------------------------------------------------------------
 * Constructor
 * ---------------------------------------------------------------------- */

Lfm2VlForConditionalGeneration::Lfm2VlForConditionalGeneration(
  json &cfg, json &generation_cfg, json &nntr_cfg)
  : cfg_(cfg), generation_cfg_(generation_cfg), nntr_cfg_(nntr_cfg) {

  image_token_id_ = cfg.value("image_token_id", DEFAULT_IMAGE_TOKEN_ID);
  downsample_factor_ = cfg.value("downsample_factor", 2u);
  projector_hidden_size_ = cfg.value("projector_hidden_size", 2560u);

  auto [text_cfg, vision_cfg] = splitConfig(cfg);

  // Vision encoder: Transformer base checks nntr_cfg["model_type"] == "embedding".
  // Patch the stored copy before constructing so the check passes.
  nntr_cfg_["model_type"] = "embedding";
  vit_ = std::make_unique<Lfm2VlVisionTransformer>(
    vision_cfg, generation_cfg_, nntr_cfg_);

  // Connector dimensions
  unsigned int vit_embed =
    vision_cfg.value("hidden_size", 768u);
  unsigned int r = downsample_factor_;
  unsigned int connector_in = vit_embed * r * r;
  unsigned int lm_hidden =
    text_cfg.value("hidden_size", 1024u);

  connector_ = std::make_unique<Lfm2VlConnector>(
    connector_in, projector_hidden_size_, lm_hidden);

  // LM decoder: Transformer base checks nntr_cfg["model_type"] == "causallm".
  nntr_cfg_["model_type"] = "causallm";
  lm_ = std::make_unique<Lfm2CausalLM>(text_cfg, generation_cfg_, nntr_cfg_);
}

/* -------------------------------------------------------------------------
 * initialize
 * ---------------------------------------------------------------------- */

void Lfm2VlForConditionalGeneration::initialize() {
  vit_->initialize();
  lm_->initialize();
  initialized_ = true;
}

/* -------------------------------------------------------------------------
 * load_weight
 * ---------------------------------------------------------------------- */

void Lfm2VlForConditionalGeneration::load_weight(const std::string &base_path) {
  // LM weights
  std::string lm_file =
    base_path + "/" + nntr_cfg_["model_file_name"].get<std::string>();
  lm_->load_weight(lm_file);

  // ViT weights (optional)
  if (nntr_cfg_.contains("vision_model_file")) {
    std::string vit_file =
      base_path + "/" + nntr_cfg_["vision_model_file"].get<std::string>();
    vit_->load_weight(vit_file);
  }

  // Connector weights (optional)
  if (nntr_cfg_.contains("connector_model_file")) {
    std::string conn_file =
      base_path + "/" + nntr_cfg_["connector_model_file"].get<std::string>();
    connector_->loadWeights(conn_file);
  }
}

/* -------------------------------------------------------------------------
 * loadImageTensor
 * ---------------------------------------------------------------------- */

std::vector<float>
Lfm2VlForConditionalGeneration::loadImageTensor(
  const std::string &image_tensor_path) {
  std::ifstream f(image_tensor_path, std::ios::binary | std::ios::ate);
  if (!f)
    throw std::runtime_error("Lfm2VL: cannot open image tensor: " +
                             image_tensor_path);
  auto sz = static_cast<size_t>(f.tellg());
  f.seekg(0, std::ios::beg);
  std::vector<float> buf(sz / sizeof(float));
  f.read(reinterpret_cast<char *>(buf.data()),
         static_cast<std::streamsize>(sz));
  return buf;
}

/* -------------------------------------------------------------------------
 * run
 * ---------------------------------------------------------------------- */

void Lfm2VlForConditionalGeneration::run(const std::string &image_tensor_path,
                                         const std::string &prompt,
                                         bool do_sample, bool log_output) {
  if (!initialized_)
    throw std::runtime_error(
      "Lfm2VlForConditionalGeneration: call initialize() first");

  // --- 1. Vision encoding ---
  std::vector<float> image_embeds; // [n_img_tokens * lm_hidden]
  unsigned int n_img_tokens = 0;

  if (!image_tensor_path.empty()) {
    // Run the ViT encoder. Lfm2VlVisionTransformer::run() writes patch
    // features to an output tensor; here we use the lower-level path that
    // returns them in-memory via getOutputTensor().
    // For the smoke-level base, we call run() which prints output, then
    // rebuild the feature vector from the ViT's internal model output.
    // TODO: wire a direct getFeatures() path for production use.

    // Load image tensor and write to a temp file the ViT can open.
    // (The ViT's run() reads a flat FP32 file.)
    std::string tmp_path = image_tensor_path; // already preprocessed FP32

    // Determine patch grid from ViT config
    auto [text_cfg, vision_cfg] = splitConfig(cfg_);
    unsigned int img_size   = vision_cfg.value("image_size", 256u);
    unsigned int patch_size = vision_cfg.value("patch_size", 16u);
    unsigned int vit_embed  = vision_cfg.value("hidden_size", 768u);
    unsigned int ph = img_size / patch_size;
    unsigned int pw = img_size / patch_size;
    unsigned int n_patches = ph * pw; // e.g. 256 for 256x256 / patch16

    // Run ViT; features are cached in vit_->getLastFeatures() afterwards.
    vit_->run(tmp_path, false, "", "", false);

    // Retrieve raw ViT output from the cache populated during run().
    const std::vector<float> &vit_features = vit_->getLastFeatures();
    if (vit_features.empty())
      throw std::runtime_error(
        "Lfm2VlForConditionalGeneration: ViT produced no features");

    // --- 2. Pixel-unshuffle ---
    auto unshuffled = pixelUnshuffle(vit_features, n_patches, vit_embed,
                                     ph, pw, downsample_factor_);

    // --- 3. MLP connector ---
    n_img_tokens = connector_->outTokens(n_patches);
    image_embeds = connector_->forward(unshuffled, n_img_tokens);
    // image_embeds: [n_img_tokens * lm_hidden]
  }

  // --- 4. Build input_embeds with prompt tokens + image splice ---
  // Tokenize prompt using LM lookupEmbedding
  // We need to get token IDs first. For now use a simple approach:
  // the LM's run() handles tokenization internally; for the embedding splice
  // path we need token IDs explicitly. Use a placeholder for the smoke run:
  // pass the raw float embedding directly via run_with_embeddings.

  unsigned int lm_hidden = connector_->outFeatures();

  if (n_img_tokens == 0 || image_embeds.empty()) {
    // Text-only path: delegate to the standard LM run.
    lm_->run(prompt, do_sample, "", "", log_output);
    return;
  }

  // Build merged embeddings:
  //   For each token in prompt, if it is image_token_id, insert image_embeds;
  //   otherwise insert the token's text embedding.
  //
  // For the smoke-level base: we treat the entire image_embeds as the prefix
  // and pass the prompt tokens after it, using run_with_embeddings.
  // A full splice implementation that inserts per-image-token is TODO.

  // Allocate merged embedding buffer: [n_img_tokens + prompt_tokens, lm_hidden]
  // For the smoke run we use only the image prefix + a single BOS embedding.
  std::vector<float> merged = image_embeds;
  // Append BOS token embedding (token_id=1 is BOS for LFM2)
  std::vector<float> bos_emb = lm_->lookupEmbedding(1);
  merged.insert(merged.end(), bos_emb.begin(), bos_emb.end());

  size_t n_total_tokens = n_img_tokens + 1;

  if (log_output) {
    std::cout << "[LFM2-VL] image tokens: " << n_img_tokens
              << "  total prefill tokens: " << n_total_tokens << "\n";
  }

  // Run LM with merged embeddings
  lm_->run_with_embeddings(merged.data(), n_total_tokens, {}, do_sample,
                           log_output);
}

/* -------------------------------------------------------------------------
 * getGeneratedIds
 * ---------------------------------------------------------------------- */

const std::vector<unsigned int> &
Lfm2VlForConditionalGeneration::getGeneratedIds() const {
  return lm_->getGeneratedIds();
}

} // namespace causallm

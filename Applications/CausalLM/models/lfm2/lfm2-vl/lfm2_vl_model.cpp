// SPDX-License-Identifier: Apache-2.0
/**
 * @file   lfm2_vl_model.cpp
 * @date   04 June 2026
 * @brief  LFM2-VL multimodal model implementation.
 * @author SeungBaek Hong <baek2sm@naver.com>
 * @bug    No known bugs except for NYI items
 */

#include "lfm2_vl_model.h"
#include "naflex_tile.h"

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

  if (cfg.value("do_image_splitting", false)) {
    unsigned int tile_size = cfg.value("tile_size", 512u);
    vision_cfg["naflex_tile_size"] = tile_size;
    vision_cfg["naflex_base_grid"] = 16u;
  }

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

  bool do_splitting = cfg_.value("do_image_splitting", false);
  if (do_splitting) {
    // Multi-tile NaFlex path: split the image into tiles, run ViT on each,
    // accumulate connector embeddings, then assemble and run the LM.
    auto [text_cfg2, vision_cfg2] = splitConfig(cfg_);
    unsigned int patch_size = vision_cfg2.value("patch_size", 16u);
    unsigned int vit_embed  = vision_cfg2.value("hidden_size", 768u);
    unsigned int tile_size  = cfg_.value("tile_size", 512u);

    NaFlexTilingParams tiling_params;
    tiling_params.tile_size = static_cast<int>(tile_size);
    tiling_params.max_tiles = cfg_.value("max_tiles", 10);
    tiling_params.min_tiles = cfg_.value("min_tiles", 2);
    tiling_params.use_thumbnail =
      nntr_cfg_.value("use_thumbnail", true);
    tiling_params.downsample_factor = static_cast<int>(downsample_factor_);
    tiling_params.max_image_tokens = cfg_.value("max_image_tokens", 256);
    tiling_params.encoder_patch_size = static_cast<int>(patch_size);
    tiling_params.max_pixels_tolerance =
      cfg_.value("max_pixels_tolerance", 2.0f);
    tiling_params.min_image_tokens = cfg_.value("min_image_tokens", 64);

    NaFlexTileResult tile_result =
      naflexTileImage(image_tensor_path, tiling_params);

    unsigned int ph = tile_size / patch_size;
    unsigned int pw = ph;
    unsigned int n_patches_per_tile = ph * pw;

    std::cout << "[LFM2-VL] NaFlex tiling: " << tile_result.n_tiles
              << " tile(s) (grid " << tile_result.grid_w << "x"
              << tile_result.grid_h << "), each " << tile_size << "x"
              << tile_size << " -> " << n_patches_per_tile << " patches"
              << std::endl;

    const char *vit_out_path = std::getenv("NNTR_VIT_OUT");
    std::ofstream vit_dump_file;
    if (vit_out_path)
      vit_dump_file.open(vit_out_path, std::ios::binary | std::ios::trunc);

    unsigned int n_img_tokens = 0;
    std::vector<float> image_embeds;
    for (int t = 0; t < tile_result.n_tiles; ++t) {
      auto tile_features = vit_->runOnTile(tile_result.tiles[t]);
      if (vit_dump_file.is_open())
        vit_dump_file.write(
          reinterpret_cast<const char *>(tile_features.data()),
          static_cast<std::streamsize>(tile_features.size() * sizeof(float)));
      auto tile_unshuffled =
        pixelUnshuffle(tile_features, n_patches_per_tile, vit_embed, ph, pw,
                       downsample_factor_);

      unsigned int tile_out_tokens = connector_->outTokens(n_patches_per_tile);
      auto tile_embeds =
        connector_->forward(tile_unshuffled, tile_out_tokens);

      image_embeds.insert(image_embeds.end(), tile_embeds.begin(),
                          tile_embeds.end());
      n_img_tokens += tile_out_tokens;
    }
    if (vit_dump_file.is_open()) {
      vit_dump_file.close();
      std::cout << "[NNTR_VIT_OUT] wrote " << tile_result.n_tiles
                << " tile(s) of raw ViT features to " << vit_out_path
                << std::endl;
    }

    runLMWithImageEmbeds(image_embeds, n_img_tokens, prompt, do_sample,
                         log_output);
    return;
  }

  // Single-tile path: ViT reads an image file (stb decode + resize + normalize)
  // and caches patch features; generateFromViTFeatures handles the rest.
  vit_->run(image_tensor_path, false, "", "", true);
  generateFromViTFeatures(prompt, do_sample, log_output);
}

void Lfm2VlForConditionalGeneration::runFromPixels(const float *chw_pixels,
                                                   size_t n_elems,
                                                   const std::string &prompt,
                                                   bool do_sample,
                                                   bool log_output) {
  if (!initialized_)
    throw std::runtime_error(
      "Lfm2VlForConditionalGeneration: call initialize() first");
  // Host already preprocessed the image to NCHW FP32 (3*IMAGE*IMAGE); feed it
  // straight into the ViT (skips the stb decode path). Features cached.
  vit_->runFromPixels(chw_pixels, n_elems, true);
  generateFromViTFeatures(prompt, do_sample, log_output);
}

void Lfm2VlForConditionalGeneration::generateFromViTFeatures(
  const std::string &prompt, bool do_sample, bool log_output) {
  // --- 1. Vision features (cached by the ViT run in the caller) ---
  const std::vector<float> &vit_features = vit_->getLastFeatures();
  if (vit_features.empty())
    throw std::runtime_error(
      "Lfm2VlForConditionalGeneration: ViT produced no features");

  // Determine patch grid from ViT config.
  auto [text_cfg, vision_cfg] = splitConfig(cfg_);
  unsigned int img_size   = vision_cfg.value("image_size", 256u);
  unsigned int patch_size = vision_cfg.value("patch_size", 16u);
  unsigned int vit_embed  = vision_cfg.value("hidden_size", 768u);
  unsigned int ph = img_size / patch_size;
  unsigned int pw = img_size / patch_size;
  unsigned int n_patches = ph * pw; // e.g. 256 for 256x256 / patch16

  // --- 2. Pixel-unshuffle ---
  auto unshuffled =
    pixelUnshuffle(vit_features, n_patches, vit_embed, ph, pw,
                   downsample_factor_);

  // --- 3. MLP connector ---
  unsigned int n_img_tokens = connector_->outTokens(n_patches);
  std::vector<float> image_embeds = // [n_img_tokens * lm_hidden]
    connector_->forward(unshuffled, n_img_tokens);

  runLMWithImageEmbeds(image_embeds, n_img_tokens, prompt, do_sample,
                       log_output);
}

void Lfm2VlForConditionalGeneration::runLMWithImageEmbeds(
  const std::vector<float> &image_embeds, unsigned int n_img_tokens,
  const std::string &prompt, bool do_sample, bool log_output) {
  // --- 4. Build full chat-template input_embeds ---
  // HF chat template (from hf_model/chat_template.jinja):
  //   {{bos_token}}<|im_start|>user\n<image>PROMPT<|im_end|>\n<|im_start|>assistant\n
  // BOS is added via embedding lookup; the rest is tokenized.
  // The <image> token (id=396) in the tokenized sequence is replaced by
  // the n_img_tokens connector feature vectors.

  unsigned int lm_hidden = connector_->outFeatures();

  if (n_img_tokens == 0 || image_embeds.empty()) {
    // Text-only path: delegate to standard LM run.
    lm_->run(prompt, do_sample, "", "", log_output);
    return;
  }

  // Build the templated text string (BOS not included -- added separately).
  // The <image> placeholder matches image_token_id_ (396) in tokenizer.
  std::string templated_text =
    "<|im_start|>user\n<|image_start|><image><|image_end|>" + prompt +
    "<|im_end|>\n<|im_start|>assistant\n";

  // Tokenize the templated text (without BOS).
  std::vector<int64_t> token_ids = lm_->tokenize(templated_text);

  // Count <image> placeholders for diagnostics.
  size_t n_image_placeholders = 0;
  for (auto tid : token_ids)
    if (static_cast<int>(tid) == image_token_id_)
      ++n_image_placeholders;

  if (log_output) {
    std::cout << "[LFM2-VL] chat-template token count (excl BOS): "
              << token_ids.size()
              << "  <image> placeholders: " << n_image_placeholders
              << "  connector image tokens per placeholder: " << n_img_tokens
              << "\n";
  }

  // Allocate merged embedding buffer.
  // Total tokens = 1 (BOS) + (token_ids.size() - n_image_placeholders) +
  //                n_img_tokens * n_image_placeholders
  size_t n_text_tokens = token_ids.size() - n_image_placeholders;
  size_t n_total_tokens =
    1 + n_text_tokens + n_img_tokens * n_image_placeholders;
  std::vector<float> merged;
  merged.reserve(n_total_tokens * lm_hidden);

  // 1) BOS embedding (token_id = 1 for LFM2).
  {
    auto bos_emb = lm_->lookupEmbedding(1);
    merged.insert(merged.end(), bos_emb.begin(), bos_emb.end());
  }

  // 2) Process token_ids: splice image features at <image> positions.
  size_t image_chunks_used = 0;
  size_t tok_pos = 1; // 0=BOS
  for (auto tid : token_ids) {
    if (static_cast<int>(tid) == image_token_id_) {
      // Replace this placeholder with n_img_tokens image feature vectors.
      size_t offset = image_chunks_used * n_img_tokens * lm_hidden;
      merged.insert(merged.end(),
                    image_embeds.begin() + static_cast<ptrdiff_t>(offset),
                    image_embeds.begin() + static_cast<ptrdiff_t>(offset) +
                      static_cast<ptrdiff_t>(n_img_tokens * lm_hidden));
      tok_pos += n_img_tokens;
      ++image_chunks_used;
    } else {
      auto tok_emb = lm_->lookupEmbedding(static_cast<unsigned int>(tid));
      merged.insert(merged.end(), tok_emb.begin(), tok_emb.end());
      ++tok_pos;
    }
  }

  // Sanity check.
  if (merged.size() != n_total_tokens * lm_hidden) {
    throw std::runtime_error(
      "[LFM2-VL] merged embedding size mismatch: got " +
      std::to_string(merged.size()) + " expected " +
      std::to_string(n_total_tokens * lm_hidden));
  }

  if (log_output) {
    std::cout << "[LFM2-VL] total prefill tokens: " << n_total_tokens << "\n";
  }

  // Run LM with merged embeddings.
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

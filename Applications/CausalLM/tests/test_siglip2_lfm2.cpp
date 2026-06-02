// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   test_siglip2_lfm2.cpp
 * @date   2 June 2026
 * @see    https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  End-to-end test: Image -> SigLIP2 ViT -> Projector -> LFM2 -> Text
 *
 * Pipeline:
 *   1. Load preprocessed [1,3,256,256] fp32 image .bin (SigLIP2 processor output)
 *   2. Lfm2VlVisionTransformer: image -> [NUM_PATCHES,768] features
 *   3. Lfm2VlProjector: [NUM_PATCHES,768] -> [OUTPUT_TOKENS,1024]
 *   4. Merge with text token embeddings (single image placeholder = OUTPUT_TOKENS tokens)
 *   5. Lfm2CausalLM::run_with_embeddings()
 *
 * Usage:
 *   ./test_siglip2_lfm2 <image_bin_path> <model_dir> [prompt] [do_sample]
 *
 *   <image_bin_path>  Path to preprocessed fp32 image .bin (1*3*256*256 floats)
 *   <model_dir>       Path to model resource directory with:
 *                       siglip_config.json, siglip_nntr_config.json,
 *                       projector_config.json, projector_nntr_config.json,
 *                       config.json, nntr_config.json, generation_config.json,
 *                       weight files, tokenizer.json
 *   [prompt]          Text prompt (default: "Describe this image briefly.")
 *   [do_sample]       0=greedy, 1=sampling (default: 0)
 */

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "json.hpp"
#include <app_context.h>
#include <engine.h>
#include <factory.h>
#include <model.h>
#include <tokenizers_cpp.h>

#include "lfm2_causallm.h"
#include "lfm2-vl/vision/lfm2_vl_vision_transformer.h"
#include "lfm2/lfm2-vl/lfm2_vl_projector.h"

using json = nlohmann::json;

static void printSection(const std::string &title) {
  std::cout << "\n[======================================================]\n"
            << "  " << title
            << "\n[======================================================]\n\n";
}

static std::string LoadBytesFromFile(const std::string &path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + path);
  }
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::string buffer(size, ' ');
  if (!file.read(&buffer[0], size)) {
    throw std::runtime_error("Failed to read file: " + path);
  }
  return buffer;
}

/**
 * @brief Build chat template text segments with a single image placeholder.
 *
 * Returns two segments: [0] system+user header, [1] prompt+end tokens.
 * The single image placeholder is conceptually between segment[0] and
 * segment[1].
 */
static std::vector<std::string>
apply_chat_template_image(const std::string &prompt) {
  std::vector<std::string> segments;
  segments.push_back("<|startoftext|><|im_start|>system\nYou are a helpful "
                     "assistant.<|im_end|>\n<|im_start|>user\n");
  segments.push_back(prompt + "<|im_end|>\n<|im_start|>assistant\n");
  return segments;
}

/**
 * @brief Merge text token embeddings and vision embeddings into inputs_embeds.
 */
static std::pair<std::vector<float>, unsigned int>
merge_text_image_embeddings(
  const std::vector<std::string> &text_segments,
  const std::unique_ptr<tokenizers::Tokenizer> &tokenizer,
  const std::unique_ptr<causallm::Lfm2CausalLM> &lfm2,
  const float *vision_embeds, unsigned int num_image_tags,
  unsigned int vision_tokens_per_image, unsigned int text_dim,
  unsigned int init_seq_len, unsigned int batch_size) {

  std::vector<float> inputs_embeds(
    static_cast<size_t>(batch_size) * init_seq_len * text_dim, 0.0f);

  size_t embed_offset = 0;
  size_t image_idx = 0;

  for (size_t seg_i = 0; seg_i < text_segments.size(); ++seg_i) {
    if (!text_segments[seg_i].empty()) {
      auto enc =
        tokenizer->Encode(text_segments[seg_i], /*add_special_token=*/false);
      for (auto id : enc) {
        std::vector<float> embed =
          lfm2->lookupEmbedding(static_cast<unsigned int>(id));
        std::copy(embed.begin(), embed.end(),
                  inputs_embeds.data() + embed_offset);
        embed_offset += text_dim;
      }
    }

    if (seg_i < num_image_tags) {
      size_t vision_start = image_idx * vision_tokens_per_image;
      for (size_t v = 0; v < vision_tokens_per_image; ++v) {
        std::copy(vision_embeds + (vision_start + v) * text_dim,
                  vision_embeds + (vision_start + v + 1) * text_dim,
                  inputs_embeds.data() + embed_offset);
        embed_offset += text_dim;
      }
      image_idx++;
    }
  }

  unsigned int actual_total_tokens =
    static_cast<unsigned int>(embed_offset / text_dim);
  return {std::move(inputs_embeds), actual_total_tokens};
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " <image_bin_path> <model_dir> [prompt] [do_sample]\n\n"
              << "  <image_bin_path>  Preprocessed fp32 image .bin "
                 "[1,3,256,256]\n"
              << "  <model_dir>       Model resource directory\n"
              << "  [prompt]          Text prompt (default: 'Describe this "
                 "image briefly.')\n"
              << "  [do_sample]       0/1 (default: 0)\n";
    return EXIT_FAILURE;
  }

  const std::string image_bin_path = argv[1];
  const std::string model_dir = argv[2];
  const std::string prompt =
    (argc >= 4) ? argv[3] : "Describe this image briefly.";
  const bool do_sample = (argc >= 5) ? (std::string(argv[4]) == "1") : false;

  try {
    // Step 1: Load configuration files
    printSection("Step 1: Load Configuration");

    json siglip_cfg =
      causallm::LoadJsonFile(model_dir + "/siglip_config.json");
    json siglip_nntr_cfg =
      causallm::LoadJsonFile(model_dir + "/siglip_nntr_config.json");

    json proj_cfg =
      causallm::LoadJsonFile(model_dir + "/projector_config.json");
    json proj_nntr_cfg =
      causallm::LoadJsonFile(model_dir + "/projector_nntr_config.json");

    json lfm2_cfg = causallm::LoadJsonFile(model_dir + "/config.json");
    json lfm2_nntr_cfg =
      causallm::LoadJsonFile(model_dir + "/nntr_config.json");
    json lfm2_generation_cfg =
      causallm::LoadJsonFile(model_dir + "/generation_config.json");

    lfm2_nntr_cfg["use_embedding"] = true;

    const std::string siglip_weight =
      model_dir + "/" +
      siglip_nntr_cfg["model_file_name"].get<std::string>();
    const std::string proj_weight =
      model_dir + "/" + proj_nntr_cfg["model_file_name"].get<std::string>();
    const std::string lfm2_weight =
      model_dir + "/" + lfm2_nntr_cfg["model_file_name"].get<std::string>();

    std::cout << "  image_bin_path : " << image_bin_path << "\n";
    std::cout << "  model_dir      : " << model_dir << "\n";
    std::cout << "  prompt         : " << prompt << "\n";

    // Step 2: Load preprocessed image
    printSection("Step 2: Load Preprocessed Image");

    const unsigned int IMAGE_SIZE = siglip_cfg.value("image_size", 256u);
    const unsigned int NUM_CHANNELS = siglip_cfg.value("num_channels", 3u);
    const size_t n_elems =
      static_cast<size_t>(1) * NUM_CHANNELS * IMAGE_SIZE * IMAGE_SIZE;

    std::vector<float> image_data(n_elems);
    {
      std::ifstream fin(image_bin_path, std::ios::binary);
      if (!fin) {
        throw std::runtime_error("Failed to open image .bin: " + image_bin_path);
      }
      fin.read(reinterpret_cast<char *>(image_data.data()),
               static_cast<std::streamsize>(n_elems * sizeof(float)));
      if (static_cast<size_t>(fin.gcount()) !=
          n_elems * sizeof(float)) {
        throw std::runtime_error(
          "Image .bin is smaller than expected (" +
          std::to_string(n_elems) + " fp32 elements).");
      }
    }
    std::cout << "  Loaded image: [1, " << NUM_CHANNELS << ", " << IMAGE_SIZE
              << ", " << IMAGE_SIZE << "]\n";

    // Step 3: SigLIP2 ViT Encoder
    printSection("Step 3: SigLIP2 ViT Encoder");

    const unsigned int PATCH_SIZE = siglip_cfg.value("patch_size", 16u);
    const unsigned int NUM_PATCHES =
      (IMAGE_SIZE / PATCH_SIZE) * (IMAGE_SIZE / PATCH_SIZE);
    const unsigned int VIT_DIM = siglip_cfg.value("hidden_size", 768u);

    json empty_gen_cfg = json::object();
    auto siglip = std::make_unique<causallm::Lfm2VlVisionTransformer>(
      siglip_cfg, empty_gen_cfg, siglip_nntr_cfg);
    siglip->initialize();
    std::cout << "  SigLIP2 ViT graph constructed.\n";
    siglip->load_weight(siglip_weight);
    std::cout << "  SigLIP2 ViT weights loaded.\n";

    auto t_vit_start = std::chrono::high_resolution_clock::now();
    auto [vision_ptr, vision_bytes] = siglip->runBuffer(image_data.data());
    auto t_vit_end = std::chrono::high_resolution_clock::now();
    const long long vit_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(t_vit_end -
                                                             t_vit_start)
        .count();

    std::cout << "  Vision features: [" << NUM_PATCHES << ", " << VIT_DIM
              << "]\n";
    std::cout << "  SigLIP2 time: " << vit_ms << " ms\n";

    // Step 4: Projector (768 -> 1024)
    printSection("Step 4: Projector (768 -> 1024)");

    auto projector = std::make_unique<causallm::Lfm2VlProjector>(
      proj_cfg, empty_gen_cfg, proj_nntr_cfg);
    projector->initialize();
    std::cout << "  Projector graph constructed.\n";
    projector->load_weight(proj_weight);
    std::cout << "  Projector weights loaded.\n";

    auto t_proj_start = std::chrono::high_resolution_clock::now();
    auto [proj_ptr, proj_size] =
      projector->run(vision_ptr, NUM_PATCHES, true);
    auto t_proj_end = std::chrono::high_resolution_clock::now();
    const long long proj_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(t_proj_end -
                                                             t_proj_start)
        .count();

    const unsigned int DOWNSAMPLE_FACTOR =
      proj_cfg.value("downsample_factor", 2u);
    const unsigned int OUTPUT_TOKENS =
      NUM_PATCHES / (DOWNSAMPLE_FACTOR * DOWNSAMPLE_FACTOR);
    const unsigned int TEXT_DIM = lfm2_cfg.value("hidden_size", 1024u);

    std::cout << "  Projected tokens: " << OUTPUT_TOKENS << " x " << TEXT_DIM
              << "\n";
    std::cout << "  Projector time: " << proj_ms << " ms\n";

    // Step 5: LFM2 CausalLM
    printSection("Step 5: LFM2 CausalLM (use_embedding=true)");

    auto lfm2 = std::make_unique<causallm::Lfm2CausalLM>(
      lfm2_cfg, lfm2_generation_cfg, lfm2_nntr_cfg);
    lfm2->initialize();
    std::cout << "  LFM2 graph constructed.\n";
    lfm2->load_weight(lfm2_weight);
    std::cout << "  LFM2 weights loaded.\n";

    // Step 6: Tokenize prompt and build merged embeddings
    printSection("Step 6: Build Merged Embeddings");

    std::string tokenizer_path =
      lfm2_nntr_cfg["tokenizer_file"].get<std::string>();
    if (!std::filesystem::exists(tokenizer_path)) {
      std::string resolved = model_dir + "/tokenizer.json";
      if (std::filesystem::exists(resolved)) {
        tokenizer_path = resolved;
      } else {
        throw std::runtime_error("Tokenizer file not found: " + tokenizer_path);
      }
    }

    auto tokenizer =
      tokenizers::Tokenizer::FromBlobJSON(LoadBytesFromFile(tokenizer_path));

    const unsigned int num_image_tags = 1;
    auto text_segments = apply_chat_template_image(prompt);

    std::cout << "  Text segments: " << text_segments.size() << "\n";
    for (size_t i = 0; i < text_segments.size(); ++i) {
      std::cout << "    seg[" << i << "]: " << text_segments[i] << "\n";
    }

    const unsigned int init_seq_len =
      lfm2_nntr_cfg["init_seq_len"].get<unsigned int>();
    const unsigned int batch_size =
      lfm2_nntr_cfg["batch_size"].get<unsigned int>();
    const float *proj_data = static_cast<const float *>(proj_ptr);

    auto [inputs_embeds, actual_total_tokens] = merge_text_image_embeddings(
      text_segments, tokenizer, lfm2, proj_data, num_image_tags,
      OUTPUT_TOKENS, TEXT_DIM, init_seq_len, batch_size);

    std::vector<int> seed_tokens;
    for (size_t seg_i = 0; seg_i < text_segments.size(); ++seg_i) {
      if (!text_segments[seg_i].empty()) {
        auto enc = tokenizer->Encode(text_segments[seg_i],
                                     /*add_special_token=*/false);
        for (auto id : enc)
          seed_tokens.push_back(id);
      }
      if (seg_i < num_image_tags)
        seed_tokens.insert(seed_tokens.end(), OUTPUT_TOKENS, 0);
    }

    std::cout << "  Output vision tokens: " << OUTPUT_TOKENS << "\n";
    std::cout << "  Actual total tokens: " << actual_total_tokens << "\n";

    // Step 7: Run LFM2 inference
    printSection("Step 7: LFM2 Inference (run_with_embeddings)");

    auto t_lfm2_start = std::chrono::high_resolution_clock::now();
    lfm2->run_with_embeddings(inputs_embeds.data(), actual_total_tokens,
                               seed_tokens, do_sample, /*log_output=*/true);
    auto t_lfm2_end = std::chrono::high_resolution_clock::now();
    const long long lfm2_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(t_lfm2_end -
                                                             t_lfm2_start)
        .count();

    // Step 8: Results
    printSection("Step 8: Results");

    const auto &generated_ids = lfm2->getGeneratedIds();
    std::cout << "  LFM2 inference time: " << lfm2_ms << " ms\n";
    std::cout << "  Generated tokens: " << generated_ids.size() << "\n";

    if (!generated_ids.empty()) {
      std::vector<int32_t> gen_ids_i32(generated_ids.begin(),
                                        generated_ids.end());
      std::string decoded = tokenizer->Decode(gen_ids_i32);
      std::cout << "  Decoded output: " << decoded << "\n";
    }

    printSection("Summary");
    std::cout << "  SigLIP2 encoder : " << vit_ms << " ms\n";
    std::cout << "  Projector       : " << proj_ms << " ms\n";
    std::cout << "  LFM2 generation : " << lfm2_ms << " ms\n";
    std::cout << "  Total           : " << (vit_ms + proj_ms + lfm2_ms)
              << " ms\n";
    std::cout << "\n[Test completed successfully]\n\n";

  } catch (const std::exception &e) {
    std::cerr << "\n[FATAL] " << e.what() << "\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

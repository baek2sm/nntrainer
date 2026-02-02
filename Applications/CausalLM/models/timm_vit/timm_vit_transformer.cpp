// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   timm_vit_transformer.cpp
 * @date   28 Jan 2026
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 */
#include "timm_vit_transformer.h"
#include <factory.h>
#include <llm_util.hpp>

namespace causallm {

void TimmViTTransformer::setupParameters(json &cfg, json &generation_cfg,
                                         json &nntr_cfg) {
  BATCH_SIZE = nntr_cfg.value("batch_size", 1);
  MODEL_TENSOR_TYPE = nntr_cfg.value("model_tensor_type", "FP32-FP32");
  EMBEDDING_DTYPE = nntr_cfg.value("embedding_dtype", "FP32");
  FC_LAYER_DTYPE = nntr_cfg.value("fc_layer_dtype", "FP32");

  NUM_VOCAB = cfg.value("vocab_size", 1000);
  DIM = cfg.value("hidden_size", 768);
  INTERMEDIATE_SIZE = cfg.value("intermediate_size", 3072);
  NUM_LAYERS = cfg.value("num_hidden_layers", 12);
  NUM_HEADS = cfg.value("num_attention_heads", 12);
  HEAD_DIM = cfg.value("head_dim", DIM / NUM_HEADS);
  NUM_KEY_VALUE_HEADS = cfg.value("num_key_value_heads", NUM_HEADS);
  MAX_POSITION_EMBEDDINGS = cfg.value("max_position_embeddings", 196);
  ROPE_THETA = cfg.value("rope_theta", 10000);
  TIE_WORD_EMBEDDINGS = cfg.value("tie_word_embeddings", false);
  NORM_EPS = cfg.value("rms_norm_eps", 1e-6);
  GQA_SIZE = NUM_HEADS / NUM_KEY_VALUE_HEADS;

  IS_CAUSAL = cfg.value("is_causal", false);
  SLIDING_WINDOW =
    cfg.contains("sliding_window") && !cfg["sliding_window"].is_null()
      ? cfg["sliding_window"].get<unsigned int>()
      : UINT_MAX;

  INIT_SEQ_LEN = nntr_cfg.value("init_seq_len", 224);
  MAX_SEQ_LEN = nntr_cfg.value("max_seq_len", 224);
  NUM_TO_GENERATE = nntr_cfg.value("num_to_generate", 0);
}

std::vector<LayerHandle> TimmViTTransformer::createPatchEmbed() {
  std::vector<LayerHandle> layers;

  int img_size = 224;
  int patch_size = 16;
  int embed_dim = DIM;

  layers.push_back(createLayer(
    "input", {withKey("name", "input_image"),
              withKey("input_shape", "3:" + std::to_string(img_size) + ":" +
                                       std::to_string(img_size))}));

  std::vector<std::string> conv_params = {
    withKey("name", "patch_embed/conv"),
    withKey("kernel_size",
            {std::to_string(patch_size), std::to_string(patch_size)}),
    withKey("filters", std::to_string(embed_dim)),
    withKey("stride", {std::to_string(patch_size), std::to_string(patch_size)}),
    withKey("padding", "valid"),
    withKey("input_layers", "input_image")};
  layers.push_back(createLayer("conv2d", conv_params));

  int grid_h = img_size / patch_size;
  int grid_w = img_size / patch_size;
  int patch_count = grid_h * grid_w;

  layers.push_back(createLayer(
    "reshape", {withKey("name", "patch_embed/flatten"),
                withKey("target_shape", "1:" + std::to_string(embed_dim) + ":" +
                                          std::to_string(patch_count)),
                withKey("input_layers", "patch_embed/conv")}));

  layers.push_back(
    createLayer("permute", {withKey("name", "patch_embed/transpose"),
                            withKey("direction", {1, 3, 2}),
                            withKey("input_layers", "patch_embed/flatten")}));

  layers.push_back(createLayer(
    "weight",
    {withKey("name", "pos_embed/weights"),
     withKey("weight_dim", "1:1:" + std::to_string(patch_count) + ":" +
                             std::to_string(embed_dim)),
     withKey("tensor_dtype", "FP32"), withKey("weight_name", "pos_embed")}));

  layers.push_back(createLayer(
    "addition",
    {withKey("name", "pos_embed/add"),
     withKey("input_layers", {"patch_embed/transpose", "pos_embed/weights"})}));

  return layers;
}

void TimmViTTransformer::constructModel() {
  std::vector<LayerHandle> layers;

  model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);

  auto patch_embed_layers = createPatchEmbed();
  layers.insert(layers.end(), patch_embed_layers.begin(),
                patch_embed_layers.end());

  for (auto &layer : layers) {
    model->addLayer(layer);
  }
}

void TimmViTTransformer::initialize() {
  registerCustomLayers();

  constructModel();

  std::vector<std::string> model_props = {
    withKey("batch_size", BATCH_SIZE), withKey("epochs", "1"),
    withKey("model_tensor_type", MODEL_TENSOR_TYPE)};

  model->setProperty(model_props);

  if (model->compile(ml::train::ExecutionMode::INFERENCE)) {
    throw std::invalid_argument("Model compilation failed.");
  }

  if (model->initialize(ml::train::ExecutionMode::INFERENCE)) {
    throw std::invalid_argument("Model initialization failed.");
  }

  is_initialized = true;

  model->summarize(std::cout, ML_TRAIN_SUMMARY_MODEL);
}

void TimmViTTransformer::registerCustomLayers() {
  Transformer::registerCustomLayers();
}

std::vector<LayerHandle>
TimmViTTransformer::createTransformerDecoderBlock(const int layer_id,
                                                  std::string input_name) {
  return {};
}

void TimmViTTransformer::run(const WSTR prompt, bool do_sample,
                             const WSTR system_prompt, const WSTR tail_prompt) {
  if (!is_initialized) {
    throw std::runtime_error("TimmViT model is not initialized. Please call "
                             "initialize() before run().");
  }

  unsigned int img_c = 3, img_h = 224, img_w = 224;
  unsigned int patch_count = (img_h / 16) * (img_w / 16);

  unsigned int input_size = BATCH_SIZE * img_c * img_h * img_w;
  float *input_sample = (float *)malloc(sizeof(float) * input_size);

  std::string image_path_str(prompt);
  std::vector<float> image_data = loadAndPreprocessImage(
    image_path_str, img_w, img_h, true); // normalize=true for [0,255] -> [0,1]

  for (size_t i = 0; i < input_size; ++i) {
    input_sample[i] = image_data[i];
  }

  std::vector<float *> input;
  input.push_back(input_sample);
  std::vector<float *> label;

  std::vector<float *> output = model->incremental_inference(
    BATCH_SIZE, input, label, patch_count, 0, patch_count, false);

  std::cout << "Output" << std::endl;
  std::cout << "First 10 values: ";
  for (int i = 0; i < std::min(10, DIM); ++i) {
    std::cout << "[" << i << "]=" << output[0][i] << " ";
  }
  std::cout << std::endl;

  free(input_sample);
}

} // namespace causallm

// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   mt5_causallm.cpp
 * @date   18 Mar 2026
 * @brief  mT5 encoder-decoder model implementation for nntrainer CausalLM.
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * @note   This is the first model in nntrainer CausalLM that uses
 *         cross-attention (encoder-decoder architecture).
 *
 *         Architecture overview:
 *         [Encoder]
 *           source_tokens -> embedding -> N encoder blocks -> encoder_norm
 *           Each encoder block: RMSNorm -> SelfAttn -> Add -> RMSNorm -> FFN ->
 *           Add
 *
 *         [KV Projection]
 *           encoder_output -> per-layer K/V projections (FC layers)
 *
 *         [Decoder]
 *           target_tokens -> embedding -> N decoder blocks -> decoder_norm ->
 *           LMHead
 *           Each decoder block:
 *             RMSNorm -> SelfAttn(causal) -> Add
 *             -> RMSNorm -> CrossAttn(Q from decoder, K/V pre-computed) -> Add
 *             -> RMSNorm -> FFN(GeGLU) -> Add
 *
 * @note   Position encoding: mT5 uses T5-style relative position bias.
 *         This implementation sets use_rope=false in mha_core.
 *         Relative position bias is NOT yet implemented (TODO).
 *         For full model accuracy, relative position bias must be added
 *         to the attention scores before softmax.
 */

#include <mt5_causallm.h>

#include <app_context.h>
#include <engine.h>
#include <llm_util.hpp>

#include <lm_head.h>
#include <mha_core.h>
#include <rms_norm.h>
#include <swiglu.h>

namespace causallm {

// ============================================================
// Config sanitization: map T5/mT5 config keys to framework keys
// ============================================================
json &MT5Transformer::sanitizeConfig(json &cfg) {
  // Map T5 config keys to standard transformer keys
  if (cfg.contains("d_model") && !cfg.contains("hidden_size")) {
    cfg["hidden_size"] = cfg["d_model"];
  }
  if (cfg.contains("d_ff") && !cfg.contains("intermediate_size")) {
    cfg["intermediate_size"] = cfg["d_ff"];
  }
  if (cfg.contains("num_heads") && !cfg.contains("num_attention_heads")) {
    cfg["num_attention_heads"] = cfg["num_heads"];
  }
  if (cfg.contains("num_layers") && !cfg.contains("num_hidden_layers")) {
    cfg["num_hidden_layers"] = cfg["num_layers"];
  }
  // T5 has no GQA - num_key_value_heads equals num_attention_heads
  if (!cfg.contains("num_key_value_heads")) {
    cfg["num_key_value_heads"] = cfg["num_attention_heads"];
  }
  // Head dim from d_kv
  if (cfg.contains("d_kv") && !cfg.contains("head_dim")) {
    cfg["head_dim"] = cfg["d_kv"];
  }
  // T5 defaults
  if (!cfg.contains("tie_word_embeddings")) {
    cfg["tie_word_embeddings"] = false;
  }
  if (!cfg.contains("max_position_embeddings")) {
    cfg["max_position_embeddings"] = 512;
  }
  if (!cfg.contains("rope_theta")) {
    cfg["rope_theta"] = 10000; // unused but required by base class
  }
  if (!cfg.contains("rms_norm_eps")) {
    cfg["rms_norm_eps"] = 1e-6; // T5 default
  }
  // T5 encoder is bidirectional
  if (!cfg.contains("is_causal")) {
    cfg["is_causal"] = true; // decoder is causal
  }

  return cfg;
}

void MT5Transformer::setupParameters(json &cfg, json &generation_cfg,
                                     json &nntr_cfg) {
  Transformer::setupParameters(cfg, generation_cfg, nntr_cfg);

  // T5-specific parameters
  NUM_ENCODER_LAYERS =
    cfg.contains("num_layers") ? cfg["num_layers"].get<int>() : NUM_LAYERS;
  NUM_DECODER_LAYERS = cfg.contains("num_decoder_layers")
                         ? cfg["num_decoder_layers"].get<int>()
                         : NUM_LAYERS;
  D_KV = cfg.contains("d_kv") ? cfg["d_kv"].get<int>() : HEAD_DIM;
}

// ============================================================
// Encoder model construction
// ============================================================
void MT5Transformer::constructEncoder() {
  encoder_model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);

  std::vector<LayerHandle> layers;

  // Encoder input
  layers.push_back(createLayer(
    "input", {withKey("name", "encoder_input0"),
              withKey("input_shape", "1:1:" + std::to_string(INIT_SEQ_LEN))}));

  // Encoder embedding (shared with decoder)
  const std::string embedding_type = "embedding_layer";
  layers.push_back(createLayer(
    embedding_type,
    {"name=encoder_embedding0", "in_dim=" + std::to_string(NUM_VOCAB),
     "weight_dtype=" + EMBEDDING_DTYPE, "out_dim=" + std::to_string(DIM)}));

  // Encoder blocks
  for (int i = 0; i < NUM_ENCODER_LAYERS; ++i) {
    std::string input_name =
      (i == 0) ? "encoder_embedding0"
               : "encoder_layer" + std::to_string(i - 1) + "_block_output";
    auto block = createEncoderBlock(i, input_name);
    layers.insert(layers.end(), block.begin(), block.end());
  }

  // Encoder final norm
  layers.push_back(createLayer(
    "rms_norm",
    {withKey("name", "encoder_output_norm"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("input_layers",
             "encoder_layer" + std::to_string(NUM_ENCODER_LAYERS - 1) +
               "_block_output"),
     withKey("packed", "false")}));

  for (auto &layer : layers) {
    encoder_model->addLayer(layer);
  }
}

std::vector<LayerHandle>
MT5Transformer::createEncoderBlock(const int layer_id,
                                   std::string input_name) {
  std::vector<LayerHandle> layers;
  std::string prefix = "encoder_layer" + std::to_string(layer_id);

  // Pre-norm for self-attention
  layers.push_back(createLayer(
    "rms_norm", {withKey("name", prefix + "_attention_norm"),
                 withKey("input_layers", input_name),
                 withKey("epsilon", std::to_string(NORM_EPS)),
                 withKey("packed", "false")}));

  // Self-attention (bidirectional, no RoPE)
  std::string norm_out = prefix + "_attention_norm";

  // Q projection
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", prefix + "_wq"),
     withKey("unit", NUM_HEADS * D_KV),
     withKey("disable_bias", "true"),
     withKey("input_layers", norm_out),
     withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));

  // K projection
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", prefix + "_wk"),
     withKey("unit", NUM_HEADS * D_KV),
     withKey("disable_bias", "true"),
     withKey("input_layers", norm_out),
     withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));

  // V projection
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", prefix + "_wv"),
     withKey("unit", NUM_HEADS * D_KV),
     withKey("disable_bias", "true"),
     withKey("input_layers", norm_out),
     withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));

  // MHA core (bidirectional self-attention, no RoPE)
  layers.push_back(createLayer(
    "mha_core",
    {withKey("name", prefix + "_attention"),
     withKey("num_heads", NUM_HEADS),
     withKey("num_heads_kv", NUM_HEADS), // no GQA in T5
     withKey("max_timestep",
             std::to_string(INIT_SEQ_LEN + NUM_TO_GENERATE)),
     withKey("is_causal", "false"), // encoder is bidirectional
     withKey("use_rope", "false"),  // T5 uses relative position bias
     withKey("input_layers",
             prefix + "_wq," + prefix + "_wk," + prefix + "_wv")}));

  // O projection
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", prefix + "_attention_out"),
     withKey("unit", DIM),
     withKey("disable_bias", "true"),
     withKey("input_layers", prefix + "_attention"),
     withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));

  // Residual connection (self-attention)
  layers.push_back(createLayer(
    "addition",
    {withKey("name", prefix + "_self_attn_add"),
     withKey("input_layers",
             input_name + "," + prefix + "_attention_out")}));

  // Pre-norm for FFN
  layers.push_back(createLayer(
    "rms_norm",
    {withKey("name", prefix + "_ffn_norm"),
     withKey("input_layers", prefix + "_self_attn_add"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));

  // FFN (GeGLU)
  auto ffn = createMlp(layer_id, DIM, INTERMEDIATE_SIZE,
                        prefix + "_ffn_norm");
  // Rename FFN layers to have encoder prefix
  // (createMlp uses "layer" + layer_id, we need "encoder_layer" + layer_id)
  // Actually, we need to create encoder-specific FFN
  // Let's inline the FFN creation with proper naming

  // Gate projection + GeLU
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", prefix + "_ffn_gate"),
     withKey("unit", INTERMEDIATE_SIZE),
     withKey("disable_bias", "true"),
     withKey("input_layers", prefix + "_ffn_norm"),
     withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));

  layers.push_back(createLayer(
    "activation",
    {withKey("name", prefix + "_ffn_gate_gelu"),
     withKey("activation", "tanh_gelu"),
     withKey("input_layers", prefix + "_ffn_gate")}));

  // Up projection
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", prefix + "_ffn_up"),
     withKey("unit", INTERMEDIATE_SIZE),
     withKey("disable_bias", "true"),
     withKey("input_layers", prefix + "_ffn_norm"),
     withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));

  // Multiply (gate * up)
  layers.push_back(createLayer(
    "multiply",
    {withKey("name", prefix + "_ffn_geglu"),
     withKey("input_layers",
             prefix + "_ffn_gate_gelu," + prefix + "_ffn_up")}));

  // Down projection
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", prefix + "_ffn_down"),
     withKey("unit", DIM),
     withKey("disable_bias", "true"),
     withKey("input_layers", prefix + "_ffn_geglu"),
     withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));

  // Residual connection (FFN)
  layers.push_back(createLayer(
    "addition",
    {withKey("name", prefix + "_block_output"),
     withKey("input_layers",
             prefix + "_self_attn_add," + prefix + "_ffn_down")}));

  return layers;
}

// ============================================================
// KV Projection model: projects encoder output to cross-attn K/V
// ============================================================
void MT5Transformer::constructKVProjectionModel() {
  kv_proj_model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);

  std::vector<LayerHandle> layers;

  // Input: encoder output (shape: 1:1:enc_seq_len:DIM)
  layers.push_back(createLayer(
    "input", {withKey("name", "kv_proj_input"),
              withKey("input_shape", "1:1:" + std::to_string(INIT_SEQ_LEN) +
                                      ":" + std::to_string(DIM))}));

  // For each decoder layer, create K and V projections
  for (int i = 0; i < NUM_DECODER_LAYERS; ++i) {
    std::string prefix = "kv_proj_layer" + std::to_string(i);

    // K projection for cross-attention
    layers.push_back(createLayer(
      "fully_connected",
      {withKey("name", prefix + "_cross_wk"),
       withKey("unit", NUM_HEADS * D_KV),
       withKey("disable_bias", "true"),
       withKey("input_layers", "kv_proj_input"),
       withKey("weight_initializer", "ones"),
       withKey("weight_dtype", FC_LAYER_DTYPE)}));

    // V projection for cross-attention
    layers.push_back(createLayer(
      "fully_connected",
      {withKey("name", prefix + "_cross_wv"),
       withKey("unit", NUM_HEADS * D_KV),
       withKey("disable_bias", "true"),
       withKey("input_layers", "kv_proj_input"),
       withKey("weight_initializer", "ones"),
       withKey("weight_dtype", FC_LAYER_DTYPE)}));
  }

  for (auto &layer : layers) {
    kv_proj_model->addLayer(layer);
  }
}

// ============================================================
// Decoder model construction
// ============================================================
void MT5Transformer::constructModel() {
  model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);

  std::vector<LayerHandle> layers;

  // Decoder token input
  layers.push_back(createLayer(
    "input", {withKey("name", "input0"),
              withKey("input_shape", "1:1:" + std::to_string(INIT_SEQ_LEN))}));

  // Pre-computed cross-attention K/V inputs (one pair per decoder layer)
  for (int i = 0; i < NUM_DECODER_LAYERS; ++i) {
    layers.push_back(createLayer(
      "input",
      {withKey("name", "cross_k_input_" + std::to_string(i)),
       withKey("input_shape", "1:1:" + std::to_string(INIT_SEQ_LEN) + ":" +
                                std::to_string(NUM_HEADS * D_KV))}));
    layers.push_back(createLayer(
      "input",
      {withKey("name", "cross_v_input_" + std::to_string(i)),
       withKey("input_shape", "1:1:" + std::to_string(INIT_SEQ_LEN) + ":" +
                                std::to_string(NUM_HEADS * D_KV))}));
  }

  // Decoder embedding
  const std::string embedding_type =
    TIE_WORD_EMBEDDINGS ? "tie_word_embeddings" : "embedding_layer";
  layers.push_back(
    createLayer(embedding_type,
                {"name=embedding0", "in_dim=" + std::to_string(NUM_VOCAB),
                 "weight_dtype=" + EMBEDDING_DTYPE,
                 "out_dim=" + std::to_string(DIM)}));

  // Decoder blocks
  for (int i = 0; i < NUM_DECODER_LAYERS; ++i) {
    std::string input_name =
      (i == 0) ? "embedding0"
               : "layer" + std::to_string(i - 1) + "_decoder_output";
    auto block = createTransformerDecoderBlock(i, input_name);
    layers.insert(layers.end(), block.begin(), block.end());
  }

  // Decoder final norm
  layers.push_back(createLayer(
    "rms_norm",
    {withKey("name", "output_norm"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("input_layers",
             "layer" + std::to_string(NUM_DECODER_LAYERS - 1) +
               "_decoder_output"),
     withKey("packed", "false")}));

  for (auto &layer : layers) {
    model->addLayer(layer);
  }
}

std::vector<LayerHandle>
MT5Transformer::createTransformerDecoderBlock(const int layer_id,
                                              std::string input_name) {
  std::vector<LayerHandle> layers;
  std::string prefix = "layer" + std::to_string(layer_id);

  // ===== Self-Attention Block =====
  // Pre-norm
  layers.push_back(createLayer(
    "rms_norm",
    {withKey("name", prefix + "_attention_norm"),
     withKey("input_layers", input_name),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));

  // Self-attention (causal, no RoPE)
  auto self_attn =
    createAttention(layer_id, INIT_SEQ_LEN, NUM_HEADS, D_KV,
                    prefix + "_attention_norm", prefix + "_attention_norm",
                    prefix + "_attention_norm");
  layers.insert(layers.end(), self_attn.begin(), self_attn.end());

  // Residual
  layers.push_back(createLayer(
    "addition",
    {withKey("name", prefix + "_self_attn_add"),
     withKey("input_layers",
             input_name + "," + prefix + "_attention_out")}));

  // ===== Cross-Attention Block =====
  // Pre-norm
  layers.push_back(createLayer(
    "rms_norm",
    {withKey("name", prefix + "_cross_attention_norm"),
     withKey("input_layers", prefix + "_self_attn_add"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));

  // Cross-attention
  auto cross_attn = createCrossAttention(
    layer_id, prefix + "_cross_attention_norm",
    "cross_k_input_" + std::to_string(layer_id),
    "cross_v_input_" + std::to_string(layer_id));
  layers.insert(layers.end(), cross_attn.begin(), cross_attn.end());

  // Residual
  layers.push_back(createLayer(
    "addition",
    {withKey("name", prefix + "_cross_attn_add"),
     withKey("input_layers",
             prefix + "_self_attn_add," + prefix + "_cross_attention_out")}));

  // ===== FFN Block =====
  // Pre-norm
  layers.push_back(createLayer(
    "rms_norm",
    {withKey("name", prefix + "_ffn_norm"),
     withKey("input_layers", prefix + "_cross_attn_add"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));

  // Gate projection + GeLU
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", prefix + "_ffn_gate"),
     withKey("unit", INTERMEDIATE_SIZE),
     withKey("disable_bias", "true"),
     withKey("input_layers", prefix + "_ffn_norm"),
     withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));

  layers.push_back(createLayer(
    "activation",
    {withKey("name", prefix + "_ffn_gate_gelu"),
     withKey("activation", "tanh_gelu"),
     withKey("input_layers", prefix + "_ffn_gate")}));

  // Up projection
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", prefix + "_ffn_up"),
     withKey("unit", INTERMEDIATE_SIZE),
     withKey("disable_bias", "true"),
     withKey("input_layers", prefix + "_ffn_norm"),
     withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));

  // Multiply (gate * up = GeGLU)
  layers.push_back(createLayer(
    "multiply",
    {withKey("name", prefix + "_ffn_geglu"),
     withKey("input_layers",
             prefix + "_ffn_gate_gelu," + prefix + "_ffn_up")}));

  // Down projection
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", prefix + "_ffn_down"),
     withKey("unit", DIM),
     withKey("disable_bias", "true"),
     withKey("input_layers", prefix + "_ffn_geglu"),
     withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));

  // Residual
  layers.push_back(createLayer(
    "addition",
    {withKey("name", prefix + "_decoder_output"),
     withKey("input_layers",
             prefix + "_cross_attn_add," + prefix + "_ffn_down")}));

  return layers;
}

// ============================================================
// Attention layer creation
// ============================================================
std::vector<LayerHandle> MT5Transformer::createAttention(
  const int layer_id, int seq_len, int n_heads, int head_dim,
  std::string query_name, std::string key_name, std::string value_name) {

  std::vector<LayerHandle> layers;
  std::string prefix = "layer" + std::to_string(layer_id);

  auto Q = prefix + "_wq";
  auto K = prefix + "_wk";
  auto V = prefix + "_wv";
  auto A = prefix + "_attention";
  auto O = prefix + "_attention_out";

  // Q projection
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", Q), withKey("unit", n_heads * head_dim),
     withKey("disable_bias", "true"), withKey("input_layers", query_name),
     withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));

  // K projection
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", K), withKey("unit", n_heads * head_dim),
     withKey("disable_bias", "true"), withKey("input_layers", key_name),
     withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));

  // V projection
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", V), withKey("unit", n_heads * head_dim),
     withKey("disable_bias", "true"), withKey("input_layers", value_name),
     withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));

  // MHA core (causal self-attention for decoder, no RoPE)
  layers.push_back(createLayer(
    "mha_core",
    {withKey("name", A), withKey("num_heads", n_heads),
     withKey("num_heads_kv", n_heads),
     withKey("max_timestep",
             std::to_string(INIT_SEQ_LEN + NUM_TO_GENERATE)),
     withKey("is_causal", "true"), // decoder self-attention is causal
     withKey("use_rope", "false"), // T5 uses relative position bias
     withKey("input_layers", Q + "," + K + "," + V)}));

  // O projection
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", O), withKey("unit", DIM),
     withKey("disable_bias", "true"), withKey("input_layers", A),
     withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));

  return layers;
}

std::vector<LayerHandle>
MT5Transformer::createCrossAttention(const int layer_id,
                                     std::string query_name,
                                     std::string cross_k_input,
                                     std::string cross_v_input) {
  std::vector<LayerHandle> layers;
  std::string prefix = "layer" + std::to_string(layer_id);

  auto Q = prefix + "_cross_wq";
  auto A = prefix + "_cross_attention";
  auto O = prefix + "_cross_attention_out";

  // Q projection (from decoder hidden state)
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", Q), withKey("unit", NUM_HEADS * D_KV),
     withKey("disable_bias", "true"), withKey("input_layers", query_name),
     withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));

  // MHA core (cross-attention: Q from decoder, K/V from encoder)
  layers.push_back(createLayer(
    "mha_core",
    {withKey("name", A), withKey("num_heads", NUM_HEADS),
     withKey("num_heads_kv", NUM_HEADS),
     withKey("max_timestep",
             std::to_string(INIT_SEQ_LEN + NUM_TO_GENERATE)),
     withKey("is_causal", "false"),
     withKey("is_cross_attention", "true"),
     withKey("use_rope", "false"),
     withKey("input_layers",
             Q + "," + cross_k_input + "," + cross_v_input)}));

  // O projection
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", O), withKey("unit", DIM),
     withKey("disable_bias", "true"), withKey("input_layers", A),
     withKey("weight_initializer", "ones"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));

  return layers;
}

std::vector<LayerHandle> MT5Transformer::createMlp(const int layer_id, int dim,
                                                   int hidden_dim,
                                                   std::string input_name) {
  // This is called by the base class but we inline FFN in encoder/decoder
  // blocks. Return empty - the caller (createEncoderBlock /
  // createTransformerDecoderBlock) handles FFN directly.
  return {};
}

void MT5Transformer::registerCustomLayers() {
  // Base class registers core layers (SwiGLU, RMSNorm, MHACore, etc.)
  // No additional custom layers needed for mT5
}

// ============================================================
// MT5CausalLM implementation
// ============================================================
MT5CausalLM::~MT5CausalLM() {
  // Free cross-attention K/V buffers
  for (auto *buf : cross_k_buffers)
    if (buf)
      free(buf);
  for (auto *buf : cross_v_buffers)
    if (buf)
      free(buf);
  if (encoder_output_buffer)
    free(encoder_output_buffer);
}

void MT5CausalLM::initialize() {
  // Register custom layers
  registerCustomLayers();

  // Build encoder model
  constructEncoder();

  // Build KV projection model
  constructKVProjectionModel();

  // Build decoder model (via CausalLM::constructModel -> MT5::constructModel)
  constructModel();

  // Add LM head to decoder
  const std::string lmhead_type =
    TIE_WORD_EMBEDDINGS ? "tie_word_embeddings" : "lm_head";
  std::vector<std::string> lmhead_prop = {
    withKey("name", "output_of_causallm"),
    withKey("unit", NUM_VOCAB),
    withKey("disable_bias", "true"),
    withKey("input_layers", "output_norm"),
    withKey("weight_dtype", LMHEAD_DTYPE),
  };
  if (TIE_WORD_EMBEDDINGS)
    lmhead_prop.emplace_back(withKey("shared_from", "embedding0"));
  model->addLayer(createLayer(lmhead_type, lmhead_prop));

  // Setup encoder model properties
  std::vector<std::string> enc_props = {
    withKey("batch_size", BATCH_SIZE), withKey("epochs", "1"),
    withKey("model_tensor_type", MODEL_TENSOR_TYPE)};
  encoder_model->setProperty(enc_props);
  encoder_model->compile(ml::train::ExecutionMode::INFERENCE);
  encoder_model->initialize(ml::train::ExecutionMode::INFERENCE);

  // Setup KV projection model properties
  kv_proj_model->setProperty(enc_props);
  kv_proj_model->compile(ml::train::ExecutionMode::INFERENCE);
  kv_proj_model->initialize(ml::train::ExecutionMode::INFERENCE);

  // Setup decoder model properties
  std::vector<std::string> model_props = {
    withKey("batch_size", BATCH_SIZE), withKey("epochs", "1"),
    withKey("model_tensor_type", MODEL_TENSOR_TYPE)};
  model->setProperty(model_props);
  model->compile(ml::train::ExecutionMode::INFERENCE);
  model->initialize(ml::train::ExecutionMode::INFERENCE);

  is_initialized = true;

  // Allocate encoder output buffer
  encoder_output_buffer =
    (float *)calloc(static_cast<size_t>(BATCH_SIZE) * INIT_SEQ_LEN * DIM,
                    sizeof(float));

  // Allocate cross-attention K/V buffers
  unsigned int kv_dim = NUM_HEADS * D_KV;
  for (int i = 0; i < NUM_DECODER_LAYERS; ++i) {
    cross_k_buffers.push_back(
      (float *)calloc(static_cast<size_t>(BATCH_SIZE) * INIT_SEQ_LEN * kv_dim,
                      sizeof(float)));
    cross_v_buffers.push_back(
      (float *)calloc(static_cast<size_t>(BATCH_SIZE) * INIT_SEQ_LEN * kv_dim,
                      sizeof(float)));
  }
}

void MT5CausalLM::load_weight(const std::string &weight_path) {
  if (!is_initialized) {
    throw std::runtime_error(
      "MT5 model is not initialized. Call initialize() first.");
  }

  // Weight file layout (from weight_converter.py):
  // [encoder_weights][kv_projection_weights][decoder_weights]
  //
  // We load each model from its own weight file:
  // - weight_path + ".encoder" for encoder model
  // - weight_path + ".kvproj" for KV projection model
  // - weight_path (main) for decoder model
  try {
    std::string enc_path = weight_path + ".encoder";
    std::string kvproj_path = weight_path + ".kvproj";

    encoder_model->load(enc_path, ml::train::ModelFormat::MODEL_FORMAT_BIN);
    kv_proj_model->load(kvproj_path, ml::train::ModelFormat::MODEL_FORMAT_BIN);
    model->load(weight_path, ml::train::ModelFormat::MODEL_FORMAT_BIN);

    std::cout << "[MT5] Loaded encoder weights from: " << enc_path << std::endl;
    std::cout << "[MT5] Loaded KV projection weights from: " << kvproj_path
              << std::endl;
    std::cout << "[MT5] Loaded decoder weights from: " << weight_path
              << std::endl;
  } catch (const std::exception &e) {
    throw std::runtime_error("Failed to load MT5 weights: " +
                             std::string(e.what()));
  }
}

void MT5CausalLM::runEncoder(const std::vector<int64_t> &input_ids,
                              unsigned int input_len) {
  encoder_seq_len = input_len;

  // Prepare encoder input
  float *enc_input =
    (float *)malloc(sizeof(float) * BATCH_SIZE * INIT_SEQ_LEN);
  for (unsigned int b = 0; b < BATCH_SIZE; ++b) {
    for (unsigned int i = 0; i < input_len; ++i) {
      enc_input[static_cast<size_t>(b) * INIT_SEQ_LEN + i] =
        static_cast<float>(input_ids[i]);
    }
  }

  // Run encoder
  std::vector<float *> enc_inputs = {enc_input};
  std::vector<float *> enc_labels = {};

  auto enc_output = encoder_model->incremental_inference(
    BATCH_SIZE, enc_inputs, enc_labels, input_len, 0, input_len, true);

  // Copy encoder output to buffer
  if (enc_output.size() > 0 && enc_output[0] != nullptr) {
    size_t enc_out_size =
      static_cast<size_t>(BATCH_SIZE) * input_len * DIM;
    memcpy(encoder_output_buffer, enc_output[0],
           enc_out_size * sizeof(float));
  }

  // Run KV projection model on encoder output
  std::vector<float *> kv_inputs = {encoder_output_buffer};
  auto kv_output = kv_proj_model->incremental_inference(
    BATCH_SIZE, kv_inputs, enc_labels, input_len, 0, input_len, true);

  // Extract K/V projections for each decoder layer
  // KV projection model outputs: [k0, v0, k1, v1, ...]
  unsigned int kv_dim = NUM_HEADS * D_KV;
  for (int i = 0; i < NUM_DECODER_LAYERS; ++i) {
    size_t kv_size =
      static_cast<size_t>(BATCH_SIZE) * input_len * kv_dim;
    if (kv_output.size() > static_cast<size_t>(i * 2 + 1)) {
      memcpy(cross_k_buffers[i], kv_output[i * 2],
             kv_size * sizeof(float));
      memcpy(cross_v_buffers[i], kv_output[i * 2 + 1],
             kv_size * sizeof(float));
    }
  }

  free(enc_input);
}

void MT5CausalLM::run(const WSTR prompt, bool do_sample,
                       const WSTR system_prompt, const WSTR tail_prompt) {
  if (!is_initialized) {
    throw std::runtime_error(
      "MT5 model is not initialized. Call initialize() before run().");
  }

  output_list.clear();
  for (unsigned int b = 0; b < BATCH_SIZE; ++b) {
    output_list.push_back("");
  }

  unsigned int generation_cnt = 0;

  // ===== Tokenize source input =====
  auto _input = tokenizer->Encode(prompt);
  unsigned int src_len = _input.size();
  if (src_len > INIT_SEQ_LEN)
    src_len = INIT_SEQ_LEN;

  std::vector<int64_t> src_ids(_input.begin(),
                               _input.begin() + src_len);

  std::cout << "[MT5] Source length: " << src_len << " tokens" << std::endl;

  // ===== Run encoder =====
  auto start_encode = std::chrono::high_resolution_clock::now();
  runEncoder(src_ids, src_len);
  auto finish_encode = std::chrono::high_resolution_clock::now();
  auto encode_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
    finish_encode - start_encode);

  std::cout << "[MT5] Encoder done: " << encode_duration.count() << " ms"
            << std::endl;

  // ===== Prepare decoder =====
  // Decoder starts with decoder_start_token_id (typically pad_token_id = 0)
  unsigned int decoder_start_token = BOS_TOKEN_ID;

  // Decoder input buffer
  float *dec_input =
    (float *)malloc(sizeof(float) * BATCH_SIZE * MAX_SEQ_LEN);
  memset(dec_input, 0, sizeof(float) * BATCH_SIZE * MAX_SEQ_LEN);

  for (unsigned int b = 0; b < BATCH_SIZE; ++b) {
    dec_input[static_cast<size_t>(b) * MAX_SEQ_LEN] =
      static_cast<float>(decoder_start_token);
    ids_history[static_cast<size_t>(b) * MAX_SEQ_LEN] = decoder_start_token;
  }

  std::vector<bool> eos_list(BATCH_SIZE, false);

  // Build input vector: [decoder_tokens, cross_k_0, cross_v_0, ...]
  std::vector<float *> dec_inputs;
  dec_inputs.push_back(dec_input);
  for (int i = 0; i < NUM_DECODER_LAYERS; ++i) {
    dec_inputs.push_back(cross_k_buffers[i]);
    dec_inputs.push_back(cross_v_buffers[i]);
  }
  std::vector<float *> dec_labels = {};

  // ===== Decoder prefill (just the start token) =====
  auto start_decode = std::chrono::high_resolution_clock::now();

  auto output = model->incremental_inference(
    BATCH_SIZE, dec_inputs, dec_labels, 1, 0, 1, false);

  // Generate first token from prefill output
  std::vector<unsigned int> id_list(generate(output[0], do_sample));

  registerOutputs(tokenizer, id_list, 1, eos_list);

  // ===== Token generation loop =====
  for (unsigned int b = 0; b < BATCH_SIZE; ++b) {
    dec_input[static_cast<size_t>(b) * MAX_SEQ_LEN] =
      static_cast<float>(id_list[b]);
  }

  for (unsigned int step = 2; step < 1 + NUM_TO_GENERATE; ++step) {
    auto output_step = model->incremental_inference(
      BATCH_SIZE, dec_inputs, dec_labels, 1, step - 1, step, false);

    std::vector<unsigned int> ids(generate(output_step[0], do_sample));

    for (unsigned int b = 0; b < BATCH_SIZE; ++b) {
      dec_input[static_cast<size_t>(b) * MAX_SEQ_LEN] =
        static_cast<float>(ids[b]);
    }

    registerOutputs(tokenizer, ids, step, eos_list);
    ++generation_cnt;

    // Check EOS
    bool all_done = true;
    for (unsigned int j = 0; j < BATCH_SIZE; ++j) {
      if (!eos_list[j] &&
          std::find(EOS_TOKEN_ID.begin(), EOS_TOKEN_ID.end(), ids[j]) !=
            EOS_TOKEN_ID.end()) {
        eos_list[j] = true;
      }
      if (!eos_list[j])
        all_done = false;
    }

    if (all_done)
      break;
  }

  auto finish_decode = std::chrono::high_resolution_clock::now();
  auto decode_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
    finish_decode - start_decode);

  std::cout << "\n\n";
  std::cout << "=================[ mT5 with NNTrainer ]===================\n";
  std::cout << "encode: " << src_len << " tokens, " << encode_duration.count()
            << " ms\n";
  std::cout << "decode: " << generation_cnt << " tokens, "
            << decode_duration.count() << " ms, "
            << ((double)generation_cnt /
                std::max<int64_t>(decode_duration.count(), 1) * 1000)
            << " TPS\n";
  std::cout << "==========================================================\n";

  free(dec_input);
}

void MT5CausalLM::registerCustomLayers() {
  CausalLM::registerCustomLayers();
  MT5Transformer::registerCustomLayers();
}

} // namespace causallm

// SPDX-License-Identifier: Apache-2.0
/**
 * @file   lfm2_vl_vision_transformer.cpp
 * @date   13 May 2026
 * @brief  CLIP/SigLIP-style Vision Transformer encoder.
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <lfm2_vl_vision_transformer.h>
#include <llm_util.hpp>
#include <model.h>

#include "../../../../image_util.h"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

namespace causallm {

namespace {

std::string toStringPrecise(float v) {
  std::ostringstream oss;
  oss << std::setprecision(20) << v;
  return oss.str();
}

const char *PATCH_EMBED_NAME = "v_patch_embd";
const char *PATCH_SEQ_NAME = "v_patch_seq";
const char *POS_EMBED_NAME = "v_pos_embd";
const char *POST_LN_NAME = "v_post_ln";

std::string blkName(int i, const char *suffix) {
  return "v_blk_" + std::to_string(i) + "_" + suffix;
}

} // namespace

json &Lfm2VlVisionTransformer::sanitizeConfig(json &cfg) {
  // base Transformer::setupParameters() calls .get<T>() on these without a
  // default, so missing keys would throw. Values written here are overwritten
  // by Lfm2VlVisionTransformer::setupParameters() immediately afterwards.
  if (!cfg.contains("vocab_size"))
    cfg["vocab_size"] = 0u;
  if (!cfg.contains("max_position_embeddings")) {
    unsigned int img = cfg.contains("naflex_tile_size")
                         ? cfg.value("naflex_tile_size", 512u)
                         : cfg.value("image_size", 256u);
    unsigned int patch = cfg.value("patch_size", 16u);
    cfg["max_position_embeddings"] = (img / patch) * (img / patch);
  }
  if (!cfg.contains("rope_theta"))
    cfg["rope_theta"] = 0u;
  if (!cfg.contains("tie_word_embeddings"))
    cfg["tie_word_embeddings"] = false;
  if (!cfg.contains("rms_norm_eps"))
    cfg["rms_norm_eps"] = cfg.value("layer_norm_eps", 1e-6f);
  if (!cfg.contains("num_attention_heads"))
    cfg["num_attention_heads"] = 12;
  if (!cfg.contains("hidden_size"))
    cfg["hidden_size"] = 768;
  if (!cfg.contains("image_height"))
    cfg["image_height"] = cfg.value("image_size", 256u);
  if (!cfg.contains("image_width"))
    cfg["image_width"] = cfg.value("image_size", 256u);
  if (!cfg.contains("num_hidden_layers"))
    cfg["num_hidden_layers"] = 12;
  if (!cfg.contains("intermediate_size"))
    cfg["intermediate_size"] = 3072;
  return cfg;
}

void Lfm2VlVisionTransformer::setupParameters(json &cfg, json &generation_cfg,
                                              json &nntr_cfg) {
  // Vision Transformer parameters. Defaults follow LFM2.5-VL's vision tower
  // (SigLIP2-style, 86M).
  DIM = cfg.value("hidden_size", 768);
  NUM_HEADS = cfg.value("num_attention_heads", 12);
  NUM_KEY_VALUE_HEADS = cfg.value("num_key_value_heads", NUM_HEADS);
  HEAD_DIM =
    cfg.contains("head_dim") ? cfg["head_dim"].get<int>() : DIM / NUM_HEADS;
  NUM_LAYERS = cfg.value("num_hidden_layers", 12);
  INTERMEDIATE_SIZE = cfg.value("intermediate_size", 3072);
  GQA_SIZE = NUM_HEADS / NUM_KEY_VALUE_HEADS;

  // ViT-specific.
  IMAGE_SIZE = cfg.value("image_size", 256u);
  PATCH_SIZE = cfg.value("patch_size", 16u);
  NUM_CHANNELS = cfg.value("num_channels", 3u);
  NAFLEX_BASE_GRID = cfg.value("naflex_base_grid", 16u);
  if (cfg.contains("naflex_tile_size"))
    IMAGE_SIZE = cfg.value("naflex_tile_size", IMAGE_SIZE);
  {
    unsigned int img_h = cfg.contains("naflex_tile_size")
                           ? IMAGE_SIZE
                           : cfg.value("image_height", IMAGE_SIZE);
    unsigned int img_w = cfg.contains("naflex_tile_size")
                           ? IMAGE_SIZE
                           : cfg.value("image_width", IMAGE_SIZE);
    PATCH_H = img_h / PATCH_SIZE;
    PATCH_W = img_w / PATCH_SIZE;
  }
  NUM_PATCHES = PATCH_H * PATCH_W;

  // Non-causal, no RoPE.
  IS_CAUSAL = false;
  ROPE_THETA = 0;
  NORM_EPS = cfg.value("layer_norm_eps", 1e-6f);

  // The "sequence length" of the vision graph is the patch grid size.
  // MAX_SEQ_LEN sizes the working/KV-cache memory pool; keep it generous
  // (TimmViT uses 1000) so the mha_core cache (max_timestep = NUM_PATCHES + 1)
  // and attention buffers fit. Read from nntr_config so it stays tunable.
  INIT_SEQ_LEN = NUM_PATCHES;
  MAX_SEQ_LEN = NUM_PATCHES + 1;
  MAX_POSITION_EMBEDDINGS = NUM_PATCHES + 1;

  // Misc.
  BATCH_SIZE = nntr_cfg.value("batch_size", 1u);
  MEMORY_SWAP = nntr_cfg.value("memory_swap", false);
  FSU_LOOKAHEAD = nntr_cfg.value("fsu_lookahead", 0u);
  MODEL_TENSOR_TYPE = nntr_cfg.value("model_tensor_type", std::string("FP32"));
  EMBEDDING_DTYPE = nntr_cfg.value("embedding_dtype", std::string("FP32"));
  FC_LAYER_DTYPE = nntr_cfg.value("fc_layer_dtype", std::string("FP32"));

  // Fields the LLM path uses but ViT does not. Kept defined so any base-class
  // code path that touches them stays well-formed.
  NUM_VOCAB = 0;
  TIE_WORD_EMBEDDINGS = false;
  EMBEDDING_SCALE = 1.0f;
  NUM_TO_GENERATE = 0;
  USE_VOCAB_SELECTION = false;
  SLIDING_WINDOW = UINT_MAX;
  ATTN_LOGIT_SOFTCAPPING = 0.0f;
}

std::pair<Tensor, Tensor> Lfm2VlVisionTransformer::constructModel() {

  // [B, C, H, W] image input.
  Tensor x = Tensor(
    {BATCH_SIZE, NUM_CHANNELS, PATCH_H * PATCH_SIZE, PATCH_W * PATCH_SIZE},
    "input0");

  // Conv2D patch embedding: kernel = stride = PATCH_SIZE, padding "valid".
  // Output: [B, DIM, H/P, W/P].
  const std::string ps = std::to_string(PATCH_SIZE);
  LayerHandle patch_embed(createLayer(
    "conv2d",
    {withKey("name", PATCH_EMBED_NAME), withKey("filters", DIM),
     withKey("kernel_size", ps + "," + ps), withKey("stride", ps + "," + ps),
     withKey("padding", "0,0"), withKey("bias_initializer", "zeros"),
     withKey("weight_initializer", "xavier_uniform")}));
  Tensor patch = patch_embed(x);

  // Conv2D output is [B, DIM, H/P, W/P] (channel-major). Reshape to
  // [B, 1, DIM, NUM_PATCHES] then permute the DIM and patch axes to get the
  // proper token sequence [B, 1, NUM_PATCHES, DIM] (each row = one patch's
  // full DIM feature vector). Mirrors TimmViT's patch_embed reorder.
  LayerHandle patch_seq(createLayer(
    "reshape", {withKey("name", PATCH_SEQ_NAME),
                withKey("target_shape", "1:" + std::to_string(DIM) + ":" +
                                          std::to_string(NUM_PATCHES))}));
  Tensor h = patch_seq(patch);

  LayerHandle patch_perm(
    createLayer("permute", {withKey("name", "v_patch_perm"),
                            withKey("direction", {1, 3, 2})}));
  h = patch_perm(h);

  // Learnable position embedding [NUM_PATCHES, DIM], added to the patch
  // sequence (loaded from the weight file as tensor "v_pos_embd").
  LayerHandle pos_embed(
    createLayer("weight", {withKey("name", POS_EMBED_NAME),
                           withKey("dim", "1:1:" + std::to_string(NUM_PATCHES) +
                                            ":" + std::to_string(DIM)),
                           withKey("tensor_dtype", "FP32"),
                           withKey("weight_name", POS_EMBED_NAME)}));
  Tensor pos = pos_embed(x);

  LayerHandle pos_add(createLayer("addition", {withKey("name", "v_pos_add")}));
  h = pos_add({h, pos});

  // Encoder blocks.
  for (int i = 0; i < NUM_LAYERS; ++i) {
    h = createEncoderBlock(i, h);
  }

  // Final post-LN.
  LayerHandle post_ln(createLayer(
    "layer_normalization", {withKey("name", POST_LN_NAME),
                            withKey("epsilon", toStringPrecise(NORM_EPS)),
                            withKey("axis", 3), withKey("packed", "false")}));
  h = post_ln(h);

  return {x, h};
}

Tensor Lfm2VlVisionTransformer::createEncoderBlock(int layer_id, Tensor input) {

  // Pre-LN.
  LayerHandle ln1(createLayer(
    "layer_normalization", {withKey("name", blkName(layer_id, "ln1")),
                            withKey("epsilon", toStringPrecise(NORM_EPS)),
                            withKey("axis", 3), withKey("packed", "false")}));
  Tensor x = ln1(input);

  // Self-attention.
  Tensor a = createSelfAttention(layer_id, x);

  // Residual.
  LayerHandle attn_res(
    createLayer("addition", {withKey("name", blkName(layer_id, "attn_res"))}));
  Tensor h = attn_res({input, a});

  // Pre-LN before MLP.
  LayerHandle ln2(createLayer(
    "layer_normalization", {withKey("name", blkName(layer_id, "ln2")),
                            withKey("epsilon", toStringPrecise(NORM_EPS)),
                            withKey("axis", 3), withKey("packed", "false")}));
  Tensor n = ln2(h);

  // MLP.
  Tensor m = createVitMlp(layer_id, n);

  // Residual.
  LayerHandle ffn_res(
    createLayer("addition", {withKey("name", blkName(layer_id, "ffn_res"))}));
  return ffn_res({h, m});
}

Tensor Lfm2VlVisionTransformer::createSelfAttention(int layer_id, Tensor x) {

  // Q / K / V projections (CLIP/SigLIP-style: bias enabled).
  LayerHandle wq(createLayer(
    "fully_connected",
    {withKey("name", blkName(layer_id, "attn_q")),
     withKey("unit", HEAD_DIM * NUM_HEADS), withKey("disable_bias", "false"),
     withKey("weight_initializer", "xavier_uniform")}));
  Tensor q = wq(x);

  LayerHandle wk(createLayer(
    "fully_connected", {withKey("name", blkName(layer_id, "attn_k")),
                        withKey("unit", HEAD_DIM * NUM_HEADS / GQA_SIZE),
                        withKey("disable_bias", "false"),
                        withKey("weight_initializer", "xavier_uniform")}));
  Tensor k = wk(x);

  LayerHandle wv(createLayer(
    "fully_connected", {withKey("name", blkName(layer_id, "attn_v")),
                        withKey("unit", HEAD_DIM * NUM_HEADS / GQA_SIZE),
                        withKey("disable_bias", "false"),
                        withKey("weight_initializer", "xavier_uniform")}));
  Tensor v = wv(x);

  // External KV cache placeholders: FP32 dtype so that mha_core's
  // apply_rotary_emb_tensor_v2 copies FP32 key/value into FP32 cache memory
  // (not UINT16).  The ViT encoder is a single-pass encoder, not an
  // autoregressive decoder, so FP32 cache is both correct and sufficient.
  const unsigned int kv_width =
    static_cast<unsigned int>(HEAD_DIM * NUM_HEADS / GQA_SIZE);
  const std::string cache_shape =
    std::to_string(BATCH_SIZE) + ":1:" + std::to_string(MAX_SEQ_LEN) + ":" +
    std::to_string(kv_width);
  LayerHandle cache_k_input(createLayer(
    "input", {withKey("name", "cache_k_l" + std::to_string(layer_id)),
              withKey("input_shape", cache_shape),
              withKey("input_dtype", "FP32")}));
  LayerHandle cache_v_input(createLayer(
    "input", {withKey("name", "cache_v_l" + std::to_string(layer_id)),
              withKey("input_shape", cache_shape),
              withKey("input_dtype", "FP32")}));
  Tensor cache_k = cache_k_input(Tensor());
  Tensor cache_v = cache_v_input(Tensor());

  // Bidirectional attention, no RoPE, external KV cache.
  LayerHandle mha(createLayer(
    "mha_core",
    {withKey("name", blkName(layer_id, "attn")),
     withKey("num_heads", NUM_HEADS),
     withKey("num_heads_kv", NUM_HEADS / GQA_SIZE),
     withKey("max_timestep", std::to_string(NUM_PATCHES + 1)),
     withKey("use_rope", "false"),
     withKey("is_causal", "false")}));
  Tensor a = mha({q, k, v, cache_k, cache_v});

  // Output projection (with bias).
  LayerHandle wo(createLayer(
    "fully_connected", {withKey("name", blkName(layer_id, "attn_out")),
                        withKey("unit", DIM), withKey("disable_bias", "false"),
                        withKey("weight_initializer", "xavier_uniform")}));
  return wo(a);
}

Tensor Lfm2VlVisionTransformer::createVitMlp(int layer_id, Tensor x) {

  LayerHandle up(createLayer(
    "fully_connected",
    {withKey("name", blkName(layer_id, "ffn_up")),
     withKey("unit", INTERMEDIATE_SIZE), withKey("disable_bias", "false"),
     withKey("weight_initializer", "xavier_uniform")}));
  Tensor h = up(x);

  LayerHandle act(
    createLayer("activation", {withKey("name", blkName(layer_id, "ffn_act")),
                               withKey("activation", "tanh_gelu")}));
  h = act(h);

  LayerHandle down(createLayer(
    "fully_connected", {withKey("name", blkName(layer_id, "ffn_down")),
                        withKey("unit", DIM), withKey("disable_bias", "false"),
                        withKey("weight_initializer", "xavier_uniform")}));
  return down(h);
}

void Lfm2VlVisionTransformer::allocateAndBindVitKVCache() {
  if (!vit_kv_cache_.isAllocated()) {
    // FP32 cache: the ViT encoder is a single-pass encoder, not autoregressive.
    // Using FP32 avoids the UINT16 memcpy bug in UIntTensor::copyData(FP32).
    const auto cache_dtype = ml::train::TensorDim::DataType::FP32;
    vit_kv_cache_.allocate(
      static_cast<unsigned int>(NUM_LAYERS), BATCH_SIZE,
      static_cast<unsigned int>(MAX_SEQ_LEN),
      static_cast<unsigned int>(NUM_HEADS / GQA_SIZE),
      static_cast<unsigned int>(HEAD_DIM), cache_dtype);
    vit_kv_cache_bound_ = false;
  }

  if (vit_kv_cache_bound_)
    return;

  // Bind each layer's FP32 KV buffers to the cache_k_l{i} / cache_v_l{i}
  // input placeholders created by createSelfAttention().
  auto find_placeholder = [this](const std::string &base) {
    for (const auto &suffix : {":0", ":input0", ":out0", ""}) {
      auto *t = model->getTensor(base + suffix);
      if (t != nullptr)
        return t;
    }
    return static_cast<nntrainer::Tensor *>(nullptr);
  };

  for (int i = 0; i < NUM_LAYERS; ++i) {
    auto &kc = vit_kv_cache_.getKeyCache(i);
    auto &vc = vit_kv_cache_.getValueCache(i);

    auto *kp = find_placeholder("cache_k_l" + std::to_string(i));
    auto *vp = find_placeholder("cache_v_l" + std::to_string(i));

    if (kp == nullptr || vp == nullptr)
      throw std::runtime_error(
        "allocateAndBindVitKVCache: placeholder not found for layer " +
        std::to_string(i));

    kp->setData(kc.getMemoryData(), kc.getOffset(), false);
    vp->setData(vc.getMemoryData(), vc.getOffset(), false);
  }

  vit_kv_cache_bound_ = true;
}

/* static */ std::vector<float>
Lfm2VlVisionTransformer::naflexInterpPosEmbed(const std::vector<float> &src,
                                              int src_h, int src_w, int dst_h,
                                              int dst_w) {
  if (src_h == dst_h && src_w == dst_w)
    return src;

  int dim = static_cast<int>(src.size()) / (src_h * src_w);
  std::vector<float> dst(static_cast<size_t>(dst_h) * dst_w * dim);

  for (int dy = 0; dy < dst_h; ++dy) {
    float sy_f = (dy + 0.5f) * static_cast<float>(src_h) / dst_h - 0.5f;
    sy_f = std::max(0.0f, std::min(sy_f, static_cast<float>(src_h - 1)));
    int sy0 = static_cast<int>(sy_f);
    int sy1 = std::min(sy0 + 1, src_h - 1);
    float ty = sy_f - sy0;

    for (int dx = 0; dx < dst_w; ++dx) {
      float sx_f = (dx + 0.5f) * static_cast<float>(src_w) / dst_w - 0.5f;
      sx_f = std::max(0.0f, std::min(sx_f, static_cast<float>(src_w - 1)));
      int sx0 = static_cast<int>(sx_f);
      int sx1 = std::min(sx0 + 1, src_w - 1);
      float tx = sx_f - sx0;

      const float *v00 = src.data() + (sy0 * src_w + sx0) * dim;
      const float *v01 = src.data() + (sy0 * src_w + sx1) * dim;
      const float *v10 = src.data() + (sy1 * src_w + sx0) * dim;
      const float *v11 = src.data() + (sy1 * src_w + sx1) * dim;
      float *out = dst.data() + (dy * dst_w + dx) * dim;

      for (int d = 0; d < dim; ++d) {
        float top = v00[d] * (1.0f - tx) + v01[d] * tx;
        float bot = v10[d] * (1.0f - tx) + v11[d] * tx;
        out[d] = top * (1.0f - ty) + bot * ty;
      }
    }
  }
  return dst;
}

void Lfm2VlVisionTransformer::load_weight(const std::string &weight_path) {
  const size_t dim = static_cast<size_t>(DIM);
  const size_t patch_size = static_cast<size_t>(PATCH_SIZE);
  const size_t n_channels = static_cast<size_t>(NUM_CHANNELS);
  const size_t base_n =
    static_cast<size_t>(NAFLEX_BASE_GRID) * NAFLEX_BASE_GRID;
  const size_t target_n = static_cast<size_t>(PATCH_H) * PATCH_W;

  // The HF converter writes patch_embedding.weight as a Linear weight
  // (dim, n_channels * patch_size^2) with input dimensions ordered HWC:
  //   col_idx = h * patch_size * n_channels + w * n_channels + c
  // The C++ ViT uses a Conv2D layer which expects CHW-ordered input:
  //   col_idx = c * patch_size^2 + h * patch_size + w
  // Reorder columns from HWC to CHW so Conv2D produces the correct output.
  const size_t in_dim = n_channels * patch_size * patch_size; // 768
  const size_t filter_floats = dim * in_dim;                  // 589824

  auto reorderPatchWeightHwcToChw =
    [&](std::vector<char> &buf, size_t offset) {
      const size_t P = patch_size;
      const size_t C = n_channels;
      std::vector<float> src(filter_floats);
      std::memcpy(src.data(), buf.data() + offset,
                  filter_floats * sizeof(float));
      // Build HWC->CHW column permutation:
      // for CHW index (c,h,w): chw_idx = c*P*P + h*P + w
      //                        hwc_idx = h*P*C + w*C + c
      std::vector<size_t> perm(in_dim);
      for (size_t c = 0; c < C; ++c)
        for (size_t h = 0; h < P; ++h)
          for (size_t w = 0; w < P; ++w)
            perm[c * P * P + h * P + w] = h * P * C + w * C + c;
      // Apply permutation to every row of the weight matrix
      std::vector<float> dst(filter_floats);
      for (size_t j = 0; j < dim; ++j)
        for (size_t chw = 0; chw < in_dim; ++chw)
          dst[j * in_dim + chw] = src[j * in_dim + perm[chw]];
      std::memcpy(buf.data() + offset, dst.data(),
                  filter_floats * sizeof(float));
    };

  const size_t bias_floats = dim;
  const size_t pre_pos_bytes = (filter_floats + bias_floats) * sizeof(float);
  const size_t pos_base_bytes = base_n * dim * sizeof(float);

  std::ifstream fin(weight_path, std::ios::binary | std::ios::ate);
  if (!fin)
    throw std::runtime_error(
      "Lfm2VlVisionTransformer::load_weight: cannot open: " + weight_path);
  auto file_size = static_cast<size_t>(fin.tellg());
  fin.seekg(0, std::ios::beg);
  std::vector<char> file_bytes(file_size);
  fin.read(file_bytes.data(), static_cast<std::streamsize>(file_size));
  fin.close();

  if (file_size < pre_pos_bytes + pos_base_bytes)
    throw std::runtime_error(
      "Lfm2VlVisionTransformer::load_weight: bin too small for pos_embed");

  if (base_n == target_n) {
    // No pos-embed interpolation needed; only reorder patch weight.
    reorderPatchWeightHwcToChw(file_bytes, 0);
    std::string tmp_path = weight_path + ".naflex_tmp.bin";
    {
      std::ofstream fout(tmp_path, std::ios::binary);
      if (!fout)
        throw std::runtime_error(
          "Lfm2VlVisionTransformer::load_weight: cannot write tmp: " +
          tmp_path);
      fout.write(file_bytes.data(),
                 static_cast<std::streamsize>(file_size));
    }
    try {
      Transformer::load_weight(tmp_path);
    } catch (...) {
      std::remove(tmp_path.c_str());
      throw;
    }
    std::remove(tmp_path.c_str());
    return;
  }

  std::vector<float> base_pos(base_n * dim);
  std::memcpy(base_pos.data(), file_bytes.data() + pre_pos_bytes,
              pos_base_bytes);

  auto interp_pos = naflexInterpPosEmbed(
    base_pos, static_cast<int>(NAFLEX_BASE_GRID),
    static_cast<int>(NAFLEX_BASE_GRID), static_cast<int>(PATCH_H),
    static_cast<int>(PATCH_W));

  const size_t target_pos_bytes = target_n * dim * sizeof(float);
  const size_t tail_offset = pre_pos_bytes + pos_base_bytes;
  const size_t tail_size = file_size - tail_offset;
  const size_t new_size = pre_pos_bytes + target_pos_bytes + tail_size;

  std::vector<char> patched(new_size);
  std::memcpy(patched.data(), file_bytes.data(), pre_pos_bytes);
  std::memcpy(patched.data() + pre_pos_bytes, interp_pos.data(),
              target_pos_bytes);
  if (tail_size > 0)
    std::memcpy(patched.data() + pre_pos_bytes + target_pos_bytes,
                file_bytes.data() + tail_offset, tail_size);

  // Reorder patch embedding columns from HWC (Linear) to CHW (Conv2D).
  reorderPatchWeightHwcToChw(patched, 0);

  std::string tmp_path = weight_path + ".naflex_tmp.bin";
  {
    std::ofstream fout(tmp_path, std::ios::binary);
    if (!fout)
      throw std::runtime_error(
        "Lfm2VlVisionTransformer::load_weight: cannot write tmp: " + tmp_path);
    fout.write(patched.data(), static_cast<std::streamsize>(new_size));
  }

  try {
    Transformer::load_weight(tmp_path);
  } catch (...) {
    std::remove(tmp_path.c_str());
    throw;
  }
  std::remove(tmp_path.c_str());
}

std::vector<float>
Lfm2VlVisionTransformer::runOnTile(const std::vector<float> &tile_pixels) {
  if (!is_initialized)
    throw std::runtime_error("runOnTile: call initialize() first");

  const size_t expected = static_cast<size_t>(BATCH_SIZE) * NUM_CHANNELS *
                          PATCH_H * PATCH_SIZE * PATCH_W * PATCH_SIZE;
  if (tile_pixels.size() != expected)
    throw std::runtime_error("runOnTile: pixel buffer size mismatch: got " +
                             std::to_string(tile_pixels.size()) +
                             " expected " + std::to_string(expected));

  allocateAndBindVitKVCache();
  vit_kv_cache_.reset();

  std::vector<std::pair<std::string, float *>> cache_inputs;
  cache_inputs.reserve(static_cast<size_t>(NUM_LAYERS) * 2);
  for (int i = 0; i < NUM_LAYERS; ++i) {
    cache_inputs.emplace_back(
      "cache_k_l" + std::to_string(i),
      reinterpret_cast<float *>(vit_kv_cache_.getKeyCache(i).getData()));
    cache_inputs.emplace_back(
      "cache_v_l" + std::to_string(i),
      reinterpret_cast<float *>(vit_kv_cache_.getValueCache(i).getData()));
  }
  std::sort(cache_inputs.begin(), cache_inputs.end(),
            [](const auto &a, const auto &b) { return a.first < b.first; });

  std::vector<float *> in_ptrs;
  in_ptrs.reserve(1 + cache_inputs.size());
  in_ptrs.push_back(const_cast<float *>(tile_pixels.data()));
  for (const auto &ci : cache_inputs)
    in_ptrs.push_back(ci.second);

  std::vector<float *> vit_label;
  auto out_ptrs = model->incremental_inference(BATCH_SIZE, in_ptrs, vit_label,
                                               NUM_PATCHES, 0, NUM_PATCHES,
                                               false);

  if (out_ptrs.empty() || out_ptrs[0] == nullptr)
    throw std::runtime_error("runOnTile: ViT produced no output");

  const size_t n_out = static_cast<size_t>(BATCH_SIZE) * NUM_PATCHES * DIM;
  return std::vector<float>(out_ptrs[0], out_ptrs[0] + n_out);
}

void Lfm2VlVisionTransformer::run(const WSTR image_tensor_path, bool,
                                  const WSTR, const WSTR, bool log_output) {

  if (!is_initialized) {
    throw std::runtime_error(
      "Lfm2VlVisionTransformer is not initialized. Call initialize() first.");
  }

  // Load image: decode, resize to IMAGE_SIZE x IMAGE_SIZE, normalize (SigLIP2: mean=std=0.5).
  const size_t n_elems =
    static_cast<size_t>(BATCH_SIZE) * NUM_CHANNELS * PATCH_H * PATCH_SIZE *
    PATCH_W * PATCH_SIZE;
  std::vector<float> image;

  if (image_tensor_path.empty())
    throw std::invalid_argument(
      "Lfm2VlVisionTransformer::run(): image_path is empty. "
      "Set 'image_path' in nntr_config.json to a valid JPEG/PNG/BMP file.");

  std::cout << "[LFM2-VL ViT] loading image file: " << image_tensor_path
            << " -> " << IMAGE_SIZE << "x" << IMAGE_SIZE << std::endl;
  image = loadAndPreprocessImage(image_tensor_path,
                                 static_cast<int>(IMAGE_SIZE),
                                 static_cast<int>(IMAGE_SIZE), true);
  if (image.size() != n_elems)
    throw std::runtime_error(
      "loadAndPreprocessImage returned unexpected size for '" +
      image_tensor_path + "'. Expected " + std::to_string(n_elems) +
      " fp32 elements (NCHW=" + std::to_string(BATCH_SIZE) + "x" +
      std::to_string(NUM_CHANNELS) + "x" +
      std::to_string(PATCH_H * PATCH_SIZE) + "x" +
      std::to_string(PATCH_W * PATCH_SIZE) + "). " +
      "Ensure the file is a decodable JPEG/PNG/BMP image.");

  runInference(image, log_output);
}

void Lfm2VlVisionTransformer::runFromPixels(const float *chw, size_t n_elems,
                                            bool log_output) {
  if (!is_initialized)
    throw std::runtime_error("Lfm2VlVisionTransformer is not initialized. "
                             "Call initialize() first.");
  const size_t expected =
    static_cast<size_t>(BATCH_SIZE) * NUM_CHANNELS * PATCH_H * PATCH_SIZE *
    PATCH_W * PATCH_SIZE;
  if (chw == nullptr || n_elems != expected)
    throw std::runtime_error(
      "Lfm2VlVisionTransformer::runFromPixels: expected " +
      std::to_string(expected) + " NCHW fp32 elements, got " +
      std::to_string(n_elems));
  std::vector<float> image(chw, chw + n_elems);
  runInference(image, log_output);
}

void Lfm2VlVisionTransformer::runInference(const std::vector<float> &image,
                                           bool log_output) {
  // Allocate and bind external KV cache buffers (no-op after first call).
  allocateAndBindVitKVCache();
  vit_kv_cache_.reset();

  // Build inference input list: [image] + [cache_k_l0, cache_v_l0, ..., cache_k_l11, cache_v_l11]
  // sorted by name to match the order getInputDimension() returns.
  std::vector<std::pair<std::string, float *>> cache_inputs;
  cache_inputs.reserve(static_cast<size_t>(NUM_LAYERS) * 2);
  for (int i = 0; i < NUM_LAYERS; ++i) {
    cache_inputs.emplace_back(
      "cache_k_l" + std::to_string(i),
      reinterpret_cast<float *>(vit_kv_cache_.getKeyCache(i).getData()));
    cache_inputs.emplace_back(
      "cache_v_l" + std::to_string(i),
      reinterpret_cast<float *>(vit_kv_cache_.getValueCache(i).getData()));
  }
  std::sort(cache_inputs.begin(), cache_inputs.end(),
            [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });

  std::vector<float *> in_ptrs;
  in_ptrs.reserve(1 + cache_inputs.size());
  in_ptrs.push_back(const_cast<float *>(image.data()));
  for (const auto &ci : cache_inputs)
    in_ptrs.push_back(ci.second);
  int vit_iters = 1;
  if (const char *it = std::getenv("NNTR_VIT_ITERS"))
    vit_iters = std::max(1, std::atoi(it));
  std::vector<float *> out_ptrs;
  std::vector<float *> vit_label;
  auto vit_t0 = std::chrono::high_resolution_clock::now();
  for (int r = 0; r < vit_iters; ++r)
    out_ptrs = model->incremental_inference(BATCH_SIZE, in_ptrs, vit_label,
                                            NUM_PATCHES, 0, NUM_PATCHES, false);
  auto vit_t1 = std::chrono::high_resolution_clock::now();
  std::cout
    << "[vit infer] "
    << std::chrono::duration<double, std::milli>(vit_t1 - vit_t0).count() /
         vit_iters
    << " ms/iter over " << vit_iters << " iters" << std::endl;

  // Debug: dump the full feature tensor to a file for parity comparison.
  if (const char *out_path = std::getenv("NNTR_VIT_OUT")) {
    if (!out_ptrs.empty() && out_ptrs[0] != nullptr) {
      const size_t n_out = static_cast<size_t>(BATCH_SIZE) * NUM_PATCHES * DIM;
      std::ofstream of(out_path, std::ios::binary);
      of.write(reinterpret_cast<const char *>(out_ptrs[0]),
               static_cast<std::streamsize>(n_out * sizeof(float)));
      std::cout << "[NNTR_VIT_OUT] wrote " << n_out << " floats to " << out_path
                << std::endl;
    }
  }

  if (log_output && !out_ptrs.empty() && out_ptrs[0] != nullptr) {
    std::cout << "Lfm2VlVisionTransformer features [" << NUM_PATCHES << "x"
              << DIM << "], first 10 values: ";
    for (int i = 0; i < 10 && i < DIM; ++i) {
      std::cout << out_ptrs[0][i] << (i == 9 ? "" : ", ");
    }
    std::cout << " ..." << std::endl;
  }

  // Cache the full output feature tensor for downstream connector use.
  if (!out_ptrs.empty() && out_ptrs[0] != nullptr) {
    const size_t n_out = static_cast<size_t>(BATCH_SIZE) * NUM_PATCHES * DIM;
    last_features_.assign(out_ptrs[0], out_ptrs[0] + n_out);
  }
}

} // namespace causallm

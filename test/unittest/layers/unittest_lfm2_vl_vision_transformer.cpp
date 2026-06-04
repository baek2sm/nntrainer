// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_lfm2_vl_vision_transformer.cpp
 * @date   13 May 2026
 * @brief  Unit tests for Lfm2VlVisionTransformer (CLIP/SigLIP-style ViT
 * encoder). These tests exercise structure-only behavior: config parsing,
 *         parameter defaults, and that initialize() builds + compiles the
 *         symbolic graph without throwing on a tiny toy configuration.
 *         No GGUF / pretrained weights are required.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <gtest/gtest.h>

#include <lfm2_vl_vision_transformer.h>

#include <vector>

namespace {
static std::vector<float> interpPosEmbed(const float *src, unsigned int src_h,
                                         unsigned int src_w, unsigned int dst_h,
                                         unsigned int dst_w, unsigned int dim) {
  if (src_h == dst_h && src_w == dst_w)
    return std::vector<float>(src, src + src_h * src_w * dim);
  std::vector<float> out(dst_h * dst_w * dim);
  for (unsigned int dy = 0; dy < dst_h; ++dy) {
    float sy_f = (dy + 0.5f) * (float(src_h) / float(dst_h)) - 0.5f;
    if (sy_f < 0.f)
      sy_f = 0.f;
    if (sy_f > float(src_h) - 1.f)
      sy_f = float(src_h) - 1.f;
    auto sy0 = static_cast<unsigned int>(sy_f);
    unsigned int sy1 = (sy0 + 1u < src_h) ? sy0 + 1u : src_h - 1u;
    float ty = sy_f - float(sy0);
    for (unsigned int dx = 0; dx < dst_w; ++dx) {
      float sx_f = (dx + 0.5f) * (float(src_w) / float(dst_w)) - 0.5f;
      if (sx_f < 0.f)
        sx_f = 0.f;
      if (sx_f > float(src_w) - 1.f)
        sx_f = float(src_w) - 1.f;
      auto sx0 = static_cast<unsigned int>(sx_f);
      unsigned int sx1 = (sx0 + 1u < src_w) ? sx0 + 1u : src_w - 1u;
      float tx = sx_f - float(sx0);
      unsigned int oi = (dy * dst_w + dx) * dim;
      for (unsigned int d = 0; d < dim; ++d) {
        out[oi + d] =
          (1.f - ty) * ((1.f - tx) * src[(sy0 * src_w + sx0) * dim + d] +
                        tx * src[(sy0 * src_w + sx1) * dim + d]) +
          ty * ((1.f - tx) * src[(sy1 * src_w + sx0) * dim + d] +
                tx * src[(sy1 * src_w + sx1) * dim + d]);
      }
    }
  }
  return out;
}

/**
 * @brief Test-only subclass that exposes a handful of the base Transformer's
 *        protected fields. We only need read access — the test does not
 *        mutate any internal state.
 */
class TestableLfm2Vl : public causallm::Lfm2VlVisionTransformer {
public:
  using causallm::Lfm2VlVisionTransformer::Lfm2VlVisionTransformer;

  unsigned int imageSize() const { return IMAGE_SIZE; }
  unsigned int patchSize() const { return PATCH_SIZE; }
  unsigned int numChannels() const { return NUM_CHANNELS; }
  unsigned int numPatches() const { return NUM_PATCHES; }
  unsigned int patchH() const { return PATCH_H; }
  unsigned int patchW() const { return PATCH_W; }
  int hiddenDim() const { return DIM; }
  int numLayers() const { return NUM_LAYERS; }
  int numHeads() const { return NUM_HEADS; }
  int headDim() const { return HEAD_DIM; }
  int intermediateSize() const { return INTERMEDIATE_SIZE; }
  unsigned int initSeqLen() const { return INIT_SEQ_LEN; }
  bool isCausal() const { return IS_CAUSAL; }
  unsigned int ropeTheta() const { return ROPE_THETA; }
  bool initialized() const { return is_initialized; }
};

/**
 * @brief Helper: build the three JSON blobs Lfm2VlVisionTransformer expects.
 *        Sized small enough for the graph to compile in a few hundred ms.
 */
struct ToyConfig {
  causallm::json cfg;
  causallm::json generation_cfg;
  causallm::json nntr_cfg;
};

ToyConfig makeToyConfig() {
  ToyConfig c;
  c.cfg = {{"hidden_size", 16},      {"num_attention_heads", 2},
           {"num_hidden_layers", 2}, {"intermediate_size", 32},
           {"image_size", 32},       {"patch_size", 16},
           {"num_channels", 3},      {"layer_norm_eps", 1e-6}};
  c.generation_cfg = causallm::json::object();
  // Provide every field the base Transformer::setupParameters() reads from
  // nntr_cfg via the unchecked [] operator (it will throw otherwise).
  c.nntr_cfg = {{"batch_size", 1},
                {"model_type", "embedding"},
                {"model_tensor_type", "FP32-FP32"},
                {"embedding_dtype", "FP32"},
                {"fc_layer_dtype", "FP32"},
                {"init_seq_len", 4},
                {"max_seq_len", 4},
                {"num_to_generate", 0},
                {"skip_tokenizer", true}};
  return c;
}

causallm::json makeFullNntrCfg() {
  return {{"batch_size", 1},
          {"model_type", "embedding"},
          {"model_tensor_type", "FP32-FP32"},
          {"embedding_dtype", "FP32"},
          {"fc_layer_dtype", "FP32"},
          {"init_seq_len", 256},
          {"max_seq_len", 256},
          {"num_to_generate", 0},
          {"skip_tokenizer", true}};
}

} // namespace

TEST(Lfm2VlVisionTransformer, setup_parameters_picks_up_overrides) {
  auto toy = makeToyConfig();
  TestableLfm2Vl vit(toy.cfg, toy.generation_cfg, toy.nntr_cfg);

  EXPECT_EQ(vit.imageSize(), 32u);
  EXPECT_EQ(vit.patchSize(), 16u);
  EXPECT_EQ(vit.numChannels(), 3u);
  // (image_size / patch_size) ^ 2 = (32/16)^2 = 4.
  EXPECT_EQ(vit.numPatches(), 4u);
  EXPECT_EQ(vit.hiddenDim(), 16);
  EXPECT_EQ(vit.numHeads(), 2);
  // head_dim defaults to hidden_size / num_attention_heads = 16 / 2 = 8.
  EXPECT_EQ(vit.headDim(), 8);
  EXPECT_EQ(vit.numLayers(), 2);
  EXPECT_EQ(vit.intermediateSize(), 32);

  // ViT is non-causal, no RoPE.
  EXPECT_FALSE(vit.isCausal());
  EXPECT_EQ(vit.ropeTheta(), 0u);

  // INIT_SEQ_LEN mirrors NUM_PATCHES.
  EXPECT_EQ(vit.initSeqLen(), 4u);
}

TEST(Lfm2VlVisionTransformer,
     setup_parameters_uses_defaults_when_fields_missing) {
  // Empty cfg: every field must fall back to the LFM2.5-VL / SigLIP2 defaults
  // baked into setupParameters().
  causallm::json cfg = causallm::json::object();
  causallm::json generation_cfg = causallm::json::object();
  causallm::json nntr_cfg = makeFullNntrCfg();
  TestableLfm2Vl vit(cfg, generation_cfg, nntr_cfg);

  EXPECT_EQ(vit.imageSize(), 256u);
  EXPECT_EQ(vit.patchSize(), 16u);
  EXPECT_EQ(vit.numChannels(), 3u);
  EXPECT_EQ(vit.numPatches(), 256u); // (256/16)^2
  EXPECT_EQ(vit.hiddenDim(), 768);
  EXPECT_EQ(vit.numHeads(), 12);
  EXPECT_EQ(vit.headDim(), 64); // 768 / 12
  EXPECT_EQ(vit.numLayers(), 12);
  EXPECT_EQ(vit.intermediateSize(), 3072);

  EXPECT_FALSE(vit.isCausal());
  EXPECT_EQ(vit.ropeTheta(), 0u);
  EXPECT_EQ(vit.initSeqLen(), 256u);
}

TEST(Lfm2VlVisionTransformer, initialize_builds_graph_without_exception) {
  // This is the structural smoke test: any mismatch between layer-name
  // uniqueness, shape inference (Conv2D -> Reshape -> mha_core ->
  // residual addition) or property parsing will surface here as a thrown
  // exception inside compile().
  auto toy = makeToyConfig();
  TestableLfm2Vl vit(toy.cfg, toy.generation_cfg, toy.nntr_cfg);

  ASSERT_NO_THROW(vit.initialize());
  EXPECT_TRUE(vit.initialized());
}

TEST(Lfm2VlVisionTransformer, initialize_supports_larger_grid) {
  // A slightly larger grid (4x4 patches, 4 heads, 3 layers) to catch
  // regressions that only manifest at non-trivial NUM_HEADS / NUM_LAYERS.
  causallm::json cfg = {{"hidden_size", 32},      {"num_attention_heads", 4},
                        {"num_hidden_layers", 3}, {"intermediate_size", 64},
                        {"image_size", 64},       {"patch_size", 16},
                        {"num_channels", 3},      {"layer_norm_eps", 1e-6}};
  causallm::json generation_cfg = causallm::json::object();
  causallm::json nntr_cfg = makeFullNntrCfg();
  nntr_cfg["init_seq_len"] = 16;
  nntr_cfg["max_seq_len"] = 16;

  TestableLfm2Vl vit(cfg, generation_cfg, nntr_cfg);
  EXPECT_EQ(vit.numPatches(), 16u); // (64/16)^2

  ASSERT_NO_THROW(vit.initialize());
  EXPECT_TRUE(vit.initialized());
}

// --- NaFlex interpolation tests ---

TEST(Lfm2VlVisionTransformer, naflex_interp_identity_2x2) {
  // Interpolating a 2x2 grid to itself must return identical values.
  std::vector<float> src = {1.f, 2.f,  3.f,  4.f,  5.f,  6.f,  7.f,  8.f,
                            9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f, 16.f};
  // 4 patches (2x2 grid), dim=4
  auto out = interpPosEmbed(src.data(), 2, 2, 2, 2, 4);
  ASSERT_EQ(out.size(), src.size());
  for (size_t i = 0; i < src.size(); ++i)
    EXPECT_FLOAT_EQ(out[i], src[i]);
}

TEST(Lfm2VlVisionTransformer, naflex_interp_upscale_1x1_to_2x2) {
  // 1x1 grid (single embedding vector [1.0, 2.0]) -> 2x2 grid.
  // Expected: all 4 outputs equal the single source vector.
  std::vector<float> src = {1.f, 2.f};
  auto out = interpPosEmbed(src.data(), 1, 1, 2, 2, 2);
  ASSERT_EQ(out.size(), 8u);
  for (size_t i = 0; i < out.size(); ++i)
    EXPECT_FLOAT_EQ(out[i], src[i % 2]);
}

TEST(Lfm2VlVisionTransformer, naflex_interp_upscale_1x1_to_3x3) {
  // 1x1 -> 3x3: align_corners=False, all outputs equal the single source.
  std::vector<float> src = {3.f, 7.f};
  auto out = interpPosEmbed(src.data(), 1, 1, 3, 3, 2);
  ASSERT_EQ(out.size(), 18u);
  for (size_t i = 0; i < out.size(); ++i)
    EXPECT_FLOAT_EQ(out[i], src[i % 2]);
}

TEST(Lfm2VlVisionTransformer, naflex_interp_upscale_2x2_to_4x4) {
  std::vector<float> src = {1.f, 3.f, 5.f, 7.f};
  std::vector<float> expected = {1.f, 1.5f, 2.5f, 3.f, 2.f, 2.5f, 3.5f, 4.f,
                                 4.f, 4.5f, 5.5f, 6.f, 5.f, 5.5f, 6.5f, 7.f};

  auto out = interpPosEmbed(src.data(), 2, 2, 4, 4, 1);

  ASSERT_EQ(out.size(), expected.size());
  for (size_t i = 0; i < out.size(); ++i)
    EXPECT_FLOAT_EQ(out[i], expected[i]);
}

TEST(Lfm2VlVisionTransformer, naflex_mode_patch_counts_nonsquare) {
  // image_height=32, image_width=64, patch_size=16 -> PATCH_H=2, PATCH_W=4
  causallm::json cfg = {{"hidden_size", 16},      {"num_attention_heads", 2},
                        {"num_hidden_layers", 1}, {"intermediate_size", 32},
                        {"image_size", 32},       {"image_height", 32},
                        {"image_width", 64},      {"patch_size", 16},
                        {"num_channels", 3},      {"layer_norm_eps", 1e-6},
                        {"naflex_mode", true},    {"naflex_base_grid", 16}};
  causallm::json gen_cfg = causallm::json::object();
  causallm::json nntr_cfg = {{"batch_size", 1},
                             {"model_type", "embedding"},
                             {"model_tensor_type", "FP32-FP32"},
                             {"embedding_dtype", "FP32"},
                             {"fc_layer_dtype", "FP32"},
                             {"init_seq_len", 8},
                             {"max_seq_len", 8},
                             {"num_to_generate", 0},
                             {"skip_tokenizer", true}};

  TestableLfm2Vl vit(cfg, gen_cfg, nntr_cfg);
  EXPECT_EQ(vit.patchH(), 2u);
  EXPECT_EQ(vit.patchW(), 4u);
  EXPECT_EQ(vit.numPatches(), 8u);

  // Graph must compile at non-square resolution.
  ASSERT_NO_THROW(vit.initialize());
  EXPECT_TRUE(vit.initialized());
}

TEST(Lfm2VlVisionTransformer, setup_parameters_lfm2vl_450m_vision_config) {
  // Verify that the LFM2-VL-450M vision_config parameters are accepted.
  causallm::json cfg = {{"hidden_size", 768},      {"num_attention_heads", 16},
                        {"num_hidden_layers", 27}, {"intermediate_size", 3072},
                        {"image_size", 256},       {"patch_size", 16},
                        {"num_channels", 3},       {"layer_norm_eps", 1e-6},
                        {"naflex_mode", true},     {"naflex_base_grid", 16}};
  causallm::json gen_cfg = causallm::json::object();
  causallm::json nntr_cfg = {{"batch_size", 1},
                             {"model_type", "embedding"},
                             {"model_tensor_type", "FP32-FP32"},
                             {"embedding_dtype", "FP32"},
                             {"fc_layer_dtype", "FP32"},
                             {"init_seq_len", 256},
                             {"max_seq_len", 256},
                             {"num_to_generate", 0},
                             {"skip_tokenizer", true}};

  TestableLfm2Vl vit(cfg, gen_cfg, nntr_cfg);
  EXPECT_EQ(vit.hiddenDim(), 768);
  EXPECT_EQ(vit.numHeads(), 16);
  EXPECT_EQ(vit.numLayers(), 27);
  EXPECT_EQ(vit.numPatches(), 256u);
  EXPECT_EQ(vit.intermediateSize(), 3072);
}

int main(int argc, char **argv) {
  int result = -1;

  try {
    ::testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error during InitGoogleTest" << std::endl;
    return 0;
  }

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS" << std::endl;
  }

  return result;
}

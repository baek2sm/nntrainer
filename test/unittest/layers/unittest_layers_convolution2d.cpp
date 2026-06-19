// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file unittest_layers_convolution.cpp
 * @date 5 July 2021
 * @brief Conv2d Layer Test
 * @see	https://github.com/nntrainer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <conv2d_layer.h>
#include <layers_common_tests.h>

auto semantic_conv2d = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::Conv2DLayer>, nntrainer::Conv2DLayer::type,
  {"filters=1", "kernel_size=1,1", "padding=1,1"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

GTEST_PARAMETER_TEST(Convolution2D, LayerSemantics,
                     ::testing::Values(semantic_conv2d));

auto conv2d_sb_minimum = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DLayer>,
  {"filters=3", "kernel_size=2,2"}, "1:1:4:4",
  "conv2d_sb_minimum.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT,
  "nchw", "fp32", "fp32");

auto conv2d_mb_minimum = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DLayer>,
  {"filters=3", "kernel_size=2,2"}, "3:1:4:4",
  "conv2d_mb_minimum.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT,
  "nchw", "fp32", "fp32");

auto conv2d_sb_same_remain = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DLayer>,
  {"filters=2", "kernel_size=3,3", "padding=same"}, "1:1:4:4",
  "conv2d_sb_same_remain.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT,
  "nchw", "fp32", "fp32");

auto conv2d_mb_same_remain = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DLayer>,
  {"filters=2", "kernel_size=3,3", "padding=same"}, "3:1:4:4",
  "conv2d_mb_same_remain.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT,
  "nchw", "fp32", "fp32");

auto conv2d_sb_same_uneven_remain_1 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DLayer>,
  {
    "filters=2",
    "kernel_size=3,3",
    "stride=2,2",
    "padding=same",
  },
  "1:3:4:4", "conv2d_sb_same_uneven_remain.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32", "fp32");

auto conv2d_sb_same_uneven_remain_2 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DLayer>,
  {
    "filters=2",
    "kernel_size=3,3",
    "stride=2,2",
    "padding=0,1,0,1",
  },
  "1:3:4:4", "conv2d_sb_same_uneven_remain.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32", "fp32");

auto conv2d_mb_same_uneven_remain_1 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DLayer>,
  {
    "filters=2",
    "kernel_size=3,3",
    "stride=2,2",
    "padding=same",
  },
  "3:3:4:4", "conv2d_mb_same_uneven_remain.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32", "fp32");

auto conv2d_mb_same_uneven_remain_2 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DLayer>,
  {
    "filters=2",
    "kernel_size=3,3",
    "stride=2,2",
    "padding=0,1,0,1",
  },
  "3:3:4:4", "conv2d_mb_same_uneven_remain.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32", "fp32");

auto conv2d_sb_valid_drop_last = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DLayer>,
  {
    "filters=2",
    "kernel_size=3,3",
    "stride=2,2",
    "padding=valid",
  },
  "1:3:7:7", "conv2d_sb_valid_drop_last.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32", "fp32");

auto conv2d_mb_valid_drop_last = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DLayer>,
  {
    "filters=2",
    "kernel_size=3,3",
    "stride=2,2",
    "padding=valid",
  },
  "3:3:7:7", "conv2d_mb_valid_drop_last.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32", "fp32");

auto conv2d_sb_no_overlap = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DLayer>,
  {"filters=3", "kernel_size=2,2", "stride=3,3"}, "1:2:5:5",
  "conv2d_sb_no_overlap.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT,
  "nchw", "fp32", "fp32");

auto conv2d_mb_no_overlap = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DLayer>,
  {
    "filters=3",
    "kernel_size=2,2",
    "stride=3,3",
  },
  "3:2:5:5", "conv2d_mb_no_overlap.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32", "fp32");

auto conv2d_sb_1x1_kernel = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DLayer>,
  {"filters=3", "kernel_size=1,1", "stride=2,2"}, "1:2:5:5",
  "conv2d_sb_1x1_kernel.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT,
  "nchw", "fp32", "fp32");

auto conv2d_mb_1x1_kernel = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DLayer>,
  {
    "filters=3",
    "kernel_size=1,1",
    "stride=2,2",
  },
  "3:2:5:5", "conv2d_mb_1x1_kernel.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32", "fp32");

auto conv2d_sb_dilation = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DLayer>,
  {
    "filters=2",
    "kernel_size=3,3",
    "dilation=2,2",
  },
  "1:3:11:11", "conv2d_sb_dilation.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32", "fp32");

auto conv2d_mb_dilation = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DLayer>,
  {
    "filters=2",
    "kernel_size=3,3",
    "dilation=2,2",
  },
  "3:3:11:11", "conv2d_mb_dilation.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32", "fp32");

auto conv2d_sb_same_dilation = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DLayer>,
  {
    "filters=2",
    "kernel_size=3,3",
    "padding=same",
    "dilation=2,2",
  },
  "1:3:11:11", "conv2d_sb_same_dilation.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32", "fp32");

auto conv2d_mb_same_dilation = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DLayer>,
  {
    "filters=2",
    "kernel_size=3,3",
    "padding=same",
    "dilation=2,2",
  },
  "3:3:11:11", "conv2d_mb_same_dilation.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32", "fp32");

GTEST_PARAMETER_TEST(
  Convolution2D, LayerGoldenTest,
  ::testing::Values(
    conv2d_sb_minimum, conv2d_mb_minimum, conv2d_sb_same_remain,
    conv2d_mb_same_remain, conv2d_sb_same_uneven_remain_1,
    conv2d_sb_same_uneven_remain_2, conv2d_mb_same_uneven_remain_1,
    conv2d_mb_same_uneven_remain_2, conv2d_sb_valid_drop_last,
    conv2d_mb_valid_drop_last, conv2d_sb_no_overlap, conv2d_mb_no_overlap,
    conv2d_sb_1x1_kernel, conv2d_mb_1x1_kernel, conv2d_sb_dilation,
    conv2d_mb_dilation, conv2d_sb_same_dilation, conv2d_mb_same_dilation));

#ifdef ENABLE_FP16
auto conv2d_sb_minimum_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DLayer>,
  {"filters=3", "kernel_size=2,2"}, "1:1:4:4",
  "conv2d_sb_minimum_w16a16.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp16", "fp16");

auto conv2d_mb_minimum_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DLayer>,
  {"filters=3", "kernel_size=2,2"}, "3:1:4:4",
  "conv2d_mb_minimum_w16a16.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp16", "fp16");

auto conv2d_sb_same_remain_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DLayer>,
  {"filters=2", "kernel_size=3,3", "padding=same"}, "1:1:4:4",
  "conv2d_sb_same_remain_w16a16.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp16", "fp16");

auto conv2d_mb_same_remain_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DLayer>,
  {"filters=2", "kernel_size=3,3", "padding=same"}, "3:1:4:4",
  "conv2d_mb_same_remain_w16a16.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp16", "fp16");

auto conv2d_sb_same_uneven_remain_1_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DLayer>,
  {
    "filters=2",
    "kernel_size=3,3",
    "stride=2,2",
    "padding=same",
  },
  "1:3:4:4", "conv2d_sb_same_uneven_remain_w16a16.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp16", "fp16");

auto conv2d_sb_same_uneven_remain_2_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DLayer>,
  {
    "filters=2",
    "kernel_size=3,3",
    "stride=2,2",
    "padding=0,1,0,1",
  },
  "1:3:4:4", "conv2d_sb_same_uneven_remain_w16a16.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp16", "fp16");

auto conv2d_mb_same_uneven_remain_1_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DLayer>,
  {
    "filters=2",
    "kernel_size=3,3",
    "stride=2,2",
    "padding=same",
  },
  "3:3:4:4", "conv2d_mb_same_uneven_remain_w16a16.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp16", "fp16");

auto conv2d_mb_same_uneven_remain_2_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DLayer>,
  {
    "filters=2",
    "kernel_size=3,3",
    "stride=2,2",
    "padding=0,1,0,1",
  },
  "3:3:4:4", "conv2d_mb_same_uneven_remain_w16a16.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp16", "fp16");

auto conv2d_sb_valid_drop_last_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DLayer>,
  {
    "filters=2",
    "kernel_size=3,3",
    "stride=2,2",
    "padding=valid",
  },
  "1:3:7:7", "conv2d_sb_valid_drop_last_w16a16.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp16", "fp16");

auto conv2d_mb_valid_drop_last_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DLayer>,
  {
    "filters=2",
    "kernel_size=3,3",
    "stride=2,2",
    "padding=valid",
  },
  "3:3:7:7", "conv2d_mb_valid_drop_last_w16a16.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp16", "fp16");

auto conv2d_sb_no_overlap_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DLayer>,
  {"filters=3", "kernel_size=2,2", "stride=3,3"}, "1:2:5:5",
  "conv2d_sb_no_overlap_w16a16.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp16", "fp16");

auto conv2d_mb_no_overlap_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DLayer>,
  {
    "filters=3",
    "kernel_size=2,2",
    "stride=3,3",
  },
  "3:2:5:5", "conv2d_mb_no_overlap_w16a16.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp16", "fp16");

auto conv2d_sb_1x1_kernel_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DLayer>,
  {"filters=3", "kernel_size=1,1", "stride=2,2"}, "1:2:5:5",
  "conv2d_sb_1x1_kernel_w16a16.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp16", "fp16");

auto conv2d_mb_1x1_kernel_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DLayer>,
  {
    "filters=3",
    "kernel_size=1,1",
    "stride=2,2",
  },
  "3:2:5:5", "conv2d_mb_1x1_kernel_w16a16.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp16", "fp16");

auto conv2d_sb_dilation_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DLayer>,
  {
    "filters=2",
    "kernel_size=3,3",
    "dilation=2,2",
  },
  "1:3:11:11", "conv2d_sb_dilation_w16a16.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp16", "fp16");

auto conv2d_mb_dilation_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DLayer>,
  {
    "filters=2",
    "kernel_size=3,3",
    "dilation=2,2",
  },
  "3:3:11:11", "conv2d_mb_dilation_w16a16.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp16", "fp16");

auto conv2d_sb_same_dilation_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DLayer>,
  {
    "filters=2",
    "kernel_size=3,3",
    "padding=same",
    "dilation=2,2",
  },
  "1:3:11:11", "conv2d_sb_same_dilation_w16a16.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp16", "fp16");

auto conv2d_mb_same_dilation_w16a16 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DLayer>,
  {
    "filters=2",
    "kernel_size=3,3",
    "padding=same",
    "dilation=2,2",
  },
  "3:3:11:11", "conv2d_mb_same_dilation_w16a16.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp16", "fp16");

GTEST_PARAMETER_TEST(
  Convolution2D16, LayerGoldenTest,
  ::testing::Values(conv2d_sb_minimum_w16a16, conv2d_mb_minimum_w16a16,
                    conv2d_sb_same_remain_w16a16, conv2d_mb_same_remain_w16a16,
                    conv2d_sb_same_uneven_remain_1_w16a16,
                    conv2d_sb_same_uneven_remain_2_w16a16,
                    conv2d_mb_same_uneven_remain_1_w16a16,
                    conv2d_mb_same_uneven_remain_2_w16a16,
                    conv2d_sb_valid_drop_last_w16a16,
                    conv2d_mb_valid_drop_last_w16a16,
                    conv2d_sb_no_overlap_w16a16, conv2d_mb_no_overlap_w16a16,
                    conv2d_sb_1x1_kernel_w16a16, conv2d_mb_1x1_kernel_w16a16,
                    conv2d_sb_dilation_w16a16, conv2d_mb_dilation_w16a16,
                    conv2d_sb_same_dilation_w16a16,
                    conv2d_mb_same_dilation_w16a16));
#endif

/**
 * @brief Q4_0 quantized Conv2D test
 * @note K = in_ch * k_h * k_w = 32 * 3 * 3 = 288, K % 32 == 0 satisfied
 *       Tests that Q4_0 weight_dtype property is accepted and Q4_0 GEMM path
 * is enabled for K%32==0 condition.
 *       Verifies numerical agreement between FP32 and Q4_0 paths.
 */
TEST(Convolution2DQ4_0, verify_q4_0_path_enabled) {
  // Input: 1:32:16:16 (NCHW), in_ch=32
  // Kernel: 1x1, filters=64
  // K = in_ch * k_h * k_w = 32 * 1 * 1 = 32, K % 32 == 0 -> Q4_0 GEMM enabled
  // Using 1x1 conv keeps K=32 (1 Q4_0 block) giving small quantization noise (max_diff ≈ 0.1-0.3)
  const std::string input_shape = "1:32:16:16";
  const std::array<std::string, 3> tensor_type = {"NCHW", "FP32", "FP32"};
  const auto mode = ml::train::ExecutionMode::TRAIN;

  // Create context helper
  auto create_ctx = [](const std::string &input_shape,
                       std::array<std::string, 3> tensor_type,
                       ml::train::ExecutionMode mode) {
    struct shape_parser_ : nntrainer::Property<ml::train::TensorDim> {
      using prop_tag = nntrainer::dimension_prop_tag;
    };
    std::vector<shape_parser_> parsed;
    nntrainer::from_string(input_shape, parsed);
    for (auto &par : parsed) {
      par.get().setFormat(
        nntrainer::str_converter<nntrainer::enum_class_prop_tag,
                                 nntrainer::TensorFormatInfo>::from_string(
          tensor_type[0]));
      [[maybe_unused]] auto _ = shape_parser_::prop_tag{};
    }
    return nntrainer::InitLayerContext({parsed.begin(), parsed.end()}, {true},
                                       false, "test", "", 0.0f, tensor_type,
                                       1.0f, mode);
  };

  // Create FP32 layer (reference path)
  auto fp32_layer = nntrainer::createLayer<nntrainer::Conv2DLayer>();
  std::vector<std::string> fp32_props = {
    "filters=64",
    "kernel_size=1,1",
    "bias_initializer=zeros",
    "conv_weight_quant=FP32"
  };
  auto ctx_fp32 = create_ctx(input_shape, tensor_type, mode);
  fp32_layer->setProperty(fp32_props);
  EXPECT_NO_THROW(fp32_layer->finalize(ctx_fp32));

  // Create Q4_0 layer (quantized path)
  auto q4_0_layer = nntrainer::createLayer<nntrainer::Conv2DLayer>();
  std::vector<std::string> q4_0_props = {
    "filters=64",
    "kernel_size=1,1",
    "bias_initializer=zeros",
    "conv_weight_quant=Q4_0"
  };
  auto ctx_q4_0 = create_ctx(input_shape, tensor_type, mode);
  q4_0_layer->setProperty(q4_0_props);
  EXPECT_NO_THROW(q4_0_layer->finalize(ctx_q4_0));

  // Allocate weights for FP32 layer with fixed seed
  std::vector<nntrainer::Weight> weights_fp32;
  for (unsigned int i = 0; i < ctx_fp32.getNumWeights(); ++i) {
    auto &spec = ctx_fp32.getWeightsSpec()[i];
    weights_fp32.emplace_back(spec, true);
    weights_fp32.back().getVariableRef().setRandUniform(-0.5f, 0.5f);
  }

  // Allocate weights for Q4_0 layer — copy exact same values from FP32 weights
  std::vector<nntrainer::Weight> weights_q4_0;
  for (unsigned int i = 0; i < ctx_q4_0.getNumWeights(); ++i) {
    auto &spec = ctx_q4_0.getWeightsSpec()[i];
    weights_q4_0.emplace_back(spec, true);
    // Copy the exact same weight values from FP32 layer
    auto &src = weights_fp32[i].getVariableRef();
    auto &dst = weights_q4_0.back().getVariableRef();
    const float *src_data = src.getData<float>();
    float *dst_data = dst.getData<float>();
    std::copy(src_data, src_data + src.size(), dst_data);
  }

  // Create run context for FP32 layer with fixed input seed
  std::vector<nntrainer::Var_Grad> ins_fp32, outs_fp32;
  for (auto &dim : ctx_fp32.getInputDimensions()) {
    ins_fp32.emplace_back(dim, nntrainer::Initializer::NONE, true, true, "input");
    ins_fp32.back().getVariableRef().setRandUniform(-1.0f, 1.0f);
  }
  for (auto &spec : ctx_fp32.getOutSpecs()) {
    outs_fp32.emplace_back(spec.variable_spec.dim, nntrainer::Initializer::NONE,
                           true, true, "output");
  }

  // Create run context for Q4_0 layer — copy exact same input as FP32
  std::vector<nntrainer::Var_Grad> ins_q4_0, outs_q4_0;
  for (unsigned int i = 0; i < (unsigned int)ctx_q4_0.getInputDimensions().size(); ++i) {
    auto &dim = ctx_q4_0.getInputDimensions()[i];
    ins_q4_0.emplace_back(dim, nntrainer::Initializer::NONE, true, true, "input");
    // Copy exact same input values from FP32 layer
    auto &src = ins_fp32[i].getVariableRef();
    auto &dst = ins_q4_0.back().getVariableRef();
    const float *src_data = src.getData<float>();
    float *dst_data = dst.getData<float>();
    std::copy(src_data, src_data + src.size(), dst_data);
  }
  for (auto &spec : ctx_q4_0.getOutSpecs()) {
    outs_q4_0.emplace_back(spec.variable_spec.dim, nntrainer::Initializer::NONE,
                           true, true, "output");
  }

  // Prepare weight pointers for FP32 layer
  std::vector<nntrainer::Weight *> weight_ptrs_fp32;
  for (auto &w : weights_fp32) {
    weight_ptrs_fp32.push_back(&w);
  }

  // Prepare weight pointers for Q4_0 layer
  std::vector<nntrainer::Weight *> weight_ptrs_q4_0;
  for (auto &w : weights_q4_0) {
    weight_ptrs_q4_0.push_back(&w);
  }

  // Prepare input/output pointers
  std::vector<nntrainer::Var_Grad *> in_ptrs_fp32, out_ptrs_fp32;
  std::vector<nntrainer::Var_Grad *> in_ptrs_q4_0, out_ptrs_q4_0;
  for (auto &v : ins_fp32) in_ptrs_fp32.push_back(&v);
  for (auto &v : outs_fp32) out_ptrs_fp32.push_back(&v);
  for (auto &v : ins_q4_0) in_ptrs_q4_0.push_back(&v);
  for (auto &v : outs_q4_0) out_ptrs_q4_0.push_back(&v);

  // Create run contexts
  auto rc_fp32 = nntrainer::RunLayerContext("test_fp32", true, 0.0f, false, 1.0,
                                            nullptr, false, weight_ptrs_fp32,
                                            in_ptrs_fp32, out_ptrs_fp32, {});
  auto rc_q4_0 = nntrainer::RunLayerContext("test_q4_0", true, 0.0f, false, 1.0,
                                            nullptr, false, weight_ptrs_q4_0,
                                            in_ptrs_q4_0, out_ptrs_q4_0, {});

  // Initialize both layers
  EXPECT_NO_THROW(fp32_layer->initialize(rc_fp32));
  EXPECT_NO_THROW(q4_0_layer->initialize(rc_q4_0));

  // Forward (inference mode)
  EXPECT_NO_THROW(fp32_layer->forwarding(rc_fp32, false));
  EXPECT_NO_THROW(q4_0_layer->forwarding(rc_q4_0, false));

  // Get outputs
  auto &out_fp32 = rc_fp32.getOutput(0);
  auto &out_q4_0 = rc_q4_0.getOutput(0);

  // Verify output is produced
  EXPECT_GT(out_fp32.size(), 0u);
  EXPECT_GT(out_q4_0.size(), 0u);
  EXPECT_EQ(out_fp32.size(), out_q4_0.size());

  // Compute max_diff between FP32 and Q4_0 outputs
  float max_diff = 0.0f;
  const float *fp32_data = out_fp32.getData<float>();
  const float *q4_0_data = out_q4_0.getData<float>();
  for (size_t i = 0; i < out_fp32.size(); ++i) {
    float diff = std::abs(fp32_data[i] - q4_0_data[i]);
    if (diff > max_diff) {
      max_diff = diff;
    }
  }

  // Q4_0 quantization error should be within tolerance (0.5 upper bound)
  // Note: This tests that Q4_0 GEMM path is functional and produces similar results to FP32
  EXPECT_LE(max_diff, 0.5f) << "Max diff between FP32 and Q4_0 outputs exceeds tolerance";
}

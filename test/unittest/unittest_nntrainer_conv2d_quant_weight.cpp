// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_nntrainer_conv2d_quant_weight.cpp
 * @date   02 September 2026
 * @brief  Contract tests for the quantized-conv2d weight guard.
 * @author Seungbaek <sb92.hong@samsung.com>
 *
 *         Conv2DLayer requests its weight as (filter, in_ch, kh, kw), so the
 *         axis a per-channel scale vector keys on is a spatial/kernel axis, not
 *         the output channel:
 *           - QINT8  : scale_size() == width()  (== kw)
 *           - QINT16 : scale_size() == height() (== kh)
 *           - QINT4  : scale_size() == height() * width() / group_size (32),
 *                      i.e. 0 for any real kernel (fewer than 32 taps)
 *         None equals the output channel count. forwarding() also never drives
 *         an int8 kernel (it only issues an FP dot, which throws on a quantized
 *         tensor). finalize() therefore rejects every quantized weight dtype;
 *         FP32/FP16 are the only accepted weights.
 *
 *         The channel-last target layout (weight (1, kh, kw, out_ch), so that
 *         QINT8 scale_size() == width() == out_ch) is pinned by the scale_size
 *         tests below; enabling it end-to-end is a follow-up PR.
 * @see    https://github.com/nntrainer/nntrainer
 * @bug    No known bugs except for NYI items
 */

#include <gtest/gtest.h>

#include <input_layer.h>
#include <layer.h>
#include <neuralnet.h>
#include <optimizer.h>
#include <tensor.h>
#include <tensor_dim.h>

using TensorDim = ml::train::TensorDim;
using DataType = TensorDim::DataType;
using Format = TensorDim::Format;

namespace {

/**
 * @brief Map a weight dtype to the model "model_tensor_type" string (WxAy).
 * @note  Only FP32 and the quantized weight types exercised here are mapped;
 *        any other type is a test authoring bug and aborts.
 */
std::string modelTensorType(DataType weight_type) {
  switch (weight_type) {
  case DataType::FP32:
    return "FP32-FP32";
  case DataType::QINT8:
    return "QINT8-FP32";
  case DataType::QINT4:
    return "QINT4-FP32";
  case DataType::QINT16:
    return "QINT16-QINT16";
  case DataType::FP16:
    return "FP16-FP32";
  default:
    ADD_FAILURE() << "unmapped weight dtype in test helper";
    return "FP32-FP32";
  }
}

/**
 * @brief Build an NN with one conv layer (conv2d, or conv1d which delegates
 * its finalize/forwarding to an inner Conv2DLayer), set the weight data type
 * and format, and initialize. On throw, capture the exception message and
 * return -1; on success return 0.
 */
int tryInitializeConvNN(Format format, DataType weight_type,
                        const char *layer_type = "conv2d",
                        std::string *message = nullptr) {
  auto nn = std::make_unique<nntrainer::NeuralNetwork>();

  const bool is_1d = std::string(layer_type) == "conv1d";
  // conv1d takes a (channel, height, width) input and a scalar kernel.
  nn->addLayer(ml::train::layer::Input(
    {"name=input",
     "input_shape=" + std::string(is_1d ? "3:1:16" : "1:3:16:16")}));
  nn->addLayer(std::shared_ptr<ml::train::Layer>(ml::train::createLayer(
    layer_type, {std::string("name=conv") + layer_type, "filters=8",
                 std::string("kernel_size=") + (is_1d ? "3" : "3,3"),
                 std::string("stride=") + (is_1d ? "1" : "1,1"),
                 std::string("padding=") + (is_1d ? "0" : "0,0")})));

  const char *fmt = format == Format::NHWC ? "NHWC" : "NCHW";
  nn->setProperty({std::string("batch_size=1"),
                   std::string("tensor_format=") + fmt,
                   "model_tensor_type=" + modelTensorType(weight_type)});
  nn->setOptimizer(ml::train::optimizer::SGD({"learning_rate=0.1"}));
  nn->setProperty({"loss=mse"});

  try {
    nn->compile();
    nn->initialize();
  } catch (const std::exception &e) {
    if (message != nullptr) {
      *message = e.what();
    }
    return -1;
  }
  return 0;
}

} // namespace

/**
 * @brief Target contract: a per-channel QINT8 weight laid out channel-last
 * (1, kh, kw, out_ch) has exactly one scale per output channel, so
 * scale_size() == width() == out_ch. This is what a future channel-last conv2d
 * weight request will rely on.
 */
TEST(ConvQuantWeightGuard, channelLastQint8ScaleSizeEqualsOutChannels) {
  const unsigned int kh = 3, kw = 3, out_ch = 8;
  nntrainer::Tensor weight(
    TensorDim(1, kh, kw, out_ch,
              TensorDim::TensorType(Format::NHWC, DataType::QINT8)),
    true, nntrainer::Initializer::NONE, "w",
    nntrainer::QScheme::PER_CHANNEL_AFFINE);

  EXPECT_EQ(weight.q_scheme(), nntrainer::QScheme::PER_CHANNEL_AFFINE);
  EXPECT_EQ(weight.width(), out_ch);
  EXPECT_EQ(weight.scale_size(), out_ch);
}

/**
 * @brief The per-tensor scheme (the activation side of the scheme) is a single
 * scale regardless of shape.
 */
TEST(ConvQuantWeightGuard, perTensorSchemeHasSingleScale) {
  nntrainer::Tensor act(
    TensorDim(1, 4, 8, 8, TensorDim::TensorType(Format::NHWC, DataType::QINT8)),
    true, nntrainer::Initializer::NONE, "act",
    nntrainer::QScheme::PER_TENSOR_AFFINE);

  EXPECT_EQ(act.q_scheme(), nntrainer::QScheme::PER_TENSOR_AFFINE);
  EXPECT_EQ(act.scale_size(), 1u);
}

/**
 * @brief Proof that the current (filter, in_ch, kh, kw) QINT8 weight layout
 * mis-sizes the scale vector: for a 3x3 filter it yields kw scales (3), not the
 * output channel count (8). This is why the guard exists.
 */
TEST(ConvQuantWeightGuard, qint8WidthLayoutKeysOnKernelWidthNotOutChannels) {
  nntrainer::Tensor weight(
    TensorDim(8, 3, 3, 3, // (filter=8, in_ch=3, kh=3, kw=3)
              TensorDim::TensorType(Format::NCHW, DataType::QINT8)),
    true, nntrainer::Initializer::NONE, "w",
    nntrainer::QScheme::PER_CHANNEL_AFFINE);

  EXPECT_EQ(weight.scale_size(), 3u); // == kw, NOT the 8 output channels
}

/**
 * @brief QINT16 per-channel scales key on height() (== kh), equally not the
 * output channel count, so it is rejected too.
 */
TEST(ConvQuantWeightGuard, qint16ScaleSizeKeysOnHeight) {
  nntrainer::Tensor weight(
    TensorDim(8, 3, 3, 3,
              TensorDim::TensorType(Format::NCHW, DataType::QINT16)),
    true, nntrainer::Initializer::NONE, "w",
    nntrainer::QScheme::PER_CHANNEL_AFFINE);

  EXPECT_EQ(weight.scale_size(), 3u); // == kh, NOT the 8 output channels
}

/**
 * @brief QINT4 per-channel scales are group-based over h*w/32, not one per
 * output channel, so it cannot satisfy the channel-last contract either.
 */
TEST(ConvQuantWeightGuard, qint4ScaleSizeIsGroupBased) {
  // (filter=8, in_ch=4, kh=8, kw=4): scale_size() == height()*width()/32
  // == 8*4/32 == 1, which is neither the output channel count (8) nor any
  // per-channel layout.
  nntrainer::Tensor weight(
    TensorDim(8, 4, 8, 4, TensorDim::TensorType(Format::NCHW, DataType::QINT4)),
    true, nntrainer::Initializer::NONE, "w",
    nntrainer::QScheme::PER_CHANNEL_AFFINE);

  EXPECT_EQ(weight.scale_size(), 1u); // h*w/32, not out_ch
}

/**
 * @brief A real conv kernel (3x3 = 9 taps) has fewer than the 32 taps a QINT4
 * group spans, so the group-based scale_size() truncates to zero — the scale
 * vector would not even exist. This pins the shape the guard actually rejects.
 */
TEST(ConvQuantWeightGuard, qint4ScaleSizeIsZeroForRealKernelShapes) {
  nntrainer::Tensor weight(
    TensorDim(8, 3, 3, 3, // (filter=8, in_ch=3, kh=3, kw=3) -> 9 taps < 32
              TensorDim::TensorType(Format::NCHW, DataType::QINT4)),
    true, nntrainer::Initializer::NONE, "w",
    nntrainer::QScheme::PER_CHANNEL_AFFINE);

  EXPECT_EQ(weight.scale_size(), 0u); // 9/32 == 0
}

/**
 * @brief finalize() rejects every quantized conv2d weight dtype, in both
 * model formats, rather than build a mis-sized per-channel scale vector.
 * Conv1DLayer delegates finalize() to an inner Conv2DLayer, so it is covered
 * by the same guard and checked here too.
 */
TEST(ConvQuantWeightGuard, quantizedConvWeightsRejected) {
  for (const char *layer : {"conv2d", "conv1d"}) {
    for (Format fmt : {Format::NCHW, Format::NHWC}) {
      for (DataType dt : {DataType::QINT8, DataType::QINT4, DataType::QINT16}) {
        std::string msg;
        EXPECT_EQ(tryInitializeConvNN(fmt, dt, layer, &msg), -1)
          << "expected rejection for " << layer << " dtype index " << (int)dt
          << " fmt " << (fmt == Format::NHWC ? "NHWC" : "NCHW");
        // Pin that the throw comes from the conv2d quantized-weight guard, not
        // an unrelated early failure during compile/initialize.
        EXPECT_NE(msg.find("quantized conv2d weights are not supported"),
                  std::string::npos)
          << "unexpected rejection message: " << msg;
      }
    }
  }
}

/**
 * @brief FP32 conv2d weights initialize fine in both formats — the guard only
 * rejects quantized weights and does not disturb the normal path.
 */
TEST(ConvQuantWeightGuard, fp32Conv2dWeightStillInitializes) {
  EXPECT_EQ(tryInitializeConvNN(Format::NCHW, DataType::FP32), 0);
  EXPECT_EQ(tryInitializeConvNN(Format::NHWC, DataType::FP32), 0);
}

#ifdef ENABLE_FP16
/**
 * @brief FP16 weights are accepted by the guard (the other half of the
 * accepted set), in both formats.
 */
TEST(ConvQuantWeightGuard, fp16Conv2dWeightStillInitializes) {
  EXPECT_EQ(tryInitializeConvNN(Format::NCHW, DataType::FP16), 0);
  EXPECT_EQ(tryInitializeConvNN(Format::NHWC, DataType::FP16), 0);
}
#endif // ENABLE_FP16

int main(int argc, char **argv) {
  int result = -1;

  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cout << "Error, InitGoogleTest Failed" << std::endl;
    return -1;
  }
  result = RUN_ALL_TESTS();

  return result;
}

// SPDX-License-Identifier: Apache-2.0
/**
 * @file   unittest_qint8_channelwise_scale.cpp
 * @date   02 September 2026
 * @brief  Contract tests for the quantized-conv2d weight guard.
 *
 *         Conv2DLayer requests its weight as (filter, in_ch, kh, kw), so the
 *         axis a per-channel scale vector keys on is a spatial/kernel axis, not
 *         the output channel:
 *           - QINT8  : scale_size() == width()  (== kw)
 *           - QINT16 : scale_size() == height() (== kh)
 *           - QINT4  : scale_size() == height() * width() / group_size (32)
 *         None equals the output channel count, so a per-channel scale vector
 *         would be silently mis-sized. forwarding() also never drives an int8
 *         kernel (it only issues an FP dot). finalize() therefore rejects every
 *         quantized weight dtype; FP32/FP16 are the only accepted weights.
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
  default:
    ADD_FAILURE() << "unmapped weight dtype in test helper";
    return "FP32-FP32";
  }
}

/**
 * @brief Build an NN with one conv2d, set the weight data type and format, and
 * run compile+initialize.
 * @return 0 on success, -1 if compile/initialize threw.
 */
/**
 * @brief initialize() an NN with one conv2d; on throw, return the exception
 * message and -1; on success return 0 with an empty message.
 */
int tryInitializeConvNN(Format format, DataType weight_type,
                        std::string *message = nullptr) {
  auto nn = std::make_unique<nntrainer::NeuralNetwork>();

  nn->addLayer(
    ml::train::layer::Input({"name=input", "input_shape=1:3:16:16"}));
  nn->addLayer(ml::train::layer::Convolution2D(
    {"filters=8", "kernel_size=3,3", "stride=1,1", "padding=0,0"}));

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
TEST(QInt8ChannelwiseScale, channelLastQint8ScaleSizeEqualsOutChannels) {
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
TEST(QInt8ChannelwiseScale, perTensorSchemeHasSingleScale) {
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
TEST(QInt8ChannelwiseScale, qint8WidthLayoutKeysOnKernelWidthNotOutChannels) {
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
TEST(QInt8ChannelwiseScale, qint16ScaleSizeKeysOnHeight) {
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
TEST(QInt8ChannelwiseScale, qint4ScaleSizeIsGroupBased) {
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
 * @brief finalize() rejects every quantized conv2d weight dtype, in both
 * model formats, rather than build a mis-sized per-channel scale vector.
 */
TEST(QInt8ChannelwiseScale, quantizedConv2dWeightsRejected) {
  for (Format fmt : {Format::NCHW, Format::NHWC}) {
    for (DataType dt : {DataType::QINT8, DataType::QINT4, DataType::QINT16}) {
      std::string msg;
      EXPECT_EQ(tryInitializeConvNN(fmt, dt, &msg), -1)
        << "expected rejection for dtype index " << (int)dt << " fmt "
        << (fmt == Format::NHWC ? "NHWC" : "NCHW");
      // Pin that the throw comes from the conv2d quantized-weight guard, not an
      // unrelated early failure during compile/initialize.
      EXPECT_NE(msg.find("quantized conv2d weights are not supported"),
                std::string::npos)
        << "unexpected rejection message: " << msg;
    }
  }
}

/**
 * @brief FP32 conv2d weights initialize fine in both formats — the guard only
 * rejects quantized weights and does not disturb the normal path.
 */
TEST(QInt8ChannelwiseScale, fp32Conv2dWeightStillInitializes) {
  EXPECT_EQ(tryInitializeConvNN(Format::NCHW, DataType::FP32), 0);
  EXPECT_EQ(tryInitializeConvNN(Format::NHWC, DataType::FP32), 0);
}

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

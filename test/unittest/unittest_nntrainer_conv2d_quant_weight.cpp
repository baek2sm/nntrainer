// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_nntrainer_conv2d_quant_weight.cpp
 * @date   02 September 2026
 * @brief  Contract tests for the quantized-conv2d weight.
 * @author Seungbaek <sb92.hong@samsung.com>
 *
 *         A per-channel scale vector has to be keyed on the output channel, and
 *         for a QINT8 tensor that fixes the shape the weight is requested in:
 *         scale_size() is width(), so the weight is requested (kh, in_ch, kw,
 *         filter_size) -- the channel last spelling whose width is the output
 *         channel count. Any other layout keys the scales on a kernel axis:
 *           - requested channel first (filter, in_ch, kh, kw), QINT8:
 *             scale_size() == width() (== kw)
 *           - QINT16 : scale_size() == height() (== kh)
 *           - QINT4  : scale_size() == height() * width() / group_size (32),
 *                      i.e. 0 for kernels up to 5x5 (< 32 taps), as used by
 *                      every conv2d in the tracked .ini configs
 *         finalize() rejects the weight dtypes that cannot satisfy the
 * contract, and rejects a QINT8 weight for a model that is not inference only
 * -- an int8 weight has no fp32 mirror for an optimizer to update. So the
 * accepted set is FP32, FP16 and an inference only channel last QINT8.
 *
 *         Storing the weight and computing with it are separate steps, and only
 * the first is in place: one int8 per tap plus one fp32 scale per output
 * channel instead of one fp32 per tap. forwarding() refuses a QINT8 weight
 * rather than computing it, because no path there handles the channel last
 * layout the weight has to have, so the dequantization that stands in for the
 * int8 kernel is checked directly, tap by tap, against hand derived values.
 * What is pinned about persistence is the saved block -- its size, its byte
 * layout and the stream level round trip -- plus, at the model level, the two
 * things that are true today: Model::load() does not restore the scales (the
 * read path it uses by default reads a quantized block as taps only), and the
 * layer says so at the forward pass instead of running on zero scales.
 * @see    https://github.com/nntrainer/nntrainer
 * @bug    No known bugs except for NYI items
 */

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <input_layer.h>
#include <layer.h>
#include <model.h>
#include <neuralnet.h>
#include <optimizer.h>
#include <tensor.h>
#include <tensor_dim.h>
#include <util_func.h>

using TensorDim = ml::train::TensorDim;
using DataType = TensorDim::DataType;
using Format = TensorDim::Format;
using ModelFormat = ml::train::ModelFormat;

namespace nntrainer {
namespace detail {
/**
 * @brief Internals of Conv2DLayer, declared here rather than in a header so
 * that no public header advertises them. The definitions live in
 * nntrainer/layers/conv2d_layer.cpp and are exported so that this file can link
 * against them; a change to either signature breaks the link of this test,
 * which is what tells the reader the pinned mapping moved.
 */
TensorDim channelLastKernelDim(unsigned int k_height, unsigned int in_ch,
                               unsigned int k_width, unsigned int filter_size,
                               const TensorDim::TensorType &t_type);
Tensor dequantizeKernel(const Tensor &weight, TensorDim::DataType dtype);
} // namespace detail
} // namespace nntrainer

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
    // FP16-FP16 (not FP16-FP32): the guard is indifferent to the activation
    // dtype, and W16A16 is the combination the suite already exercises in CI.
    return "FP16-FP16";
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
 * @brief The layout the conv2d weight is requested in: a per-channel QINT8
 * weight laid out channel-last has exactly one scale per output channel, so
 * scale_size() == width() == out_ch. This is what Conv2DLayer::finalize()
 * relies on, and the shape the persistence tests below save and restore.
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
 * @brief finalize() rejects every quantized conv2d weight that would key its
 * per-channel scale vector on something other than the output channel, in both
 * model formats. Conv1DLayer delegates finalize() to an inner Conv2DLayer, so
 * it is covered by the same guard and checked here too.
 *
 * QINT8 on a channel last model is the one quantized weight the scale-axis
 * guard admits; it is still refused here because these models are built to
 * train, and a quantized weight has no fp32 mirror to optimize. That is a
 * different rejection, so it is pinned separately below.
 */
TEST(ConvQuantWeightGuard, quantizedConvWeightsRejected) {
  for (const char *layer : {"conv2d", "conv1d"}) {
    for (Format fmt : {Format::NCHW, Format::NHWC}) {
      for (DataType dt : {DataType::QINT8, DataType::QINT4, DataType::QINT16}) {
        // channel last QINT8 is the case the scale-axis guard now admits, so it
        // is refused for the other reason and checked separately
        if (fmt == Format::NHWC && dt == DataType::QINT8) {
          continue;
        }
        std::string msg;
        EXPECT_EQ(tryInitializeConvNN(fmt, dt, layer, &msg), -1)
          << "expected rejection for " << layer << " dtype index " << (int)dt
          << " fmt " << (fmt == Format::NHWC ? "NHWC" : "NCHW");
        // Pin that the throw comes from the conv2d quantized-weight guard, not
        // an unrelated early failure during compile/initialize.
        EXPECT_NE(msg.find("quantized conv2d weight is supported only as a "
                           "QINT8 weight"),
                  std::string::npos)
          << "unexpected rejection message: " << msg;
      }
    }
  }
}

/**
 * @brief The scale-axis guard admits a QINT8 weight for a channel last model,
 * so a training model refuses it for its own reason: the weight pool would
 * initialize and the optimizer would update int8 storage. Pinned separately
 * from the scale-axis rejection so the two boundaries cannot be confused.
 */
TEST(ConvQuantWeightGuard, channelLastQint8RejectedForTraining) {
  for (const char *layer : {"conv2d", "conv1d"}) {
    std::string msg;
    EXPECT_EQ(tryInitializeConvNN(Format::NHWC, DataType::QINT8, layer, &msg),
              -1)
      << "expected a training model to reject a QINT8 " << layer << " weight";
    EXPECT_NE(msg.find("QINT8 conv2d weight is inference only"),
              std::string::npos)
      << "unexpected rejection message: " << msg;
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

namespace {

/**
 * shape of the weight the dequantization golden below pins: 2x2 kernel, 2 input
 * channels, 3 output channels. Requested the way Conv2DLayer::finalize()
 * requests a quantized weight, so the request axes are (kh, in_ch, kw, filter).
 */
constexpr unsigned int DQ_KH = 2;
constexpr unsigned int DQ_IN_CH = 2;
constexpr unsigned int DQ_KW = 2;
constexpr unsigned int DQ_FILTERS = 3;

/**
 * @brief One scale per output channel, deliberately not a power of two and
 * never 1 or 0, so that a tap that was scaled by the wrong channel's scale, or
 * by none at all, is a different number rather than the same one. Each is an
 * odd multiple of a power of two (3/128, 5/256, 7/512), which makes every
 * expected product below exact in fp32 and lets the comparison be bit for bit.
 */
constexpr float DQ_SCALES[DQ_FILTERS] = {
  0.0234375f,  /* 3/128  */
  0.01953125f, /* 5/256  */
  0.013671875f /* 7/512  */
};

/**
 * @brief The int8 codes of the weight, laid out as the request lays the axes
 * out: [kh][in_ch][kw][filter]. Negative codes, the -128/127 endpoints and
 * small odd magnitudes are all present, so neither a lost sign nor a lost byte
 * of a tap survives.
 */
constexpr int8_t DQ_CODES[DQ_KH][DQ_IN_CH][DQ_KW][DQ_FILTERS] = {
  {
    /* kh = 0 */
    {/* in_ch = 0 */ {17, 63, -128}, /* kw = 0 */
     {127, 7, 11}},                  /* kw = 1 */
    {/* in_ch = 1 */ {-19, -64, 37}, /* kw = 0 */
     {-88, -13, -3}}                 /* kw = 1 */
  },
  {
    /* kh = 1 */
    {/* in_ch = 0 */ {3, -99, 5},  /* kw = 0 */
     {64, 21, -127}},              /* kw = 1 */
    {/* in_ch = 1 */ {-10, 2, -1}, /* kw = 0 */
     {-41, -128, 100}}             /* kw = 1 */
  },
};

/**
 * @brief The operand dequantizeKernel() has to produce, laid out the way the
 * operand is laid out: [filter][in_ch][kh][kw]. Derivation, one entry at a
 * time:
 *
 *   expected[f][c][h][w] = DQ_CODES[h][c][w][f] * DQ_SCALES[f]
 *
 * The dot consumes the classical (filter_size, in_ch, kh, kw) operand while the
 * weight is stored (kh, in_ch, kw, filter), so the kernel row h and the kernel
 * column w move from the request's batch() and height() to the operand's
 * height() and width(), in_ch stays channel(), and filter moves from width() to
 * batch(). The scale of a tap is the scale of its output channel -- index f,
 * not c and not w -- which is the one thing about this mapping that a wrong
 * choice would hide, since in_ch, kh and kw all range over 2 here and only f
 * ranges over 3.
 *
 * Each entry below is written literally and annotated with the code and the
 * scale it comes from. Every product is an integer over 128, 256 or 512, so it
 * is exact in fp32 and the test can compare it bit for bit. Two are worked out
 * in full to make the index order checkable by hand:
 *   expected[0][0][0][0] = DQ_CODES[0][0][0][0] * DQ_SCALES[0] = 17 * 3/128
 *                        = 51/128 = 0.3984375
 *   expected[2][0][1][1] = DQ_CODES[1][0][1][2] * DQ_SCALES[2] = -127 * 7/512
 *                        = -889/512 = -1.736328125
 *
 * Reading a stored tap as float rather than int8 -- what Tensor::getValue does
 * by default, since its template argument is float and getData<float>() hands
 * back the raw buffer -- reinterprets four int8 codes as one float and yields
 * values of an entirely different magnitude, so every one of the 24 comparisons
 * below fails rather than one.
 */
constexpr float DQ_EXPECTED[DQ_FILTERS][DQ_IN_CH][DQ_KH][DQ_KW] = {
  {
    // output channel 0, scale 3/128
    {
      // input channel 0
      {0.3984375f /* kh0 kw0: 17*3/128 */, 2.9765625f /* kh0 kw1: 127*3/128 */},
      {0.0703125f /* kh1 kw0: 3*3/128 */, 1.5f /* kh1 kw1: 64*3/128 */},
    },
    {
      // input channel 1
      {-0.4453125f /* kh0 kw0: -19*3/128 */, -2.0625f /* kh0 kw1: -88*3/128 */},
      {-0.234375f /* kh1 kw0: -10*3/128 */,
       -0.9609375f /* kh1 kw1: -41*3/128 */},
    },
  },
  {
    // output channel 1, scale 5/256
    {
      // input channel 0
      {1.23046875f /* kh0 kw0: 63*5/256 */, 0.13671875f /* kh0 kw1: 7*5/256 */},
      {-1.93359375f /* kh1 kw0: -99*5/256 */,
       0.41015625f /* kh1 kw1: 21*5/256 */},
    },
    {
      // input channel 1
      {-1.25f /* kh0 kw0: -64*5/256 */, -0.25390625f /* kh0 kw1: -13*5/256 */},
      {0.0390625f /* kh1 kw0: 2*5/256 */, -2.5f /* kh1 kw1: -128*5/256 */},
    },
  },
  {
    // output channel 2, scale 7/512
    {
      // input channel 0
      {-1.75f /* kh0 kw0: -128*7/512 */, 0.150390625f /* kh0 kw1: 11*7/512 */},
      {0.068359375f /* kh1 kw0: 5*7/512 */,
       -1.736328125f /* kh1 kw1: -127*7/512 */},
    },
    {
      // input channel 1
      {0.505859375f /* kh0 kw0: 37*7/512 */,
       -0.041015625f /* kh0 kw1: -3*7/512 */},
      {-0.013671875f /* kh1 kw0: -1*7/512 */,
       1.3671875f /* kh1 kw1: 100*7/512 */},
    },
  },
};

} // namespace

/**
 * @brief The dequantization itself, tap by tap, against the values derived
 * above. It cannot be reached through forwarding(): finalize() admits a QINT8
 * weight only for a channel last model and forwarding() refuses to compute one
 * until the int8 kernel lands. So the helper is checked directly, rather than
 * leaving the arithmetic the whole storage decision exists for untested.
 */
TEST(ConvQuantWeightDequant, dequantizeKernelMatchesHandComputedTaps) {
  TensorDim weight_dim = nntrainer::detail::channelLastKernelDim(
    DQ_KH, DQ_IN_CH, DQ_KW, DQ_FILTERS,
    TensorDim::TensorType(Format::NHWC, DataType::QINT8));
  ASSERT_EQ(weight_dim.batch(), DQ_KH);
  ASSERT_EQ(weight_dim.channel(), DQ_IN_CH);
  ASSERT_EQ(weight_dim.height(), DQ_KW);
  ASSERT_EQ(weight_dim.width(), DQ_FILTERS);

  nntrainer::Tensor weight(weight_dim, true, nntrainer::Initializer::NONE, "w",
                           nntrainer::QScheme::PER_CHANNEL_AFFINE);
  ASSERT_EQ(weight.scale_size(), DQ_FILTERS)
    << "one scale per output channel is what the scale lookup assumes";

  for (unsigned int f = 0; f < DQ_FILTERS; ++f) {
    weight.getScale<float>()[f] = DQ_SCALES[f];
  }
  for (unsigned int h = 0; h < DQ_KH; ++h) {
    for (unsigned int c = 0; c < DQ_IN_CH; ++c) {
      for (unsigned int w = 0; w < DQ_KW; ++w) {
        for (unsigned int f = 0; f < DQ_FILTERS; ++f) {
          /// the weight axes are (kh, in_ch, kw, filter)
          weight.setValue(h, c, w, f, static_cast<float>(DQ_CODES[h][c][w][f]));
        }
      }
    }
  }
  /// the codes really were stored as int8, so the golden compares like with
  /// like
  for (unsigned int h = 0; h < DQ_KH; ++h) {
    for (unsigned int c = 0; c < DQ_IN_CH; ++c) {
      for (unsigned int w = 0; w < DQ_KW; ++w) {
        for (unsigned int f = 0; f < DQ_FILTERS; ++f) {
          ASSERT_EQ(weight.getValue<int8_t>(h, c, w, f), DQ_CODES[h][c][w][f]);
        }
      }
    }
  }

  nntrainer::Tensor kernel =
    nntrainer::detail::dequantizeKernel(weight, DataType::FP32);

  // the classical operand the dot consumes: (filter_size, in_ch, kh, kw)
  ASSERT_EQ(kernel.getDataType(), DataType::FP32);
  ASSERT_EQ(kernel.batch(), DQ_FILTERS);
  ASSERT_EQ(kernel.channel(), DQ_IN_CH);
  ASSERT_EQ(kernel.height(), DQ_KH);
  ASSERT_EQ(kernel.width(), DQ_KW);

  for (unsigned int f = 0; f < DQ_FILTERS; ++f) {
    for (unsigned int c = 0; c < DQ_IN_CH; ++c) {
      for (unsigned int h = 0; h < DQ_KH; ++h) {
        for (unsigned int w = 0; w < DQ_KW; ++w) {
          /// exact, so no tolerance: an integer over a power of two
          EXPECT_EQ(kernel.getValue(f, c, h, w), DQ_EXPECTED[f][c][h][w])
            << "at (filter=" << f << ", in_ch=" << c << ", kh=" << h
            << ", kw=" << w << "), code " << (int)DQ_CODES[h][c][w][f]
            << " times scale " << DQ_SCALES[f];
        }
      }
    }
  }
}

namespace {

/** conv2d properties of the inference model the persistence tests share */
constexpr unsigned int RT_IN_CH = 3;
constexpr unsigned int RT_FILTERS = 5;
constexpr unsigned int RT_KH = 3;
constexpr unsigned int RT_KW = 3;
constexpr unsigned int RT_BATCH = 2;

/**
 * @brief Temporary file of this test suite, removed when the test ends. The
 * save path of a model has to be a real file. The name is relative to the
 * working directory the binary runs in, the way the other test binaries of this
 * directory name their scratch models, so that no absolute path has to exist on
 * the machine -- and no Windows path has to be spelled -- for the test to run.
 */
class TempFile {
public:
  explicit TempFile(const std::string &name) :
    path("unittest_conv2d_qint8_" + name) {}
  ~TempFile() { std::remove(path.c_str()); }
  const std::string &pathfile() const { return path; }

private:
  std::string path;
};

/**
 * @brief Build the inference only channel last QINT8 model the persistence
 * tests compare: an input layer and one conv2d. No optimizer and no loss,
 * because a model carrying a quantized weight is refused for training (see
 * channelLastQint8RejectedForTraining). bias_initializer is set so that the
 * bias is a defined tensor of the file the round trip compares.
 */
std::unique_ptr<nntrainer::NeuralNetwork> buildQint8InferenceNN() {
  auto nn = std::make_unique<nntrainer::NeuralNetwork>();

  nn->addLayer(ml::train::layer::Input({"name=input", "input_shape=1:3:8:8"}));
  nn->addLayer(ml::train::layer::Convolution2D(
    {"name=conv", "filters=" + std::to_string(RT_FILTERS),
     "kernel_size=" + std::to_string(RT_KH) + "," + std::to_string(RT_KW),
     "stride=1,1", "padding=0,0", "bias_initializer=ones"}));

  nn->setProperty({"batch_size=" + std::to_string(RT_BATCH),
                   "tensor_format=NHWC", "model_tensor_type=QINT8-FP32"});

  nn->compile(ml::train::ExecutionMode::INFERENCE);
  nn->initialize(ml::train::ExecutionMode::INFERENCE);
  return nn;
}

/**
 * @brief Fetch the conv2d weight of an initialized model, i.e. the tensor that
 * save() writes and load() restores. The conv2d is the second node of the flat
 * graph; the first is the input layer, which carries no weight.
 */
nntrainer::Tensor &convWeight(nntrainer::NeuralNetwork &nn) {
  auto graph = nn.getFlatGraph();
  if (graph.size() != 2u) {
    throw std::logic_error("expected exactly an input and a conv2d node");
  }
  return graph.at(1)->getWeight(0);
}

/**
 * @brief A code that is not zero at any tap, so that a payload silently
 * reinitialized to zero, or a scale vector read from the wrong offset, shows up
 * as a mismatch instead of as two equal zeros.
 */
void fillTapPattern(nntrainer::Tensor &weight) {
  int8_t *storage = weight.getAddress<int8_t>(0);
  const size_t taps = weight.size();
  for (size_t i = 0; i < taps; ++i) {
    /// codes within [-127, 127], none of them the tampering code below
    storage[i] = static_cast<int8_t>((i * 37 + 11) % 201) - 100;
  }
}

/**
 * @brief Write a scale vector that differs per output channel. The values are
 * not powers of two, so a scale stored as anything other than its fp32 form is
 * caught, and none of them is 1 or 0, so a vector that was never applied is
 * too.
 */
void fillScalePattern(nntrainer::Tensor &weight) {
  float *scales = weight.getScale<float>();
  for (unsigned int f = 0; f < weight.scale_size(); ++f) {
    scales[f] = 0.001f + 0.0025f * static_cast<float>(f + 1);
  }
}

/**
 * @brief Copy of the int8 payload of a tensor, in storage order.
 */
std::vector<int8_t> snapshotPayload(const nntrainer::Tensor &weight) {
  const int8_t *storage = weight.getAddress<int8_t>(0);
  return std::vector<int8_t>(storage, storage + weight.size());
}

/**
 * @brief Copy of the scale vector of a tensor, one entry per output channel.
 */
std::vector<float> snapshotScales(const nntrainer::Tensor &weight) {
  const float *scales = weight.getScale<float>();
  return std::vector<float>(scales, scales + weight.scale_size());
}

/**
 * @brief A code no tap of the pattern above takes, written into the bytes of
 * the payload by the test that checks a payload edit does not go unnoticed.
 */
constexpr int8_t kTamperedCode = 127;

/**
 * @brief Bytes a quantized weight occupies in the model file: the quantization
 * scheme that prefixes the block, then the int8 taps and the fp32 scale per
 * output channel that follow it.
 */
size_t quantizedWeightBytes(const nntrainer::Tensor &weight) {
  return sizeof(uint16_t) + weight.getMemoryBytes();
}

/**
 * @brief Offset the conv2d weight block starts at in the model file. The graph
 * has exactly one tensor before it (its own bias) and the weight is requested
 * first, so the offset is known as long as nothing else landed in front of it —
 * which is asserted rather than assumed, because a silent shift would make the
 * byte level tests read the wrong bytes instead of failing.
 */
size_t convWeightOffset(nntrainer::NeuralNetwork &nn) {
  auto graph = nn.getFlatGraph();
  if (graph.size() != 2u || graph.at(0)->getNumWeights() != 0u ||
      graph.at(1)->getWeight(0).getDataType() != DataType::QINT8) {
    throw std::logic_error(
      "the conv2d weight is not the first tensor the model "
      "writes, so its file offset is not known");
  }
  return 0;
}

} // namespace

/**
 * @brief The model file holds the quantized weight, not a dequantized copy of
 * it: the int8 taps plus one fp32 scale per output channel, and not the fp32
 * per tap an unquantized weight costs. The whole point of storing a conv2d
 * weight quantized is this size, so it is pinned as a byte count:
 *   sizeof(uint16_t)                        quantization scheme
 *   + filter_size * in_ch * kh * kw         int8 taps
 *   + filter_size * sizeof(float)           one fp32 scale per channel
 *   + filter_size * sizeof(float)           fp32 bias of the same layer
 * A scale vector that was silently not serialized leaves the file short by
 * exactly the second fp32 term, which is the failure that matters most here.
 */
TEST(ConvQuantWeightPersistence, qint8WeightFileIsQuantizedSize) {
  auto nn = buildQint8InferenceNN();
  nntrainer::Tensor &weight = convWeight(*nn);

  // the request established the channel last spelling and its scale axis
  ASSERT_EQ(weight.getDataType(), DataType::QINT8);
  ASSERT_EQ(weight.q_scheme(), nntrainer::QScheme::PER_CHANNEL_AFFINE);
  ASSERT_EQ(weight.batch(), RT_KH);
  ASSERT_EQ(weight.channel(), RT_IN_CH);
  ASSERT_EQ(weight.height(), RT_KW);
  ASSERT_EQ(weight.width(), RT_FILTERS);
  ASSERT_EQ(weight.scale_size(), RT_FILTERS);

  TempFile file("layout.bin");
  nn->save(file.pathfile(), ModelFormat::MODEL_FORMAT_BIN);
  std::ifstream stream(file.pathfile(), std::ios::binary | std::ios::ate);
  ASSERT_TRUE(stream.is_open());
  const size_t file_size = stream.tellg();

  EXPECT_EQ(file_size,
            quantizedWeightBytes(weight) + RT_FILTERS * sizeof(float))
    << "the file must hold the int8 payload, the fp32 scale vector and the "
       "bias";
  /// the same weight stored in fp32 would cost strictly more
  EXPECT_LT(quantizedWeightBytes(weight), weight.size() * sizeof(float));
}

/**
 * @brief The byte layout itself: the scheme prefixes the block, the int8
 * payload follows it unchanged, and the filter_size fp32 scales trail the
 * payload, each one the scale of the output channel at that index. Pinned
 * against the raw bytes rather than through a reader, since a reader can
 * compensate for a moved prefix and the model would still load a weight whose
 * scales are the wrong numbers for the payload.
 */
TEST(ConvQuantWeightPersistence, qint8WeightByteLayout) {
  auto nn = buildQint8InferenceNN();
  nntrainer::Tensor &weight = convWeight(*nn);
  fillTapPattern(weight);
  fillScalePattern(weight);

  const std::vector<int8_t> taps = snapshotPayload(weight);
  const std::vector<float> scales = snapshotScales(weight);

  TempFile file("bytes.bin");
  nn->save(file.pathfile(), ModelFormat::MODEL_FORMAT_BIN);
  std::ifstream stream(file.pathfile(), std::ios::binary);
  ASSERT_TRUE(stream.is_open());
  const size_t offset = convWeightOffset(*nn);
  stream.seekg(static_cast<std::streamoff>(offset));
  ASSERT_FALSE(stream.fail());

  const size_t block_size = quantizedWeightBytes(weight);
  std::vector<char> block(block_size);
  stream.read(block.data(), block_size);
  ASSERT_FALSE(stream.fail()) << "the weight block does not fit in the file";

  uint16_t qscheme = 0;
  std::memcpy(&qscheme, block.data(), sizeof(qscheme));
  EXPECT_EQ(qscheme,
            static_cast<uint16_t>(nntrainer::QScheme::PER_CHANNEL_AFFINE))
    << "the scheme prefix is what tells a reader how many scales follow";

  const char *payload = block.data() + sizeof(uint16_t);
  EXPECT_EQ(std::memcmp(payload, taps.data(), taps.size()), 0)
    << "the int8 payload must reach the file unchanged";

  // one fp32 scale per output channel, trailing the payload
  for (unsigned int f = 0; f < RT_FILTERS; ++f) {
    float on_disk = 0.0f;
    std::memcpy(&on_disk, payload + taps.size() + f * sizeof(float),
                sizeof(float));
    EXPECT_EQ(on_disk, scales[f]) << "at output channel " << f;
  }
}

/**
 * @brief The primitive the model load path is built on, invoked directly: the
 * per-node read brings the payload and the scales back exactly, with the
 * tensor's own file offset honored, which is where the two-byte scheme prefix
 * is accounted for.
 *
 * @note This is the std::ifstream overload, not the overload Model::load() uses
 * by default. That distinction is the subject of the ConvQuantWeightLoad tests
 * below: the default build maps the model file and hands the layers a mapped
 * view, and a quantized weight read through that source keeps neither its
 * payload's alignment nor its scales. Calling the stream overload by hand is
 * what pins that the bytes this suite saves are restorable at all; what the
 * model actually does with them is tested where the model does it.
 */
TEST(ConvQuantWeightPersistence, qint8WeightReadBackFromModelFile) {
  auto nn = buildQint8InferenceNN();
  nntrainer::Tensor &weight = convWeight(*nn);
  fillTapPattern(weight);
  fillScalePattern(weight);

  const std::vector<int8_t> taps = snapshotPayload(weight);
  const std::vector<float> scales = snapshotScales(weight);

  TempFile file("model_read.bin");
  nn->save(file.pathfile(), ModelFormat::MODEL_FORMAT_BIN);

  // a second model of the same graph, restoring the weights from that file
  auto restoring = buildQint8InferenceNN();
  restoring->load(file.pathfile(), ModelFormat::MODEL_FORMAT_BIN);
  std::ifstream stream(file.pathfile(), std::ios::binary);
  ASSERT_TRUE(stream.is_open());
  restoring->getFlatGraph().at(1)->read(
    stream, false, ml::train::ExecutionMode::INFERENCE, false,
    std::numeric_limits<size_t>::max(), true, -1);

  nntrainer::Tensor &restored = convWeight(*restoring);
  ASSERT_EQ(restored.getDataType(), DataType::QINT8);
  ASSERT_EQ(restored.scale_size(), RT_FILTERS);

  EXPECT_EQ(snapshotPayload(restored), taps);
  EXPECT_EQ(snapshotScales(restored), scales);
}

/**
 * @brief Sensitivity of the round trip below: an edit to one payload byte of
 * the file has to come back as a different payload, and must leave the scales
 * alone. Without this, a payload and a scale vector that were both silently
 * ignored would look like a passing round trip.
 */
TEST(ConvQuantWeightPersistence, qint8TensorTamperedPayloadIsCaught) {
  const TensorDim dim(RT_KH, RT_IN_CH, RT_KW, RT_FILTERS,
                      TensorDim::TensorType(Format::NHWC, DataType::QINT8));

  nntrainer::Tensor weight(dim, true, nntrainer::Initializer::NONE, "w",
                           nntrainer::QScheme::PER_CHANNEL_AFFINE);
  fillTapPattern(weight);
  fillScalePattern(weight);

  const std::vector<int8_t> taps = snapshotPayload(weight);
  const std::vector<float> scales = snapshotScales(weight);

  TempFile file("tampered.bin");
  {
    std::ofstream stream(file.pathfile(), std::ios::binary);
    ASSERT_TRUE(stream.is_open());
    weight.save(stream);
  }
  // edit one payload byte, between the scheme prefix and the scale vector
  {
    std::fstream stream(file.pathfile(),
                        std::ios::binary | std::ios::in | std::ios::out);
    ASSERT_TRUE(stream.is_open());
    const size_t at = sizeof(uint16_t) + taps.size() / 2;
    stream.seekp(static_cast<std::streamoff>(at));
    const int8_t tampered = kTamperedCode;
    stream.write(reinterpret_cast<const char *>(&tampered), sizeof(tampered));
    ASSERT_FALSE(stream.fail());
  }

  nntrainer::Tensor restored(dim, true, nntrainer::Initializer::NONE, "w",
                             nntrainer::QScheme::PER_CHANNEL_AFFINE);
  std::ifstream stream(file.pathfile(), std::ios::binary);
  ASSERT_TRUE(stream.is_open());
  restored.read(stream, 0, false);

  EXPECT_NE(snapshotPayload(restored), taps)
    << "a payload edit would not be visible through the round trip";
  /// the scales trail the edited byte, so they are still the saved ones
  EXPECT_EQ(snapshotScales(restored), scales);
}

/**
 * @brief The same persistence at the level of the tensor itself, without a
 * model in front: the primitive the model save path is built on round trips a
 * QINT8 payload and its scales, and restores the scheme from the file rather
 * than from the tensor it is read into. Independent of the model save order, so
 * when a field is ever added in front of the weight it is the byte layout tests
 * above that have to follow, not this persistence contract.
 */
TEST(ConvQuantWeightPersistence, qint8TensorSaveReadRoundtrip) {
  const TensorDim dim(RT_KH, RT_IN_CH, RT_KW, RT_FILTERS,
                      TensorDim::TensorType(Format::NHWC, DataType::QINT8));

  nntrainer::Tensor weight(dim, true, nntrainer::Initializer::NONE, "w",
                           nntrainer::QScheme::PER_CHANNEL_AFFINE);
  fillTapPattern(weight);
  fillScalePattern(weight);

  const std::vector<int8_t> taps = snapshotPayload(weight);
  const std::vector<float> scales = snapshotScales(weight);

  TempFile file("tensor.bin");
  {
    std::ofstream stream(file.pathfile(), std::ios::binary);
    ASSERT_TRUE(stream.is_open());
    weight.save(stream);
  }

  // read into a tensor whose scheme and scales are not the ones written
  nntrainer::Tensor restored(dim, true, nntrainer::Initializer::NONE, "w",
                             nntrainer::QScheme::PER_CHANNEL_AFFINE);
  std::ifstream stream(file.pathfile(), std::ios::binary);
  ASSERT_TRUE(stream.is_open());
  restored.read(stream, 0, false);

  EXPECT_EQ(restored.q_scheme(), nntrainer::QScheme::PER_CHANNEL_AFFINE);
  EXPECT_EQ(snapshotPayload(restored), taps);
  EXPECT_EQ(snapshotScales(restored), scales);
}

namespace {

/**
 * @brief Whether a scale vector is all zeros, i.e. whether any scale arrived at
 * all. A function rather than the inline predicate because the vector here is
 * usually a temporary, and std::all_of(v.begin(), v.end(), ...) on a temporary
 * evaluates v twice and compares iterators into two different vectors.
 */
bool allZeroScale(const std::vector<float> &scales) {
  return std::all_of(scales.begin(), scales.end(),
                     [](float s) { return s == 0.0f; });
}

/**
 * @brief Forward a model once with an input of its own input shape, so that a
 * weight that cannot be used has to report it. The input values do not matter:
 * the assertions below are about whether the forward pass runs and, when it
 * does not, about why.
 */
void forwardOnce(nntrainer::NeuralNetwork &nn) {
  const TensorDim in_dim = nn.getFlatGraph().at(0)->getOutputDimensions()[0];
  auto input = std::make_shared<nntrainer::Tensor>(in_dim);
  input->setValue(0.5f);
  nn.forwarding({input}, {}, false);
}

} // namespace

/**
 * @brief A QINT8 weight cannot be computed by this layer yet, and saying so is
 * the point: finalize() admits a QINT8 weight only for a channel last model,
 * and nothing in Conv2DLayer computes a channel last kernel -- the channel
 * first path builds its im2col columns against a channel first operand, and the
 * channel last path has no branch on the weight dtype at all, so it would hand
 * int8 taps to a floating point dot. That is a number that looks like an
 * output. A model built the only way finalize() allows therefore has to be
 * refused at the forward pass rather than produce one.
 */
TEST(ConvQuantWeightLoad, channelLastQint8ForwardIsRefused) {
  auto nn = buildQint8InferenceNN();
  nntrainer::Tensor &weight = convWeight(*nn);
  fillTapPattern(weight);
  fillScalePattern(weight);

  ASSERT_EQ(weight.getDataType(), DataType::QINT8);
  ASSERT_EQ(weight.getFormat(), Format::NHWC);

  // scales present, so this is the layout refusal and not the missing scale one
  EXPECT_THROW(
    {
      try {
        forwardOnce(*nn);
      } catch (const std::runtime_error &e) {
        EXPECT_NE(std::string(e.what()).find("channel last QINT8 weight "
                                             "computation is not implemented"),
                  std::string::npos)
          << "unexpected rejection: " << e.what();
        throw;
      }
    },
    std::runtime_error);
}

/**
 * @brief Write a marker into the scale area of a weight, so that a later read
 * can tell whether anything wrote that area. A marker rather than a guess at
 * what allocation left there: a freshly allocated QINT8 tensor's scale area is
 * whatever the allocator handed back, and under glibc's MALLOC_PERTURB_ (which
 * meson test sets) it is a nonzero byte pattern rather than zeros. An assertion
 * of the form "the scales are zero because nothing wrote them" is therefore
 * true only for one allocator state, and it is not the invariant worth pinning.
 */
void fillScaleMarker(nntrainer::Tensor &weight, float marker) {
  float *scales = weight.getScale<float>();
  for (size_t f = 0; f < weight.scale_size(); ++f) {
    scales[f] = marker;
  }
}

/**
 * @brief What Model::load() -- the path every user of a saved model takes --
 * does with a QINT8 conv2d weight's scale vector.
 *
 * The block save() writes is the quantization scheme, then the int8 taps, then
 * one fp32 scale per output channel. The default build reads a model through a
 * mapped view of the file, and a quantized weight reaching that path is read
 * through the base TensorBase::read, which knows nothing about the block: it
 * reads bytes() -- one byte per tap -- starting where the scheme prefix sits.
 * So two things follow, deterministically: the payload comes back shifted by
 * the width of the prefix, and the scale area past the end of that read is
 * never written at all.
 *
 * Both are pinned against a marker rather than against zero. The load itself
 * cannot notice: it reads the byte count it expects and every read succeeds.
 */
TEST(ConvQuantWeightLoad, modelLoadNeverWritesTheScaleArea) {
  auto nn = buildQint8InferenceNN();
  fillTapPattern(convWeight(*nn));
  fillScalePattern(convWeight(*nn));
  const std::vector<int8_t> taps = snapshotPayload(convWeight(*nn));
  const std::vector<float> scales = snapshotScales(convWeight(*nn));
  ASSERT_FALSE(allZeroScale(scales));

  TempFile file("load_path.bin");
  nn->save(file.pathfile(), ModelFormat::MODEL_FORMAT_BIN);

  constexpr float kMarker = -17.5f;
  auto restoring = buildQint8InferenceNN();
  fillScaleMarker(convWeight(*restoring), kMarker);
  restoring->load(file.pathfile(), ModelFormat::MODEL_FORMAT_BIN);

  nntrainer::Tensor &restored = convWeight(*restoring);
  ASSERT_EQ(restored.getDataType(), DataType::QINT8);
  ASSERT_EQ(restored.scale_size(), RT_FILTERS);

  // the read stopped before the scale area, so the marker is still there
  for (unsigned int f = 0; f < RT_FILTERS; ++f) {
    EXPECT_EQ(restored.getScale<float>()[f], kMarker)
      << "at output channel " << f
      << ": the load wrote the scale area, so this expectation and the two "
         "refusal tests below have to be revisited";
  }
  // and what it wrote in front of it is the block, shifted by the prefix
  const int8_t *payload = restored.getAddress<int8_t>(0);
  EXPECT_EQ(static_cast<uint16_t>(payload[0]) |
              (static_cast<uint16_t>(payload[1]) << 8),
            static_cast<uint16_t>(nntrainer::QScheme::PER_CHANNEL_AFFINE))
    << "the taps start where the scheme prefix is, i.e. the payload is shifted";
  EXPECT_NE(std::vector<int8_t>(payload, payload + taps.size()), taps)
    << "the payload came back identical, so the read no longer starts too "
       "early";
}

/**
 * @brief A model loaded this way cannot compute, and has to say so whichever
 * way its scale area happens to read. What the load leaves there is allocator
 * state, so the refusal is pinned for both values it can take, and neither one
 * is allowed to produce numbers:
 *   - scales read as zero: every tap of the layer would dequantize to zero, an
 *     all zero output that nothing else marks as wrong, so the weight itself is
 *     refused as one that arrived without scales.
 *   - scales read as anything else: the weight is a channel last weight, which
 *     nothing in Conv2DLayer computes, so the layout is refused.
 * Together these are the property that matters: loading a QINT8 conv2d model
 * and running it either stops with a reason or does not happen.
 */
TEST(ConvQuantWeightLoad, loadedModelRefusesRatherThanComputing) {
  auto nn = buildQint8InferenceNN();
  fillTapPattern(convWeight(*nn));
  fillScalePattern(convWeight(*nn));

  TempFile file("load_refuse.bin");
  nn->save(file.pathfile(), ModelFormat::MODEL_FORMAT_BIN);

  // scales read as zero: the weight cannot carry a zero scale
  auto zeroed = buildQint8InferenceNN();
  zeroed->load(file.pathfile(), ModelFormat::MODEL_FORMAT_BIN);
  fillScaleMarker(convWeight(*zeroed), 0.0f);
  EXPECT_THROW(
    {
      try {
        forwardOnce(*zeroed);
      } catch (const std::runtime_error &e) {
        EXPECT_NE(std::string(e.what()).find("has no scale factors"),
                  std::string::npos)
          << "unexpected rejection: " << e.what();
        throw;
      }
    },
    std::runtime_error);

  // scales read as something: the layout is what stops it instead
  auto other = buildQint8InferenceNN();
  other->load(file.pathfile(), ModelFormat::MODEL_FORMAT_BIN);
  fillScaleMarker(convWeight(*other), -17.5f);
  EXPECT_THROW(
    {
      try {
        forwardOnce(*other);
      } catch (const std::runtime_error &e) {
        EXPECT_NE(std::string(e.what()).find(
                    "channel last QINT8 weight computation is not implemented"),
                  std::string::npos)
          << "unexpected rejection: " << e.what();
        throw;
      }
    },
    std::runtime_error);
}

/**
 * @brief The same weight through the std::ifstream read, which does know the
 * block layout: the scales come back and the missing scale refusal does not
 * fire. Read alongside the test above, this says the file is fine and the model
 * load path is what loses the scales -- if both refused for the same reason the
 * diagnosis would be the save side instead.
 */
TEST(ConvQuantWeightLoad, ifstreamReadRestoresScalesWhereLoadDoesNot) {
  auto nn = buildQint8InferenceNN();
  fillTapPattern(convWeight(*nn));
  fillScalePattern(convWeight(*nn));
  const std::vector<float> scales = snapshotScales(convWeight(*nn));

  TempFile file("load_contrast.bin");
  nn->save(file.pathfile(), ModelFormat::MODEL_FORMAT_BIN);

  auto restoring = buildQint8InferenceNN();
  restoring->load(file.pathfile(), ModelFormat::MODEL_FORMAT_BIN);
  std::ifstream stream(file.pathfile(), std::ios::binary);
  ASSERT_TRUE(stream.is_open());
  restoring->getFlatGraph().at(1)->read(
    stream, false, ml::train::ExecutionMode::INFERENCE, false,
    std::numeric_limits<size_t>::max(), true, -1);

  EXPECT_EQ(snapshotScales(convWeight(*restoring)), scales);
  EXPECT_FALSE(allZeroScale(snapshotScales(convWeight(*restoring))));
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

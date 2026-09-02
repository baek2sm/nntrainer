// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_layers_nhwc.cpp
 * @date   02 September 2026
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  NHWC coverage for the layers with a layout-specific forward path.
 *
 * Every case runs the layer twice over the same logical inputs: once stored as
 * NHWC and once as NCHW. NCHW is the layout those forwards have always served,
 * so it is the reference, and the two must agree exactly -- they perform the
 * same reductions over the same values, only at different addresses. Cases that
 * pin an element mapping outright additionally state hand-computed values,
 * because two implementations that are wrong the same way would otherwise pass.
 *
 * @see https://github.com/nntrainer/nntrainer
 */
#include <gtest/gtest.h>

#include <concat_layer.h>
#include <nhwc_layer_test_util.h>
#include <pooling2d_layer.h>
#include <upsample2d_layer.h>

using nhwc_test::runForwardNhwc;

namespace {

/** deterministic filler, distinct per element, in logical order */
static std::vector<float> ramp(size_t n, float scale = 1.0f) {
  std::vector<float> v(n);
  for (size_t i = 0; i < n; ++i)
    v[i] = std::sin((float)(i + 1) * 0.7f) * scale;
  return v;
}

/**
 * @brief Runs a pooling layer over the given layout and flattens the output in
 * logical order.
 */
static std::vector<float> runPool(const std::vector<std::string> &props,
                                  const nhwc_test::Shape &shape,
                                  const std::vector<float> &values,
                                  ml::train::TensorDim::Format format,
                                  bool training = false) {
  auto layer = nntrainer::createLayer<nntrainer::Pooling2DLayer>();
  layer->setProperty(props);
  auto r =
    nhwc_test::runForward(layer.get(), {shape}, {values}, format, training);
  EXPECT_EQ(r.size(), 1u);
  return nhwc_test::toLogical(r[0]);
}

TEST(NhwcPooling, maxMatchesNchw) {
  const auto values = ramp(2 * 3 * 4 * 4, 4.0f);
  const std::vector<std::string> props{"pooling=max", "pool_size=2,2",
                                       "stride=2,2"};
  auto nhwc =
    runPool(props, {2, 3, 4, 4}, values, ml::train::TensorDim::Format::NHWC);
  auto nchw =
    runPool(props, {2, 3, 4, 4}, values, ml::train::TensorDim::Format::NCHW);
  ASSERT_EQ(nhwc.size(), 2u * 3u * 2u * 2u);
  ASSERT_EQ(nhwc.size(), nchw.size());
  for (size_t i = 0; i < nhwc.size(); ++i)
    EXPECT_EQ(nhwc[i], nchw[i]) << "at logical index " << i;
}

TEST(NhwcPooling, maxStrideOneSamePaddingMatchesNchw) {
  const auto values = ramp(2 * 3 * 5 * 6, 4.0f);
  const std::vector<std::string> props{"pooling=max", "pool_size=3,3",
                                       "stride=1,1", "padding=same"};
  auto nhwc =
    runPool(props, {2, 3, 5, 6}, values, ml::train::TensorDim::Format::NHWC);
  auto nchw =
    runPool(props, {2, 3, 5, 6}, values, ml::train::TensorDim::Format::NCHW);
  ASSERT_EQ(nhwc.size(), nchw.size());
  for (size_t i = 0; i < nhwc.size(); ++i)
    EXPECT_EQ(nhwc[i], nchw[i]) << "at logical index " << i;
}

/**
 * An asymmetric patch offset exercises the case where a patch reaches past the
 * left and top edge only, so the in-range count that averages divide by varies
 * per output element.
 */
TEST(NhwcPooling, averagePaddingMatchesNchw) {
  const auto values = ramp(2 * 2 * 4 * 4, 4.0f);
  const std::vector<std::string> props{"pooling=average", "pool_size=3,3",
                                       "stride=1,1", "padding=1,1"};
  auto nhwc =
    runPool(props, {2, 2, 4, 4}, values, ml::train::TensorDim::Format::NHWC);
  auto nchw =
    runPool(props, {2, 2, 4, 4}, values, ml::train::TensorDim::Format::NCHW);
  ASSERT_EQ(nhwc.size(), nchw.size());
  for (size_t i = 0; i < nhwc.size(); ++i)
    EXPECT_FLOAT_EQ(nhwc[i], nchw[i]) << "at logical index " << i;
}

TEST(NhwcPooling, averageMatchesNchw) {
  const auto values = ramp(1 * 4 * 6 * 6, 4.0f);
  const std::vector<std::string> props{"pooling=average", "pool_size=2,2",
                                       "stride=2,2"};
  auto nhwc =
    runPool(props, {1, 4, 6, 6}, values, ml::train::TensorDim::Format::NHWC);
  auto nchw =
    runPool(props, {1, 4, 6, 6}, values, ml::train::TensorDim::Format::NCHW);
  ASSERT_EQ(nhwc.size(), nchw.size());
  for (size_t i = 0; i < nhwc.size(); ++i)
    EXPECT_FLOAT_EQ(nhwc[i], nchw[i]) << "at logical index " << i;
}

TEST(NhwcPooling, globalMaxMatchesNchw) {
  const auto values = ramp(2 * 3 * 5 * 5, 4.0f);
  const std::vector<std::string> props{"pooling=global_max"};
  auto nhwc =
    runPool(props, {2, 3, 5, 5}, values, ml::train::TensorDim::Format::NHWC);
  auto nchw =
    runPool(props, {2, 3, 5, 5}, values, ml::train::TensorDim::Format::NCHW);
  ASSERT_EQ(nhwc.size(), 2u * 3u);
  ASSERT_EQ(nhwc.size(), nchw.size());
  for (size_t i = 0; i < nhwc.size(); ++i)
    EXPECT_EQ(nhwc[i], nchw[i]) << "at logical index " << i;
}

TEST(NhwcPooling, globalAverageMatchesNchw) {
  const auto values = ramp(2 * 3 * 5 * 5, 4.0f);
  const std::vector<std::string> props{"pooling=global_average"};
  auto nhwc =
    runPool(props, {2, 3, 5, 5}, values, ml::train::TensorDim::Format::NHWC);
  auto nchw =
    runPool(props, {2, 3, 5, 5}, values, ml::train::TensorDim::Format::NCHW);
  ASSERT_EQ(nhwc.size(), nchw.size());
  for (size_t i = 0; i < nhwc.size(); ++i)
    EXPECT_FLOAT_EQ(nhwc[i], nchw[i]) << "at logical index " << i;
}

/**
 * [1, 2, 2, 2] holding 1..8 in logical order is channel 0 = {1,2,3,4} and
 * channel 1 = {5,6,7,8}, so the 2x2 max over the single plane of each is
 * {4, 8}. That the answer is not {7, 8} is the point: in NHWC storage those
 * eight values sit as four pixels of two channels, and a forward that pooled
 * contiguous memory instead of channel runs would return 7 and 8.
 */
TEST(NhwcPooling, maxChannelMappingIsExact) {
  auto layer = nntrainer::createLayer<nntrainer::Pooling2DLayer>();
  layer->setProperty({"pooling=max", "pool_size=2,2", "stride=2,2"});
  auto r =
    runForwardNhwc(layer.get(), {{{1, 2, 2, 2}}}, {{1, 2, 3, 4, 5, 6, 7, 8}});
  ASSERT_EQ(r.size(), 1u);
  EXPECT_EQ(r[0].getFormat(), ml::train::TensorDim::Format::NHWC);
  nhwc_test::expectClose(r[0], {4.0f, 8.0f}, 0.0f);
}

/**
 * The NHWC forward records no pool_helper, which is what the max-pool backward
 * reads back. Failing loudly is the contract; computing a derivative from an
 * unwritten helper is what would happen otherwise.
 */
TEST(NhwcPooling, trainingThrows) {
  EXPECT_THROW(runPool({"pooling=max", "pool_size=2,2", "stride=2,2"},
                       {1, 2, 4, 4}, ramp(1 * 2 * 4 * 4, 4.0f),
                       ml::train::TensorDim::Format::NHWC, true),
               std::exception);
}

/** [1, 2, 2, 2] with 1..8 -> nearest x (2, 2) -> [1, 2, 4, 4]. */
TEST(NhwcUpsample, nearestChannelMappingIsExact) {
  auto layer = nntrainer::createLayer<nntrainer::Upsample2dLayer>();
  layer->setProperty({"upsample=nearest", "kernel_size=2,2"});
  auto r =
    runForwardNhwc(layer.get(), {{{1, 2, 2, 2}}}, {{1, 2, 3, 4, 5, 6, 7, 8}});
  ASSERT_EQ(r.size(), 1u);
  EXPECT_EQ(r[0].getFormat(), ml::train::TensorDim::Format::NHWC);
  /// channel 0 plane {1,2,3,4} and channel 1 plane {5,6,7,8}, each repeated
  /// twice per row and every row duplicated
  const std::vector<float> expected = {1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4,
                                       4, 3, 3, 4, 4, 5, 5, 6, 6, 5, 5,
                                       6, 6, 7, 7, 8, 8, 7, 7, 8, 8};
  nhwc_test::expectClose(r[0], expected, 0.0f);
}

static std::vector<float> runUpsample(const std::vector<std::string> &props,
                                      const nhwc_test::Shape &shape,
                                      const std::vector<float> &values,
                                      ml::train::TensorDim::Format format) {
  auto layer = nntrainer::createLayer<nntrainer::Upsample2dLayer>();
  layer->setProperty(props);
  auto r = nhwc_test::runForward(layer.get(), {shape}, {values}, format);
  EXPECT_EQ(r.size(), 1u);
  return nhwc_test::toLogical(r[0]);
}

TEST(NhwcUpsample, nearestMatchesNchw) {
  const auto values = ramp(2 * 4 * 5 * 3, 3.0f);
  const std::vector<std::string> props{"upsample=nearest", "kernel_size=2,3"};
  auto nhwc = runUpsample(props, {2, 4, 5, 3}, values,
                          ml::train::TensorDim::Format::NHWC);
  auto nchw = runUpsample(props, {2, 4, 5, 3}, values,
                          ml::train::TensorDim::Format::NCHW);
  ASSERT_EQ(nhwc.size(), 2u * 4u * 10u * 9u);
  ASSERT_EQ(nhwc.size(), nchw.size());
  for (size_t i = 0; i < nhwc.size(); ++i)
    EXPECT_EQ(nhwc[i], nchw[i]) << "at logical index " << i;
}

/**
 * Bilinear is carried in float and stored back in the tensor dtype; on FP32
 * storage that round trip is exact, so the two layouts must match bit for bit.
 */
TEST(NhwcUpsample, bilinearMatchesNchw) {
  const auto values = ramp(1 * 3 * 4 * 4, 3.0f);
  const std::vector<std::string> props{"upsample=bilinear", "kernel_size=2,2"};
  auto nhwc = runUpsample(props, {1, 3, 4, 4}, values,
                          ml::train::TensorDim::Format::NHWC);
  auto nchw = runUpsample(props, {1, 3, 4, 4}, values,
                          ml::train::TensorDim::Format::NCHW);
  ASSERT_EQ(nhwc.size(), 1u * 3u * 8u * 8u);
  ASSERT_EQ(nhwc.size(), nchw.size());
  for (size_t i = 0; i < nhwc.size(); ++i)
    EXPECT_EQ(nhwc[i], nchw[i]) << "at logical index " << i;
}

/** the corner of an unaligned-corners bilinear upsample keeps its input value
 */
TEST(NhwcUpsample, bilinearKeepsCorners) {
  auto layer = nntrainer::createLayer<nntrainer::Upsample2dLayer>();
  layer->setProperty({"upsample=bilinear", "kernel_size=2,2"});
  auto r =
    runForwardNhwc(layer.get(), {{{1, 2, 2, 2}}}, {{1, 2, 3, 4, 5, 6, 7, 8}});
  ASSERT_EQ(r.size(), 1u);
  /// logical (b 0, c 0, h 0, w 0) is 1 and the last row/column keeps 3 and 7
  EXPECT_NEAR(r[0].getValue(0, 0, 0, 0), 1.0f, 1e-5f);
  EXPECT_NEAR(r[0].getValue(0, 0, 0, 3), 2.0f, 1e-5f);
  EXPECT_NEAR(r[0].getValue(0, 0, 3, 0), 3.0f, 1e-5f);
  EXPECT_NEAR(r[0].getValue(0, 0, 3, 3), 4.0f, 1e-5f);
  EXPECT_NEAR(r[0].getValue(0, 1, 0, 0), 5.0f, 1e-5f);
}

static std::vector<float> runConcat(const std::vector<std::string> &props,
                                    const std::vector<nhwc_test::Shape> &shapes,
                                    const std::vector<std::vector<float>> &vals,
                                    ml::train::TensorDim::Format format) {
  auto layer = nntrainer::createLayer<nntrainer::ConcatLayer>();
  layer->setProperty(props);
  auto r = nhwc_test::runForward(layer.get(), shapes, vals, format);
  EXPECT_EQ(r.size(), 1u);
  return nhwc_test::toLogical(r[0]);
}

/**
 * Channel-axis concat of [1, 2, 2,2] = 1..8 and [1, 1, 2,2] = 9..12 gives
 * [1, 3, 2, 2]: the second input becomes channel 2, so in logical order the
 * planes simply append. In storage the same result interleaves per pixel, which
 * is what the layout-specific path has to get right.
 */
TEST(NhwcConcat, channelAxisIsExact) {
  auto layer = nntrainer::createLayer<nntrainer::ConcatLayer>();
  layer->setProperty({"axis=1"});
  auto r = runForwardNhwc(layer.get(), {{{1, 2, 2, 2}}, {{1, 1, 2, 2}}},
                          {{1, 2, 3, 4, 5, 6, 7, 8}, {9, 10, 11, 12}});
  ASSERT_EQ(r.size(), 1u);
  EXPECT_EQ(r[0].getFormat(), ml::train::TensorDim::Format::NHWC);
  nhwc_test::expectClose(r[0], {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, 0.0f);
}

TEST(NhwcConcat, channelAxisMatchesNchw) {
  const auto v0 = ramp(2 * 2 * 3 * 2, 5.0f);
  const auto v1 = ramp(2 * 1 * 3 * 2, 5.0f);
  const std::vector<std::string> props{"axis=1"};
  const std::vector<nhwc_test::Shape> shapes{{2, 2, 3, 2}, {2, 1, 3, 2}};
  auto nhwc =
    runConcat(props, shapes, {v0, v1}, ml::train::TensorDim::Format::NHWC);
  auto nchw =
    runConcat(props, shapes, {v0, v1}, ml::train::TensorDim::Format::NCHW);
  ASSERT_EQ(nhwc.size(), 2u * 3u * 3u * 2u);
  ASSERT_EQ(nhwc.size(), nchw.size());
  for (size_t i = 0; i < nhwc.size(); ++i)
    EXPECT_EQ(nhwc[i], nchw[i]) << "at logical index " << i;
}

TEST(NhwcConcat, heightAxisMatchesNchw) {
  const auto v0 = ramp(2 * 2 * 2 * 3, 5.0f);
  const auto v1 = ramp(2 * 2 * 3 * 3, 5.0f);
  const std::vector<std::string> props{"axis=2"};
  const std::vector<nhwc_test::Shape> shapes{{2, 2, 2, 3}, {2, 2, 3, 3}};
  auto nhwc =
    runConcat(props, shapes, {v0, v1}, ml::train::TensorDim::Format::NHWC);
  auto nchw =
    runConcat(props, shapes, {v0, v1}, ml::train::TensorDim::Format::NCHW);
  ASSERT_EQ(nhwc.size(), 2u * 2u * 5u * 3u);
  ASSERT_EQ(nhwc.size(), nchw.size());
  for (size_t i = 0; i < nhwc.size(); ++i)
    EXPECT_EQ(nhwc[i], nchw[i]) << "at logical index " << i;
}

TEST(NhwcConcat, widthAxisMatchesNchw) {
  const auto v0 = ramp(2 * 2 * 3 * 2, 5.0f);
  const auto v1 = ramp(2 * 2 * 3 * 3, 5.0f);
  const std::vector<std::string> props{"axis=3"};
  const std::vector<nhwc_test::Shape> shapes{{2, 2, 3, 2}, {2, 2, 3, 3}};
  auto nhwc =
    runConcat(props, shapes, {v0, v1}, ml::train::TensorDim::Format::NHWC);
  auto nchw =
    runConcat(props, shapes, {v0, v1}, ml::train::TensorDim::Format::NCHW);
  ASSERT_EQ(nhwc.size(), 2u * 2u * 3u * 5u);
  ASSERT_EQ(nhwc.size(), nchw.size());
  for (size_t i = 0; i < nhwc.size(); ++i)
    EXPECT_EQ(nhwc[i], nchw[i]) << "at logical index " << i;
}

/**
 * A batch-axis concat is refused before a forward ever runs: the axis property
 * only accepts 1..3, so the NHWC path is never handed an axis it cannot place.
 * The check matters because the generic path would not refuse -- it copies
 * whole leading slices and would return a concatenation that looks plausible
 * and is wrong.
 */
TEST(NhwcConcat, batchAxisIsRejectedByProperty) {
  auto layer = nntrainer::createLayer<nntrainer::ConcatLayer>();
  EXPECT_THROW(layer->setProperty({"axis=0"}), std::exception);
}

} /* namespace */

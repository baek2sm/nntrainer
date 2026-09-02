// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_nntrainer_quant_scale_roundtrip.cpp
 * @date   02 September 2026
 * @brief  Save/read round-trip tests for the scale factors of the affine
 *         quantized tensor types (QINT4, QINT8, QINT16).
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seungbaek <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 *         An affine quantized tensor holds its scale factors in memory as
 *         full-precision fp32, appended right after the packed data, and
 *         serializes them in that same layout. Int4QTensor alone sized the
 *         serialized scale region as sizeof(uint16_t) per scale while
 *         allocating sizeof(float) for it, so only 2 of every 4 scale bytes
 *         reached the file: a per-tensor scale came back as 0, and in a
 *         per-channel vector every scale past the first came back denormal.
 *         CharTensor/ShortTensor size both sides with sizeof(float) and are
 *         included as controls, so the asymmetry shows up within one suite.
 *
 *         The tensors here are built through Tensor(const TensorDim &, ...) and
 *         filled through the public accessors on purpose: the vector-of-vectors
 *         + scales constructor is unusable for QINT4 today (it validates
 *         scales.size() against scale_size() before the dimension is set), so
 *         going through it would test that instead.
 *
 *         @note This suite pins an on-disk layout: a serialized quantized
 *         tensor is [qscheme: u16][packed data][scales: fp32 * scale_size()].
 *         The QINT4 size is asserted in absolute bytes, because shrinking that
 *         region again would be a compatibility break rather than a refactor.
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <tensor.h>
#include <tensor_dim.h>

using nntrainer::Initializer;
using nntrainer::QScheme;
using nntrainer::Tdatatype;
using nntrainer::Tensor;
using nntrainer::TensorDim;
using nntrainer::Tformat;

namespace {

/// @brief Group size Int4QTensor uses for PER_CHANNEL_AFFINE (its default).
constexpr unsigned int kQint4GroupSize = 32;

/// @brief Size of the qscheme field save() writes ahead of the data bytes.
constexpr size_t kQuantizationInfoBytes = sizeof(uint16_t);

/**
 * @brief Fill @a target with a deterministic payload in [-8, 7].
 *
 * That is the range Int4QTensor::setValue() accepts, and it is storable by the
 * QINT8/QINT16 controls as well, so one payload serves all three types.
 */
void fillPayload(Tensor &target) {
  for (unsigned int i = 0; i < target.batch(); ++i) {
    for (unsigned int j = 0; j < target.channel(); ++j) {
      for (unsigned int k = 0; k < target.height(); ++k) {
        for (unsigned int l = 0; l < target.width(); ++l) {
          const int slot = static_cast<int>((i * 7 + j * 5 + k * 3 + l) % 16);
          target.setValue(i, j, k, l, static_cast<float>(slot - 8));
        }
      }
    }
  }
}

/**
 * @brief Allocate a quantized tensor of @a type holding @a n_scales scales.
 *
 * The qscheme has to be requested through this constructor: it is the one that
 * forwards it to the underlying tensor, and the scale region is sized from it.
 */
Tensor makeQuantTensor(unsigned int b, unsigned int c, unsigned int h,
                       unsigned int w, Tdatatype type, QScheme qscheme,
                       size_t n_scales, const std::string &name) {
  Tensor tensor(TensorDim(b, c, h, w, {Tformat::NCHW, type}), true,
                Initializer::NONE, name, qscheme);

  // if the qscheme did not reach the underlying tensor, the scale region would
  // be the wrong size and this test would prove nothing.
  EXPECT_EQ(tensor.scale_size(), n_scales)
    << "tensor " << name << " was built with the wrong scale region";

  fillPayload(tensor);
  return tensor;
}

/**
 * @brief Write @a n distinct scales into @a target.
 *
 * Dyadic multiples of 1.5: exactly representable, so both the round-trip
 * assertion and the byte comparison against the file are bit-exact rather than
 * tolerance checks, and a narrowed or shifted read cannot accidentally
 * reproduce a neighbour's value.
 */
void fillScales(Tensor &target, size_t n) {
  ASSERT_EQ(target.scale_size(), n);
  for (size_t i = 0; i < n; ++i) {
    target.getScale<float>()[i] =
      static_cast<float>(1.5 * static_cast<double>(i + 1));
  }
}

/**
 * @brief Write @a source to @a path and read it back into @a destination, the
 *        way a weight file is persisted and reloaded.
 *
 * @note Neither tensor is const: save()/read() validate and release the backing
 * memory around the stream operation.
 */
void saveAndRead(Tensor &source, Tensor &destination, const std::string &path) {
  {
    std::ofstream save_file(path, std::ios::out | std::ios::binary);
    source.save(save_file);
  }
  {
    std::ifstream read_file(path, std::ios::in | std::ios::binary);
    destination.read(read_file);
  }
}

/**
 * @brief Size of the file at @a path in bytes.
 */
size_t fileSize(const std::string &path) {
  std::ifstream file(path, std::ios::in | std::ios::binary | std::ios::ate);
  return static_cast<size_t>(file.tellg());
}

/**
 * @brief Assert that every scale of @a expected came back in @a actual exactly.
 */
void expectScalesEqual(const Tensor &expected, const Tensor &actual) {
  ASSERT_EQ(expected.scale_size(), actual.scale_size());
  for (size_t i = 0; i < expected.scale_size(); ++i) {
    EXPECT_EQ(expected.getScale<float>()[i], actual.getScale<float>()[i])
      << "scale[" << i << "] did not survive the save/read round trip";
  }
}

/**
 * @brief Bytes the packed data occupies ahead of the scale region, spelled out
 *        per dtype the way each tensor class lays it out.
 *
 * Deliberately not taken from getMemoryBytes(): the scale-region check below
 * has to stay independent of the value under test.
 */
size_t packedDataBytes(const Tensor &tensor) {
  switch (tensor.getDataType()) {
  case Tdatatype::QINT4:
    return (tensor.size() + 1) / 2; /// two nibbles per byte
  case Tdatatype::QINT8:
    return tensor.size(); /// one byte per element
  case Tdatatype::QINT16:
    return tensor.size() * sizeof(int16_t);
  default:
    ADD_FAILURE() << "unexpected dtype "
                  << static_cast<int>(tensor.getDataType());
    return 0;
  }
}

/**
 * @brief Assert that the file holds the fp32 scale factors verbatim at the
 *        offset the layout defines, independently of getMemoryBytes().
 *
 * Reading the bytes back rather than only comparing the reloaded tensor means a
 * change that narrows or relocates the scale region fails here even if save()
 * and read() are kept self-consistent.
 */
void expectSerializedScalesAreFp32(const Tensor &source,
                                   const std::string &path) {
  const size_t scale_bytes = source.scale_size() * sizeof(float);
  const size_t scale_offset = kQuantizationInfoBytes + packedDataBytes(source);

  std::vector<char> on_disk(scale_bytes, 0);
  std::ifstream file(path, std::ios::in | std::ios::binary);
  file.seekg(static_cast<std::streamoff>(scale_offset));
  file.read(on_disk.data(), static_cast<std::streamsize>(scale_bytes));

  ASSERT_EQ(static_cast<size_t>(file.gcount()), scale_bytes)
    << "only " << file.gcount() << " of the " << scale_bytes
    << " expected scale bytes are on disk at offset " << scale_offset;

  EXPECT_EQ(0,
            std::memcmp(on_disk.data(), source.getScale<float>(), scale_bytes))
    << "the serialized scale region does not hold the fp32 scale factors";
}

/**
 * @brief Round-trip one quantized type and assert the scales are preserved.
 */
void roundTripScales(const std::string &case_name, Tdatatype type,
                     QScheme qscheme, unsigned int b, unsigned int c,
                     unsigned int h, unsigned int w, size_t n_scales) {
  Tensor source =
    makeQuantTensor(b, c, h, w, type, qscheme, n_scales, case_name + "_src");
  Tensor destination =
    makeQuantTensor(b, c, h, w, type, qscheme, n_scales, case_name + "_dst");

  fillScales(source, n_scales);

  const std::string path = case_name + "_roundtrip.bin";
  saveAndRead(source, destination, path);

  EXPECT_EQ(source, destination);
  expectScalesEqual(source, destination);
  expectSerializedScalesAreFp32(source, path);

  EXPECT_EQ(0, std::remove(path.c_str()));
}

} // namespace

/**
 * @brief A QINT4 per-tensor scale survives save/read unchanged.
 */
TEST(nntrainer_QuantScaleRoundTrip, qint4_per_tensor_p) {
  roundTripScales("qint4_per_tensor", Tdatatype::QINT4,
                  QScheme::PER_TENSOR_AFFINE, 1, 2, 3, 5, 1);
}

/**
 * @brief A QINT4 per-channel scale vector survives save/read unchanged.
 */
TEST(nntrainer_QuantScaleRoundTrip, qint4_per_channel_p) {
  const unsigned int h = 4, w = 16;
  const size_t n_scales = static_cast<size_t>(h) * w / kQint4GroupSize;
  ASSERT_GT(n_scales, 1u); /// the shape must exercise a scale vector

  roundTripScales("qint4_per_channel", Tdatatype::QINT4,
                  QScheme::PER_CHANNEL_AFFINE, 1, 1, h, w, n_scales);
}

/**
 * @brief Pin the serialized size of a QINT4 tensor in absolute bytes.
 *
 * qscheme (2) + packed nibbles (32) + two fp32 scales (8). Spelling the
 * expected value out, rather than deriving it from getMemoryBytes(), is what
 * makes this fail if the scale region is ever narrowed to 2 bytes per scale
 * again.
 */
TEST(nntrainer_QuantScaleRoundTrip, qint4_saved_size_p) {
  const unsigned int h = 4, w = 16;
  const size_t n_scales = static_cast<size_t>(h) * w / kQint4GroupSize;
  ASSERT_EQ(n_scales, 2u);

  Tensor source =
    makeQuantTensor(1, 1, h, w, Tdatatype::QINT4, QScheme::PER_CHANNEL_AFFINE,
                    n_scales, "qint4_saved_size");
  fillScales(source, n_scales);
  ASSERT_EQ((source.size() + 1) / 2, 32u);

  const std::string path = "qint4_saved_size.bin";
  {
    std::ofstream save_file(path, std::ios::out | std::ios::binary);
    source.save(save_file);
  }

  /// 2 bytes qscheme + 32 bytes of packed data + 2 fp32 scales
  EXPECT_EQ(fileSize(path), kQuantizationInfoBytes + 32u + 8u);
  /// and getMemoryBytes() describes exactly the payload save() wrote
  EXPECT_EQ(source.getMemoryBytes(), 32u + 8u);

  EXPECT_EQ(0, std::remove(path.c_str()));
}

/**
 * @brief Control: QINT8 per-channel scales already round-trip, because
 *        CharTensor sizes its scale region with sizeof(float) on both sides.
 */
TEST(nntrainer_QuantScaleRoundTrip, qint8_per_channel_control_p) {
  const unsigned int w = 8;
  roundTripScales("qint8_per_channel", Tdatatype::QINT8,
                  QScheme::PER_CHANNEL_AFFINE, 1, 1, 1, w, w);
}

/**
 * @brief Control: QINT16 per-tensor scales already round-trip.
 */
TEST(nntrainer_QuantScaleRoundTrip, qint16_per_tensor_control_p) {
  roundTripScales("qint16_per_tensor", Tdatatype::QINT16,
                  QScheme::PER_TENSOR_AFFINE, 1, 2, 3, 4, 1);
}

int main(int argc, char **argv) {
  int result = -1;
  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Failed to initialize google test" << std::endl;
  }

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Failed to run all tests" << std::endl;
  }

  return result;
}

// SPDX-License-Identifier: Apache-2.0
/**
 * @file   unittest_nntrainer_quant_read_qscheme.cpp
 * @date   02 September 2026
 * @brief  Tests that a quantized tensor refuses to read storage whose
 *         quantization scheme differs from the one it was created with
 * @see    https://github.com/nntrainer/nntrainer
 * @author seungbaek <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * @detail The scale factor area of a quantized tensor is sized from the
 * quantization scheme held at construction. read() used to take the scheme
 * stored in the file first and size the read from it, so a tensor created as
 * PER_TENSOR_AFFINE reading PER_CHANNEL_AFFINE storage asked for more bytes
 * than it owns, and the reverse direction silently left the trailing scale
 * factors at zero. Both directions must be rejected instead.
 */

#include <gtest/gtest.h>

#include <fstream>
#include <memory>
#include <string>
#include <variant>

#include <quantizer.h>
#include <tensor.h>
#include <uint_tensor.h>

using nntrainer::QScheme;
using nntrainer::Tensor;
using TensorDim = ml::train::TensorDim;

namespace {

/**
 * @brief A quantized tensor type, by the name used in test traces and the data
 * type it maps to.
 */
struct QuantType {
  std::string name;
  TensorDim::DataType dtype;
};

/**
 * @brief The quantized tensor types reachable through Tensor that have to tell
 * the two schemes apart when reading.
 *
 * @note UINT8 is covered separately, below, because constructing it through
 * Tensor() silently loses the requested scheme. QINT4 is excluded from the
 * round-trip case for the same kind of reason, see that test.
 */
std::vector<QuantType> quantTypes() {
  return {
    {"QINT8", TensorDim::DataType::QINT8},
    {"QINT16", TensorDim::DataType::QINT16},
    {"QINT4", TensorDim::DataType::QINT4},
    {"UINT4", TensorDim::DataType::UINT4},
  };
}

/// @brief the dim every tensor in this file uses
TensorDim testDim(const TensorDim::DataType &dtype) {
  TensorDim dim(1, 2, 4, 8);
  dim.setTensorType({TensorDim::Format::NCHW, dtype});
  return dim;
}

/**
 * @brief Create an allocated quantized tensor of the given scheme.
 *
 * @note dim is (1, 2, 4, 8) so that a per-channel tensor holds strictly more
 * scale factors than a per-tensor one for every type used here, which is what
 * turns a scheme mismatch into a read size mismatch.
 */
Tensor makeTensor(const TensorDim::DataType &dtype, QScheme qscheme) {
  return Tensor(testDim(dtype), true /*alloc_now*/,
                nntrainer::Initializer::NONE, "quant_read_qscheme", qscheme);
}

/**
 * @brief Create an allocated unsigned 8-bit quantized tensor of the given
 * scheme, bypassing the Tensor facade.
 *
 * @note Tensor() forwards the scheme to every other quantized type but hands
 * UINT8 to UInt8Tensor without it, so a UINT8 tensor built through the facade
 * is always PER_TENSOR_AFFINE and no scheme of a file can mismatch it. The
 * read path under test lives in UIntTensor, so it is exercised directly here.
 */
std::unique_ptr<nntrainer::UInt8Tensor> makeUInt8Tensor(QScheme qscheme) {
  return std::make_unique<nntrainer::UInt8Tensor>(
    testDim(TensorDim::DataType::UINT8), true /*alloc_now*/,
    nntrainer::Initializer::NONE, "quant_read_qscheme", qscheme);
}

/**
 * @brief Fill every scale factor with a distinguishable non-zero value.
 *
 * @note Exactly scale_size() factors are written: the area is only that wide,
 * so writing more of them would overrun the tensor for the very reason this
 * file tests. The quantized values are left at zero because setValue() is out
 * of range for the narrower types; the scale factors are what has to survive
 * the round-trip here.
 */
void fillScales(float *scales, size_t scale_size) {
  for (size_t i = 0; i < scale_size; ++i) {
    scales[i] = 1.5f + static_cast<float>(i);
  }
}

void fillScales(Tensor &tensor) {
  float *scales = tensor.getScale<float>();
  EXPECT_NE(scales, nullptr);
  fillScales(scales, tensor.scale_size());
}

std::string filePath(const std::string &name) {
  return "qscheme_" + name + ".bin";
}

/**
 * @brief Write a tensor created with @p created_qscheme out to a file, so that
 * the file itself carries that scheme.
 */
void saveSchemeFile(const TensorDim::DataType &dtype, QScheme created_qscheme,
                    const std::string &path) {
  Tensor source = makeTensor(dtype, created_qscheme);
  fillScales(source);

  std::ofstream out(path, std::ios::out | std::ios::binary);
  EXPECT_TRUE(out.is_open());
  source.save(out);
}

/**
 * @brief Read a whole file through the std::ifstream overload of read().
 */
void readSchemeFile(nntrainer::TensorBase &tensor, const std::string &path) {
  std::ifstream in(path, std::ios::in | std::ios::binary);
  ASSERT_TRUE(in.is_open());
  tensor.read(in);
}

/** @brief overload taking the Tensor facade, which is not a TensorBase */
void readSchemeFile(Tensor &tensor, const std::string &path) {
  std::ifstream in(path, std::ios::in | std::ios::binary);
  ASSERT_TRUE(in.is_open());
  tensor.read(in);
}

/**
 * @brief Read a whole file through the ReadSource overload of read().
 */
void readSchemeSource(Tensor &tensor, const std::string &path) {
  std::ifstream in(path, std::ios::in | std::ios::binary);
  ASSERT_TRUE(in.is_open());

  nntrainer::ReadSource src(&in);
  tensor.read(src);
}

/** @brief both orderings of the two schemes the read path has to tell apart */
std::vector<std::pair<QScheme, QScheme>> schemePairs() {
  return {{QScheme::PER_TENSOR_AFFINE, QScheme::PER_CHANNEL_AFFINE},
          {QScheme::PER_CHANNEL_AFFINE, QScheme::PER_TENSOR_AFFINE}};
}

} // namespace

/**
 * @brief Reading storage saved with a different scheme is rejected, for every
 * quantized tensor type and in both directions.
 *
 * @note Before the guard none of these eight cases passed, and none of them
 * merely reported a failure either: the per-tensor tensor reading per-channel
 * storage wrote (scale_size - 1) * sizeof(float) bytes past its buffer and
 * aborted or crashed the binary (SIGABRT/SIGSEGV), so a run of the previous
 * code does not even reach a gtest verdict. The per-channel tensor reading
 * per-tensor storage stayed silent and left the trailing scale factors at zero,
 * which dequantizes those channels as zero. Asserting the throw pins both.
 */
TEST(nntrainer_QuantReadQScheme, mismatched_qscheme_throws_p) {
  for (const auto &type : quantTypes()) {
    for (const auto &pair : schemePairs()) {
      const QScheme created = pair.first;
      const QScheme stored = pair.second;
      const std::string path = filePath(type.name);

      SCOPED_TRACE(type.name + " created " +
                   nntrainer::qSchemeToString(created) + ", storage " +
                   nntrainer::qSchemeToString(stored));

      saveSchemeFile(type.dtype, stored, path);

      Tensor target = makeTensor(type.dtype, created);
      EXPECT_THROW(readSchemeFile(target, path), std::invalid_argument);

      EXPECT_EQ(std::remove(path.c_str()), 0);
    }
  }
}

/**
 * @brief The unsigned 8-bit tensor rejects a mismatched scheme as well.
 *
 * @note UINT8 is built directly rather than through Tensor, which drops the
 * scheme on this data type; see makeUInt8Tensor().
 */
TEST(nntrainer_QuantReadQScheme, mismatched_qscheme_throws_uint8_p) {
  const std::string path = filePath("UINT8");

  for (const auto &pair : schemePairs()) {
    const QScheme created = pair.first;
    const QScheme stored = pair.second;
    SCOPED_TRACE("UINT8 created " + nntrainer::qSchemeToString(created) +
                 ", storage " + nntrainer::qSchemeToString(stored));

    {
      auto source = makeUInt8Tensor(stored);
      fillScales(static_cast<float *>(source->getScale()),
                 source->scale_size());

      std::ofstream out(path, std::ios::out | std::ios::binary);
      ASSERT_TRUE(out.is_open());
      source->save(out);
    }

    auto target = makeUInt8Tensor(created);
    EXPECT_THROW(readSchemeFile(*target, path), std::invalid_argument);
    EXPECT_EQ(target->q_scheme(), created);
  }

  EXPECT_EQ(std::remove(path.c_str()), 0);
}

/**
 * @brief The error names the tensor, the scheme of the storage and the scheme
 * it was created with, so a failure says which side is wrong.
 */
TEST(nntrainer_QuantReadQScheme, mismatch_message_names_both_schemes_p) {
  const std::string path = filePath("QINT8");
  saveSchemeFile(TensorDim::DataType::QINT8, QScheme::PER_CHANNEL_AFFINE, path);

  Tensor target =
    makeTensor(TensorDim::DataType::QINT8, QScheme::PER_TENSOR_AFFINE);

  try {
    readSchemeFile(target, path);
    FAIL() << "a mismatched scheme must throw";
  } catch (const std::invalid_argument &e) {
    const std::string what = e.what();
    EXPECT_NE(what.find("CharTensor::read"), std::string::npos) << what;
    EXPECT_NE(what.find("PER_CHANNEL_AFFINE"), std::string::npos) << what;
    EXPECT_NE(what.find("PER_TENSOR_AFFINE"), std::string::npos) << what;
  }

  EXPECT_EQ(std::remove(path.c_str()), 0);
}

/**
 * @brief A rejected read leaves the tensor describing the buffer it owns.
 *
 * @note The failure path restores the creation-time scheme, so the object stays
 * consistent afterwards. save() writes the scheme tag and then exactly
 * getMemoryBytes() bytes, so that byte count is what a tensor that still
 * matches its own buffer has to produce. Had the scheme from storage been kept,
 * the same save() would have written past the end of the buffer with an object
 * that looks healthy.
 */
TEST(nntrainer_QuantReadQScheme, rejected_read_keeps_tensor_consistent_p) {
  for (const auto &type : quantTypes()) {
    const std::string path = filePath(type.name);
    SCOPED_TRACE(type.name);

    saveSchemeFile(type.dtype, QScheme::PER_CHANNEL_AFFINE, path);

    Tensor target = makeTensor(type.dtype, QScheme::PER_TENSOR_AFFINE);
    EXPECT_THROW(readSchemeFile(target, path), std::invalid_argument);

    // the scheme still matches the allocation, so the sizing is unchanged
    EXPECT_EQ(target.q_scheme(), QScheme::PER_TENSOR_AFFINE);

    const std::string again = path + ".again";
    {
      std::ofstream out(again, std::ios::out | std::ios::binary);
      ASSERT_TRUE(out.is_open());
      EXPECT_NO_THROW(target.save(out));
      EXPECT_EQ(out.tellp(), static_cast<std::streampos>(
                               sizeof(uint16_t) + target.getMemoryBytes()));
    }

    EXPECT_EQ(std::remove(path.c_str()), 0);
    EXPECT_EQ(std::remove(again.c_str()), 0);
  }
}

/**
 * @brief The ReadSource overload rejects a mismatched scheme as well.
 *
 * @note An ifstream pointer is handed over as the source, which is one of the
 * two alternatives of ReadSource, so this exercises the same entry point a
 * safetensors backed weight read uses.
 *
 * @note Only QINT4 and UINT4 are covered: they are the Tensor-reachable types
 * that override read(ReadSource). CharTensor and ShortTensor do not, so their
 * ReadSource reads fall back to TensorBase::read, which reads bytes() and never
 * looks at the scheme tag at all.
 */
TEST(nntrainer_QuantReadQScheme, mismatched_qscheme_throws_readsource_p) {
  for (const auto &type : quantTypes()) {
    if (type.dtype != TensorDim::DataType::QINT4 &&
        type.dtype != TensorDim::DataType::UINT4)
      continue;

    const std::string path = filePath(type.name);
    SCOPED_TRACE(type.name);

    saveSchemeFile(type.dtype, QScheme::PER_CHANNEL_AFFINE, path);

    Tensor target = makeTensor(type.dtype, QScheme::PER_TENSOR_AFFINE);
    EXPECT_THROW(readSchemeSource(target, path), std::invalid_argument);

    EXPECT_EQ(target.q_scheme(), QScheme::PER_TENSOR_AFFINE);

    EXPECT_EQ(std::remove(path.c_str()), 0);
  }

  // the unsigned 8-bit tensor overrides read(ReadSource) as well
  const std::string path = filePath("UINT8");
  {
    auto source = makeUInt8Tensor(QScheme::PER_CHANNEL_AFFINE);
    fillScales(static_cast<float *>(source->getScale()), source->scale_size());

    std::ofstream out(path, std::ios::out | std::ios::binary);
    ASSERT_TRUE(out.is_open());
    source->save(out);
  }

  auto target = makeUInt8Tensor(QScheme::PER_TENSOR_AFFINE);
  {
    std::ifstream in(path, std::ios::in | std::ios::binary);
    ASSERT_TRUE(in.is_open());
    nntrainer::ReadSource src(&in);
    /// @note the derived declarations of read() hide the defaults declared on
    /// TensorBase, so the trailing arguments are spelled out.
    EXPECT_THROW(target->read(src, 0, false), std::invalid_argument);
  }
  EXPECT_EQ(target->q_scheme(), QScheme::PER_TENSOR_AFFINE);

  EXPECT_EQ(std::remove(path.c_str()), 0);
}

/**
 * @brief Matching schemes still round-trip, so the guard does not reject too
 * much.
 *
 * @note QINT4 is left out: the size it gives its scale factor area on disk
 * disagrees with the one it allocates in memory independently of this guard, so
 * a QINT4 scale round-trip fails with or without it and belongs to that change.
 * UINT8 is exercised directly here for the reason given at makeUInt8Tensor().
 */
TEST(nntrainer_QuantReadQScheme, matching_qscheme_round_trip_p) {
  for (const auto &type : quantTypes()) {
    if (type.dtype == TensorDim::DataType::QINT4)
      continue;

    for (const auto &qscheme :
         {QScheme::PER_TENSOR_AFFINE, QScheme::PER_CHANNEL_AFFINE}) {
      const std::string path = filePath(type.name);
      SCOPED_TRACE(type.name + " " + nntrainer::qSchemeToString(qscheme));

      Tensor source = makeTensor(type.dtype, qscheme);
      fillScales(source);

      {
        std::ofstream out(path, std::ios::out | std::ios::binary);
        ASSERT_TRUE(out.is_open());
        source.save(out);
      }

      Tensor target = makeTensor(type.dtype, qscheme);
      ASSERT_NO_THROW(readSchemeFile(target, path));

      EXPECT_EQ(target.q_scheme(), qscheme);
      // a matching read restores the scale factors, not just the data
      EXPECT_TRUE(source == target);

      EXPECT_EQ(std::remove(path.c_str()), 0);
    }
  }

  for (const auto &qscheme :
       {QScheme::PER_TENSOR_AFFINE, QScheme::PER_CHANNEL_AFFINE}) {
    const std::string path = filePath("UINT8");
    SCOPED_TRACE("UINT8 " + nntrainer::qSchemeToString(qscheme));

    auto source = makeUInt8Tensor(qscheme);
    fillScales(static_cast<float *>(source->getScale()), source->scale_size());

    {
      std::ofstream out(path, std::ios::out | std::ios::binary);
      ASSERT_TRUE(out.is_open());
      source->save(out);
    }

    auto target = makeUInt8Tensor(qscheme);
    ASSERT_NO_THROW(readSchemeFile(*target, path));

    EXPECT_EQ(target->q_scheme(), qscheme);
    EXPECT_TRUE(*source == *target);

    EXPECT_EQ(std::remove(path.c_str()), 0);
  }
}

/**
 * @brief Main() of the test
 */
int main(int argc, char **argv) {
  int result = -1;

  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error during InitGoogleTest" << std::endl;
    return 0;
  }

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS()" << std::endl;
  }

  return result;
}

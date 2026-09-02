// SPDX-License-Identifier: Apache-2.0
/**
 * @file   unittest_nntrainer_bn_qscheme.cpp
 * @date   02 September 2026
 * @brief  Tests that a batch normalization layer keeps one quantization scheme
 *         for its weights across every save/read mode pairing
 * @see    https://github.com/nntrainer/nntrainer
 * @author seungbaek <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * @detail A batch normalization layer hands Layer::save() and Layer::read() a
 * quantized temporary whenever it is asked to store its weights as something
 * other than FP32, because the layer keeps full precision weights while
 * training. That temporary used to be built with the Tensor constructor's
 * default scheme, PER_TENSOR_AFFINE, while the weights themselves come from the
 * tensor pool, which requests every weight as PER_CHANNEL_AFFINE. The two agree
 * on disk only while the BN weight is one element wide, which is the case for
 * the usual axis=1 layout, so the mismatch was invisible. Any BN whose reduced
 * axis is not channel-sized, and any pairing of a save made in one mode with a
 * read made in the other, put the layer on two different schemes.
 *
 * These tests pin the scheme the layer writes, and that all four pairings of
 * the save mode with the read mode round trip, for a BN weight of width one and
 * of width three.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <bn_layer.h>
#include <input_layer.h>
#include <layer.h>
#include <layer_node.h>
#include <model.h>
#include <neuralnet.h>
#include <optimizer.h>
#include <quantizer.h>
#include <tensor.h>
#include <tensor_dim.h>

using namespace nntrainer;
using TensorDim = ml::train::TensorDim;
using DataType = TensorDim::DataType;
using Mode = ml::train::ExecutionMode;

namespace {

/**
 * @brief A one-input model holding a single batch normalization layer.
 *
 * @param axis BN axis, so the weight is channel-sized for 1 and width-sized
 * for 3, which is what varies the number of scale factors.
 * @param mode execution mode the model is finalized in. A model finalized for
 * inference keeps its weights in the quantized type; a model finalized for
 * training forces them to FP32, see BatchNormalizationLayer::finalize.
 */
std::unique_ptr<NeuralNetwork> makeBNModel(unsigned int axis, Mode mode) {
  auto nn = std::make_unique<NeuralNetwork>();

  nn->addLayer(ml::train::layer::Input({"name=input", "input_shape=1:4:3"}));
  nn->addLayer(ml::train::layer::BatchNormalization(
    {"name=bn", "axis=" + std::to_string(axis), "weight_dtype=QINT8"}));
  nn->setOptimizer(ml::train::optimizer::SGD({"learning_rate=0.1"}));
  nn->setProperty({"loss=mse", "batch_size=1"});

  nn->compile(mode);
  nn->initialize(mode);
  return nn;
}

/** @brief the batch normalization node of a model built by makeBNModel */
LayerNode *bnNode(NeuralNetwork &nn) {
  for (auto &ln : nn.getFlatGraph()) {
    if (ln->getType() == BatchNormalizationLayer::type)
      return ln.get();
  }
  ADD_FAILURE() << "no batch normalization layer in the graph";
  return nullptr;
}

/**
 * @brief Save the BN weights under @p save_mode the way NeuralNetwork::save
 * does, so the mode and the target data type are chosen independently of the
 * mode the model was finalized in.
 */
void saveBN(LayerNode &bn, const std::string &path, Mode save_mode) {
  std::ofstream out(path, std::ios::out | std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(out.is_open());
  bn.save(out, false, save_mode, DataType::QINT8);
}

/** @brief Read the BN weights back under @p read_mode. */
void readBN(LayerNode &bn, const std::string &path, Mode read_mode) {
  std::ifstream in(path, std::ios::in | std::ios::binary);
  ASSERT_TRUE(in.is_open());
  bn.read(in, false, read_mode);
}

/**
 * @brief The quantization scheme tag stored at the front of the first weight in
 * a file saved by saveBN().
 *
 * @note This is the one observable that tells the two schemes apart from
 * outside the tensor: BatchNormalizationLayer writes each weight through
 * Tensor::save(), which begins with the uint16_t scheme written by
 * CharTensor::save_quantization_info().
 */
QScheme firstScheme(const std::string &path) {
  std::ifstream in(path, std::ios::in | std::ios::binary);
  EXPECT_TRUE(in.is_open());
  uint16_t tag = 0xFFFF;
  in.read(reinterpret_cast<char *>(&tag), sizeof(tag));
  return static_cast<QScheme>(tag);
}

/// @brief the file every case in this suite writes, removed after each case
const std::string kPath = "unittest_bn_qscheme.bin";

} // namespace

/**
 * @brief Saving and reading in inference mode round trips the weights.
 *
 * @note Inference mode is the pairing where the layer does not build a
 * temporary at all: the quantized weight is written and read directly, so this
 * is the reference the other three pairings have to match. The values survive
 * exactly because both sides use the same tensor.
 */
TEST(nntrainer_BNQScheme, inference_round_trip_p) {
  for (const unsigned int axis : {1u, 3u}) {
    SCOPED_TRACE("axis " + std::to_string(axis));
    std::remove(kPath.c_str());

    auto nn = makeBNModel(axis, Mode::INFERENCE);
    LayerNode &bn = *bnNode(*nn);

    Tensor &weight = bn.getWeight(0);
    const std::vector<int8_t> expected(weight.getData<int8_t>(),
                                       weight.getData<int8_t>() +
                                         weight.getDim().getDataLen());

    ASSERT_NO_THROW(saveBN(bn, kPath, Mode::INFERENCE));
    ASSERT_NO_THROW(readBN(bn, kPath, Mode::INFERENCE));

    const std::vector<int8_t> actual(weight.getData<int8_t>(),
                                     weight.getData<int8_t>() +
                                       weight.getDim().getDataLen());
    EXPECT_EQ(expected, actual);

    std::remove(kPath.c_str());
  }
}

/**
 * @brief Saving and reading in training mode round trips without a scheme
 * mismatch.
 *
 * @note Both sides go through the layer's quantized temporary. The values are
 * not compared here: training keeps the weights in FP32, so this path converts
 * on the way in and on the way out and is only required to not lose the tensor.
 */
TEST(nntrainer_BNQScheme, training_round_trip_p) {
  for (const unsigned int axis : {1u, 3u}) {
    SCOPED_TRACE("axis " + std::to_string(axis));
    std::remove(kPath.c_str());

    auto nn = makeBNModel(axis, Mode::INFERENCE);
    LayerNode &bn = *bnNode(*nn);

    ASSERT_NO_THROW(saveBN(bn, kPath, Mode::TRAIN));
    ASSERT_NO_THROW(readBN(bn, kPath, Mode::TRAIN));

    std::remove(kPath.c_str());
  }
}

/**
 * @brief A file written in one mode is readable in the other.
 *
 * @note This is the pairing the scheme guard in the quantized tensors rejects:
 * the inference side reads and writes the weight itself, which the tensor pool
 * made PER_CHANNEL_AFFINE, while the training side used to build its temporary
 * with the constructor default. A BN weight that is one element wide hid the
 * difference because both schemes then need exactly one scale factor; axis=3
 * gives a weight three elements wide, where the two schemes ask for different
 * numbers of bytes.
 */
TEST(nntrainer_BNQScheme, cross_mode_round_trip_p) {
  for (const unsigned int axis : {1u, 3u}) {
    SCOPED_TRACE("axis " + std::to_string(axis));

    for (const auto &pair : std::vector<std::pair<Mode, Mode>>{
           {Mode::INFERENCE, Mode::TRAIN}, {Mode::TRAIN, Mode::INFERENCE}}) {
      std::remove(kPath.c_str());
      SCOPED_TRACE(
        "saved " +
        std::string(pair.first == Mode::TRAIN ? "TRAIN" : "INFERENCE") +
        ", read " +
        std::string(pair.second == Mode::TRAIN ? "TRAIN" : "INFERENCE"));

      auto nn = makeBNModel(axis, Mode::INFERENCE);
      LayerNode &bn = *bnNode(*nn);

      ASSERT_NO_THROW(saveBN(bn, kPath, pair.first));
      ASSERT_NO_THROW(readBN(bn, kPath, pair.second));

      std::remove(kPath.c_str());
    }
  }
}

/**
 * @brief Every mode pairing writes the same scheme, the one the weights carry.
 *
 * @note Pinning the tag rather than only the absence of a throw is what keeps
 * the two sides from drifting apart again silently: while a BN weight is one
 * element wide both schemes need one scale factor, so a regression back to the
 * constructor default would still round trip on axis=1 and only the tag shows
 * it. PER_CHANNEL_AFFINE is what the tensor pool requests for every weight, see
 * TensorPool::request.
 */
TEST(nntrainer_BNQScheme, all_pairings_write_the_weight_scheme_p) {
  for (const unsigned int axis : {1u, 3u}) {
    SCOPED_TRACE("axis " + std::to_string(axis));

    auto nn = makeBNModel(axis, Mode::INFERENCE);
    LayerNode &bn = *bnNode(*nn);
    /// the scheme of the weight itself, which the layer has to write with
    const QScheme weight_scheme = bn.getWeight(0).q_scheme();
    ASSERT_EQ(weight_scheme, QScheme::PER_CHANNEL_AFFINE);

    for (const auto &save_mode : {Mode::TRAIN, Mode::INFERENCE}) {
      std::remove(kPath.c_str());
      SCOPED_TRACE(std::string("saved under ") +
                   (save_mode == Mode::TRAIN ? "TRAIN" : "INFERENCE"));

      ASSERT_NO_THROW(saveBN(bn, kPath, save_mode));
      EXPECT_EQ(firstScheme(kPath), weight_scheme)
        << "the scheme written by save(mode="
        << (save_mode == Mode::TRAIN ? "TRAIN" : "INFERENCE")
        << ") must be the scheme of the weight";

      std::remove(kPath.c_str());
    }
  }
}

/**
 * @brief A model finalized for training also round trips both modes.
 *
 * @note Here the weights themselves are FP32, since the layer needs full
 * precision to train, while the file holds the quantized type. This is the
 * configuration in which asking the weight for its quantization scheme is not
 * even defined, so the scheme the temporary is built with cannot be taken from
 * it at this point.
 */
TEST(nntrainer_BNQScheme, train_finalized_model_round_trip_p) {
  for (const unsigned int axis : {1u, 3u}) {
    SCOPED_TRACE("axis " + std::to_string(axis));
    std::remove(kPath.c_str());

    auto nn = makeBNModel(axis, Mode::TRAIN);
    LayerNode &bn = *bnNode(*nn);
    ASSERT_EQ(bn.getWeight(0).getDataType(), DataType::FP32);

    ASSERT_NO_THROW(saveBN(bn, kPath, Mode::TRAIN));
    ASSERT_NO_THROW(readBN(bn, kPath, Mode::TRAIN));

    std::remove(kPath.c_str());
  }
}

/**
 * @brief A training-finalized model reads a file written in inference mode.
 *
 * @note This is the pairing the cases above leave untested: the file comes from
 * the inference path, which writes the PER_CHANNEL_AFFINE tag of the pool
 * weight, while the model reading it was finalized for training and holds its
 * weights as FP32. The read rebuilds the quantized temporary with the same
 * PER_CHANNEL_AFFINE this commit gives the save side, so the two agree and the
 * scheme guard stays silent. It pins the intended half of the guard: a file
 * that carries the weight's real scheme is accepted even when the reader is a
 * training model. Rejecting a file that carries the old PER_TENSOR tag cannot
 * be pinned here, because no path in this binary writes that tag any more; it
 * is a cross-binary property and is recorded in the commit message instead.
 */
TEST(nntrainer_BNQScheme, train_finalized_reads_inference_saved_p) {
  for (const unsigned int axis : {1u, 3u}) {
    SCOPED_TRACE("axis " + std::to_string(axis));
    std::remove(kPath.c_str());

    /// a file written in inference mode carries the weight's PER_CHANNEL scheme
    auto writer = makeBNModel(axis, Mode::INFERENCE);
    ASSERT_NO_THROW(saveBN(*bnNode(*writer), kPath, Mode::INFERENCE));
    ASSERT_EQ(firstScheme(kPath), QScheme::PER_CHANNEL_AFFINE);

    /// a training model, whose weights are FP32, reads that file back
    auto reader = makeBNModel(axis, Mode::TRAIN);
    LayerNode &bn = *bnNode(*reader);
    ASSERT_EQ(bn.getWeight(0).getDataType(), DataType::FP32);
    ASSERT_NO_THROW(readBN(bn, kPath, Mode::TRAIN));

    std::remove(kPath.c_str());
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

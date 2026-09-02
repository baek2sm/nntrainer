// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file unittest_layers_convolution2d_nhwc.cpp
 * @date 2 September 2026
 * @brief Conv2d channel last (NHWC) test
 * @see	https://github.com/nntrainer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug No known bugs except for NYI items
 *
 * @detail These cases drive Conv2DLayer directly instead of through a golden
 * file so that the two layouts can be compared against each other: one layer
 * setup, fed a logically identical input and holding logically identical
 * weights, run once channel first and once channel last. Everything is filled
 * and read by logical index, so a physical layout difference shows up as a
 * value difference instead of being hidden by the fill order.
 *
 * A channel last convolution sums the same taps in another order than the
 * channel first path does, so the two agree within FP rounding rather than bit
 * for bit. The bound below is measured over the shapes registered here.
 *
 * Scope: channel last is supported by forwarding() only. Backwarding is
 * refused by name and Conv2DNhwcBackward pins that, including the shape whose
 * dimensions happen to line up without the guard.
 */
#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <base_properties.h>
#include <conv2d_layer.h>
#include <layer_context.h>
#include <nntrainer_test_util.h>
#include <tensor.h>
#include <var_grad.h>
#include <weight.h>

using namespace nntrainer;

namespace {

/**
 * @brief A Conv2DLayer with its weights, input and output tensors built for one
 * tensor format.
 */
class ConvHarness {
public:
  ConvHarness(const std::vector<std::string> &props, const std::string &format,
              const std::string &input_shape, unsigned int batch) :
    layer(nntrainer::createLayer<nntrainer::Conv2DLayer>()) {

    layer->setProperty(props);

    auto format_enum =
      str_converter<enum_class_prop_tag,
                    nntrainer::TensorFormatInfo>::from_string(format);

    TensorDim in_dim(input_shape, format_enum,
                     ml::train::TensorDim::DataType::FP32);
    in_dim.batch(batch);

    init_context = std::make_unique<InitLayerContext>(
      std::vector<TensorDim>{in_dim}, std::vector<bool>{true}, false,
      "conv_nhwc", "", 0.0, std::array<std::string, 3>{format, "fp32", "fp32"},
      1.0, ml::train::ExecutionMode::TRAIN);
    layer->finalize(*init_context);

    for (const auto &spec : init_context->getWeightsSpec()) {
      weights.emplace_back(spec, true);
      weights.back().getGradientRef().setZero();
    }

    for (const auto &dim : init_context->getInputDimensions())
      inputs.emplace_back(dim, Initializer::NONE, true, true, "input");

    for (const auto &spec : init_context->getOutSpecs())
      outputs.emplace_back(spec.variable_spec.dim, Initializer::NONE, true,
                           true, "output");

    for (const auto &spec : init_context->getTensorsSpec())
      tensors.emplace_back(spec, true);

    /// the column matrix is scratch, but a channel last run reads it before it
    /// writes the taps that fall inside the input, so start it off dirty: an
    /// uninitialized buffer is as close as a unit test gets to the memory the
    /// allocator hands over, and garbage there must not reach the output.
    for (auto &t : tensors)
      t.getVariableRef().setRandNormal();

    fillAll();
  }

  ConvHarness(const ConvHarness &) = delete;
  ConvHarness &operator=(const ConvHarness &) = delete;

  void forward(bool training = false) {
    auto rc = makeRunContext();
    layer->forwarding(rc, training);
  }

  /**
   * @brief Fill the incoming derivative of every output, elementwise in logical
   * order. That gradient is what calcDerivative() and calcGradient() read.
   */
  void fillIncomingGradient(std::function<float(unsigned int)> value) {
    unsigned int idx = 0;
    for (auto &out : outputs) {
      Tensor &g = out.getGradientRef();
      for (size_t b = 0; b < g.batch(); ++b)
        for (size_t c = 0; c < g.channel(); ++c)
          for (size_t h = 0; h < g.height(); ++h)
            for (size_t w = 0; w < g.width(); ++w)
              g.setValue(b, c, h, w, value(idx++));
    }
  }

  void calcDerivative() {
    auto rc = makeRunContext();
    layer->calcDerivative(rc);
  }

  void calcGradient() {
    auto rc = makeRunContext();
    layer->calcGradient(rc);
  }

  /// clear every input gradient, so that a refused calcDerivative() can be
  /// checked to have written nothing at all
  void zeroInputGrad() {
    for (auto &in : inputs)
      in.getGradientRef().setZero();
  }

  /// input gradient flattened in logical (batch, channel, height, width) order
  std::vector<float> readInputGrad() const {
    std::vector<float> values;
    for (const auto &in : inputs)
      append(in.getGradient(), values);
    return values;
  }

  /// the input and the weights, for a case whose expectation is computed by
  /// hand rather than read from the other layout
  Tensor &inputRef(unsigned int idx = 0) {
    return inputs[idx].getVariableRef();
  }

  Tensor &weightRef(unsigned int idx) { return weights[idx].getVariableRef(); }

  /// overwrite the scratch the layer requested, in logical order, with values
  /// that cannot survive a re-forward untouched
  void poisonScratch() {
    for (auto &t : tensors) {
      Tensor &var = t.getVariableRef();
      unsigned int idx = 0;
      for (size_t b = 0; b < var.batch(); ++b)
        for (size_t c = 0; c < var.channel(); ++c)
          for (size_t h = 0; h < var.height(); ++h)
            for (size_t w = 0; w < var.width(); ++w)
              var.setValue(b, c, h, w, 3.5f + (idx++) % 7);
    }
  }

  /**
   * @brief Emulate a runtime batch change the way NetworkGraph::setBatchSize
   * does: release the memory, resize every tensor, hand the new batch to the
   * layer, then allocate again. The order matters, a batch is only updatable
   * while the tensor is not allocated.
   */
  void setBatch(unsigned int batch) {
    forEach([](Var_Grad &vg) {
      vg.getVariableRef().deallocate();
      vg.getGradientRef().deallocate();
    });

    /// the graph resizes the input and the output specs itself, the layer has
    /// to resize the tensor it requested in finalize()
    for (auto &in : inputs)
      in.setBatchSize(batch);
    for (auto &out : outputs)
      out.setBatchSize(batch);

    auto rc = makeRunContext();
    layer->setBatch(rc, batch);

    forEach([](Var_Grad &vg) {
      vg.getVariableRef().allocate();
      vg.getGradientRef().allocate();
      vg.getGradientRef().setZero();
    });

    /// reallocation does not initialize, and the point of the test is the
    /// values, so fill the grown input again
    fillInputs();
  }

  /// output values flattened in logical (batch, channel, height, width) order
  std::vector<float> readOutput() const {
    std::vector<float> values;
    for (const auto &out : outputs)
      append(out.getVariable(), values);
    return values;
  }

  TensorDim getOutputDim() const { return outputs.front().getDim(); }

  /// how many tensors the layer asked the memory planner for
  size_t getNumRequestedTensors() const { return tensors.size(); }

  /// batch of the column matrix the layer requested in finalize(). that is the
  /// only tensor the layer requests, and only on the channel last path.
  unsigned int getScratchBatch() const {
    return tensors.front().getDim().batch();
  }

private:
  RunLayerContext makeRunContext() {
    auto view = [](auto &container) {
      using value_t = typename std::decay_t<decltype(container)>::value_type;
      std::vector<std::remove_cv_t<value_t> *> ret;
      ret.reserve(container.size());
      for (auto &e : container)
        ret.push_back(&e);
      return ret;
    };

    return RunLayerContext("conv_nhwc", true, 0.0f, false, 1.0, nullptr, false,
                           view(weights), view(inputs), view(outputs),
                           view(tensors));
  }

  /**
   * @brief Fill the input and the weights by logical index.
   *
   * Filling by logical index is what makes the comparison across the two
   * formats meaningful: both runs then hold the same values at the same logical
   * coordinates, while the stored order differs. Every bias element is
   * distinct, so a channel mix-up in the bias add cannot pass either.
   */
  void fillAll() {
    fillInputs();
    /// weights[0] is the filter, weights[1] the bias when bias is enabled
    for (size_t i = 0; i < weights.size(); ++i)
      fill(weights[i].getVariableRef(), i == 0 ? 0.125f : 0.5f);
  }

  /// refill the input, e.g. after a batch growth left the new rows undefined
  void fillInputs() {
    for (auto &in : inputs)
      fill(in.getVariableRef(), 0.25f);
  }

  /**
   * @brief Fill by logical index with the shared deterministic random vector so
   * that the two formats hold identical values.
   *
   * The draws are advanced in logical order from the same fixed seed, so the
   * channel first and the channel last run put the same number at the same
   * coordinate. They are ordinary fractions rather than exactly representable
   * ones, so both paths round and cancel the way they would around trained
   * weights instead of agreeing by construction.
   */
  static void fill(Tensor &var, float scale) {
    auto values = generate_random_vector<float>(var.size());
    size_t idx = 0;
    for (size_t b = 0; b < var.batch(); ++b)
      for (size_t c = 0; c < var.channel(); ++c)
        for (size_t h = 0; h < var.height(); ++h)
          for (size_t w = 0; w < var.width(); ++w)
            var.setValue(b, c, h, w, values[idx++] * scale);
  }

  /// run over every non weight tensor pack of the harness
  template <typename F> void forEach(F f) {
    for (auto &pack : {std::ref(inputs), std::ref(outputs), std::ref(tensors)})
      for (auto &vg : pack.get())
        f(vg);
  }

  static void append(const Tensor &var, std::vector<float> &values) {
    for (size_t b = 0; b < var.batch(); ++b)
      for (size_t c = 0; c < var.channel(); ++c)
        for (size_t h = 0; h < var.height(); ++h)
          for (size_t w = 0; w < var.width(); ++w)
            values.push_back(var.getValue(b, c, h, w));
  }

  std::unique_ptr<Layer> layer;
  std::unique_ptr<InitLayerContext> init_context;
  std::vector<Weight> weights;
  std::vector<Var_Grad> inputs, outputs, tensors;
};

/// largest absolute elementwise difference, NaN propagates so it never passes
static float maxAbsErr(const std::vector<float> &a,
                       const std::vector<float> &b) {
  EXPECT_EQ(a.size(), b.size());
  float err = 0.f;
  for (size_t i = 0; i < std::min(a.size(), b.size()); ++i)
    err = std::max(err, std::abs(a[i] - b[i]));
  return err;
}

/**
 * @brief Measured channel last vs channel first bound over every shape below.
 *
 * FP32 only, so the difference is summation order: the channel first path
 * reduces over the columns of its column matrix, the channel last path over the
 * rows of its operand. Over the registered shapes the largest difference is
 * 5.96e-08 and the next 2.98e-08; both are exact powers of two, which is what a
 * single extra rounding of a result of order one looks like. The bound leaves
 * about 16 times headroom on that, while anything actually wrong with the
 * layout is orders of magnitude bigger: reading the filter in another tap order
 * moves an output by O(0.1) at least, and a stale padded tap or a channel
 * mix-up likewise. Do not raise this bound to make a failure go away.
 */
static constexpr float kNhwcNchwMaxAbsErr = 1e-6f;

struct ConvCase {
  std::vector<std::string> props;
  std::string input_shape;
};

std::string describe(const ConvCase &c) {
  std::string s = c.input_shape;
  for (const auto &p : c.props)
    s += " " + p;
  return s;
}

} // namespace

class Conv2DNhwcParity : public ::testing::TestWithParam<ConvCase> {};

/**
 * @brief A channel last convolution matches the channel first convolution of
 * the same setup within kNhwcNchwMaxAbsErr.
 */
TEST_P(Conv2DNhwcParity, nhwc_matches_nchw) {
  const ConvCase &c = GetParam();
  /// defaults first, then the case: loadProperties() walks the vector in order
  /// and consumes every entry, so a key given twice is set twice and the last
  /// setting is what stands. The case entries therefore override the defaults
  /// below, which is what case 5 (kernel_size=3,2) relies on to be a non square
  /// kernel at all. Conv2DNhwcKernelShape pins that reading without depending
  /// on the order of a duplicate key.
  std::vector<std::string> props{"filters=3", "kernel_size=2,2"};
  for (const auto &p : c.props)
    props.push_back(p);

  ConvHarness nchw(props, "nchw", c.input_shape, 2);
  ConvHarness nhwc(props, "nhwc", c.input_shape, 2);

  /// the format has to survive finalize, otherwise this compared two nchw runs
  EXPECT_EQ(nhwc.getOutputDim().getFormat(),
            ml::train::TensorDim::Format::NHWC);
  EXPECT_EQ(nchw.getOutputDim().getFormat(),
            ml::train::TensorDim::Format::NCHW);

  EXPECT_NO_THROW(nchw.forward());
  EXPECT_NO_THROW(nhwc.forward());

  const auto nchw_out = nchw.readOutput();
  const auto nhwc_out = nhwc.readOutput();
  EXPECT_GT(nchw_out.size(), 0u);
  EXPECT_LE(maxAbsErr(nchw_out, nhwc_out), kNhwcNchwMaxAbsErr)
    << "case: " << describe(c);
}

INSTANTIATE_TEST_SUITE_P(
  Conv2DNhwcShapes, Conv2DNhwcParity,
  ::testing::Values(
    /// valid padding
    ConvCase{{"padding=valid"}, "3:5:5"},
    /// explicit symmetric padding, exercises the taps outside the input that
    /// the gather skips and therefore the zeroing of the column matrix
    ConvCase{{"padding=1,1"}, "3:5:5"},
    /// asymmetric custom padding, exercises the top/left vs bottom/right split
    ConvCase{{"padding=0,1,2,3"}, "3:5:5"},
    /// "same" padding on an even kernel, the uneven split case
    ConvCase{{"padding=same"}, "3:4:4"},
    /// stride 2
    ConvCase{{"padding=valid", "stride=2,2"}, "3:6:6"},
    /// non square kernel and stride
    ConvCase{{"padding=1,1", "kernel_size=3,2", "stride=2,1"}, "2:7:6"},
    /// dilation, where a contiguous copy of an input row is not valid
    ConvCase{{"padding=2,2", "dilation=2,2"}, "3:7:7"},
    /// single input channel
    ConvCase{{"padding=1,1"}, "1:4:4"},
    /// odd channel count, so the channel run is not a power of two
    ConvCase{{"padding=1,1", "filters=5"}, "7:5:6"}));

/**
 * @brief A batch larger than the worker count through the channel last path.
 *
 * The column matrix is one layer private tensor, so every batch slice and every
 * worker thread shares it. forwardingChannelLast() hands each worker a range of
 * batches, so a batch that exceeds the worker count, which the default
 * nntr-num-threads of 4 does here, leaves at least one worker walking several
 * slices of that one buffer in turn. That is the shape this case adds over the
 * parity cases: a slice that zeroed or gathered outside its own rows lands on a
 * slice that the same worker touches next, which is not what an out of range
 * write between two workers looks like. The bound is the same one the parity
 * cases use, and it does catch this: zeroing slice 0 for every batch instead of
 * the slice at hand moves an output by 3.3e-02.
 */
TEST(Conv2DNhwcStaleRows, batch_gt_one_matches_nchw) {
  const unsigned int batch = 5;
  std::vector<std::string> props{"filters=3", "kernel_size=2,2", "padding=1,1"};

  ConvHarness nchw(props, "nchw", "3:5:5", batch);
  ConvHarness nhwc(props, "nhwc", "3:5:5", batch);
  EXPECT_NO_THROW(nchw.forward());
  EXPECT_NO_THROW(nhwc.forward());

  EXPECT_LE(maxAbsErr(nchw.readOutput(), nhwc.readOutput()),
            kNhwcNchwMaxAbsErr);
}

/**
 * @brief Only the channel last path asks for the column matrix.
 *
 * The column matrix is planned memory, so it is a footprint the channel first
 * path used to not have. It must not gain one: a graph that never runs channel
 * last still must not reserve the scratch, and the channel last path must
 * reserve exactly one tensor for it rather than one per call.
 */
TEST(Conv2DNhwcScratch, requested_only_by_channel_last) {
  std::vector<std::string> props{"filters=3", "kernel_size=2,2", "padding=1,1"};
  ConvHarness nchw(props, "nchw", "3:5:5", 2);
  ConvHarness nhwc(props, "nhwc", "3:5:5", 2);

  EXPECT_EQ(nchw.getNumRequestedTensors(), 0u);
  EXPECT_EQ(nhwc.getNumRequestedTensors(), 1u);
  /// sized to the batch, which is what setBatch has to keep true
  EXPECT_EQ(nhwc.getScratchBatch(), 2u);
}

/**
 * @brief The channel last column matrix follows a runtime batch change.
 *
 * The layer is finalized at batch 1 and then forwarded at batch 3, which is
 * what happens when a model initialized at one batch is run with a larger one.
 * The column matrix keeps the batch it was requested with unless Conv2DLayer
 * overrides setBatch, and the batch 3 forward then slices past its planned
 * storage, which Tensor::getBatchSlice rejects with "Creating shared tensor of
 * size bigger than tensor memory". So what is checked is that the layer resized
 * the tensor it requested, that nothing is thrown, and that the grown run still
 * matches the channel first path.
 */
TEST(Conv2DNhwcSetBatch, grow_batch_after_finalize) {
  std::vector<std::string> props{"filters=3", "kernel_size=2,2", "padding=1,1"};
  ConvHarness nhwc(props, "nhwc", "3:5:5", 1);
  ASSERT_NO_THROW(nhwc.forward());
  ASSERT_EQ(nhwc.getScratchBatch(), 1u);

  ASSERT_NO_THROW(nhwc.setBatch(3));
  EXPECT_EQ(nhwc.getScratchBatch(), 3u);
  ASSERT_NO_THROW(nhwc.forward());

  ConvHarness nchw(props, "nchw", "3:5:5", 3);
  nchw.forward();
  EXPECT_EQ(nhwc.getOutputDim().batch(), 3u);
  EXPECT_LE(maxAbsErr(nchw.readOutput(), nhwc.readOutput()),
            kNhwcNchwMaxAbsErr);
}

/**
 * @brief A second forward over the same column matrix ignores whatever the
 * first one left behind, and so does one after the batch has moved.
 *
 * Conv2DNhwcStaleRows.batch_gt_one_matches_nchw only ever runs one forward, so
 * nothing it reads back can come from an earlier forward. This case makes the
 * contents stale on purpose: it forwards, overwrites the whole matrix with
 * values that cannot be produced by a gather, and forwards again. The second
 * run has to reproduce the first.
 *
 * That the other case is not vacuous is not the allocator's doing either: the
 * ConvHarness constructor dirties every requested tensor with
 * setRandNormal(), which is what turns a missing col.setZero() into a wrong
 * output rather than an unnoticed pass against a buffer that happened to start
 * at zero.
 *
 * The matrix is then walked through a batch change, 2 to 3 to 2 to 3, which is
 * what a model run with another batch in between looks like. The batch slices
 * are views into the one scratch buffer, but ConvHarness::setBatch() releases
 * and reallocates that buffer the way NetworkGraph::setBatchSize does, so a
 * pattern written before a batch change may not survive it: what a
 * deallocate hands back to the allocator, and what the next allocate hands
 * out, is unspecified. What the regrown rows hold either way is memory the
 * forward never wrote, which it has to keep out just the same, and the run
 * at batch 3 is the one reading rows a batch 2 forward never wrote. Since
 * the poison may not have survived the realloc, the body dirties the matrix
 * again after the last one: safe either way, the final run then reads a
 * buffer known to hold non gather values.
 */
TEST(Conv2DNhwcStaleRows, reused_scratch_is_not_read) {
  std::vector<std::string> props{"filters=3", "kernel_size=2,2", "padding=1,1"};
  ConvHarness nhwc(props, "nhwc", "3:5:5", 2);

  ASSERT_NO_THROW(nhwc.forward());
  const auto at_2 = nhwc.readOutput();
  ASSERT_GT(at_2.size(), 0u);

  nhwc.poisonScratch();
  ASSERT_NO_THROW(nhwc.forward());
  const auto at_2_again = nhwc.readOutput();

  ASSERT_EQ(at_2.size(), at_2_again.size());
  EXPECT_LE(maxAbsErr(at_2, at_2_again), kNhwcNchwMaxAbsErr)
    << "the second forward read rows the first one left in the column matrix";

  /// grow past the batch the harness was built at: the row the batch 2 forward
  /// never gathered into has to be zeroed rather than read as it is
  ASSERT_NO_THROW(nhwc.setBatch(3));
  ASSERT_NO_THROW(nhwc.forward());
  const auto at_3 = nhwc.readOutput();

  ASSERT_NO_THROW(nhwc.setBatch(2));
  ASSERT_NO_THROW(nhwc.forward());
  ASSERT_NO_THROW(nhwc.setBatch(3));
  EXPECT_EQ(nhwc.getScratchBatch(), 3u);
  /// dirty it again, after the last reallocation: whether the poison from
  /// before it survived the deallocate and realloc is unspecified, so this is
  /// what makes the batch 3 run read a buffer known to hold values a gather
  /// would never write, safe either way
  nhwc.poisonScratch();
  ASSERT_NO_THROW(nhwc.forward());

  /// a harness that was batch 3 from the start, filled with the same
  /// deterministic values, is the reference the walk above has to land on
  ConvHarness at_3_fresh(props, "nhwc", "3:5:5", 3);
  ASSERT_NO_THROW(at_3_fresh.forward());
  EXPECT_LE(maxAbsErr(at_3_fresh.readOutput(), nhwc.readOutput()),
            kNhwcNchwMaxAbsErr)
    << "the forward after a shrink and a regrow read stale rows";
  /// and it is still the same run as before the walk, not a different wrong one
  EXPECT_LE(maxAbsErr(at_3, nhwc.readOutput()), kNhwcNchwMaxAbsErr);
}

/**
 * @brief Channel last backwarding is refused by name.
 *
 * forwarding() is the only entry point with a channel last path. What happens
 * without an explicit guard depends on a dimension check that is not about the
 * layout: Tensor::dot flattens the leading dimensions according to the format,
 * so for an input whose channels times taps equals out_h times out_w the
 * channel first backward shapes happen to fit, the dot runs, col2im walks the
 * transposed product and the outgoing gradient is written without an error.
 * 4 channels with a 3x3 valid kernel over 8:8 is exactly that shape: 4 * 9 taps
 * equals the 6 * 6 outputs. The other shape here is the common 3 channels over
 * 5:5, where the dimensions do not line up, so that both outcomes of that check
 * are pinned to the same answer.
 */
TEST(Conv2DNhwcBackward, channel_last_backward_is_refused) {
  struct BwdCase {
    std::vector<std::string> props;
    std::string input_shape;
  };
  const std::vector<BwdCase> cases{
    /// taps * channels == out_h * out_w, the shape that passes the dot
    BwdCase{{"filters=4", "kernel_size=3,3", "padding=valid"}, "4:8:8"},
    BwdCase{{"filters=3", "kernel_size=2,2", "padding=1,1"}, "3:5:5"}};

  for (const auto &c : cases) {
    /// the training execution mode is what lets a forward only request reach
    /// the backward entry points at all
    ConvHarness nhwc(c.props, "nhwc", c.input_shape, 2);
    ASSERT_NO_THROW(nhwc.forward(true));
    nhwc.fillIncomingGradient([](unsigned int) { return 1.0f; });
    nhwc.zeroInputGrad();

    EXPECT_THROW(nhwc.calcDerivative(), std::runtime_error)
      << "case: " << c.input_shape << " " << describe({c.props, ""});
    EXPECT_THROW(nhwc.calcGradient(), std::runtime_error)
      << "case: " << c.input_shape << " " << describe({c.props, ""});

    /// refused rather than half computed: nothing reached the input gradient
    for (float v : nhwc.readInputGrad())
      EXPECT_EQ(v, 0.0f) << "case: " << c.input_shape;

    /// the channel first path of the same setup still backwardes, and this is
    /// also the control that says the harness is wired to the backward entry
    /// points at all: if the gradients were not connected, the two calls above
    /// would have nothing to do and the refusal would prove nothing
    ConvHarness nchw(c.props, "nchw", c.input_shape, 2);
    ASSERT_NO_THROW(nchw.forward(true));
    nchw.fillIncomingGradient([](unsigned int) { return 1.0f; });
    nchw.zeroInputGrad();
    EXPECT_NO_THROW(nchw.calcDerivative());
    EXPECT_NO_THROW(nchw.calcGradient());

    bool any = false;
    for (float v : nchw.readInputGrad())
      any = any || v != 0.0f;
    EXPECT_TRUE(any)
      << "case: " << c.input_shape
      << ": the channel first backward wrote no gradient either, "
         "so the harness is not driving backwarding";
  }
}

/**
 * @brief A channel last convolution matches a hand computed convolution, not
 * only the channel first one.
 *
 * Every other case here compares the two layouts to each other, and the two
 * share the weight request, the dot and the flatten dot helper, so a mistake
 * the two share would agree with itself. This case pins absolute values
 * computed by hand, so the layout itself is pinned and not just its symmetry.
 *
 * Setup: one input channel, one filter, 2x2 kernel, valid padding, bias on,
 * input 1:3:3, so two outputs per row and four taps per output. The filter is
 * requested as (filters, channel, kh, kw), the input row by row.
 *
 *   input                     filter w  = [[1, 2],   bias b = 1
 *   1 2 3                                     [3, 4]]
 *   4 5 6
 *   7 8 9
 *
 * Each output is the four taps under the kernel plus the bias, so with the
 * kernel over the top left corner the taps are 1, 2, 4, 5 and
 *
 *   out[0][0] = 1*1 + 2*2 + 4*3 + 5*4 + 1 = 1 + 4 + 12 + 20 + 1 = 38
 *   out[0][1] = 2*1 + 3*2 + 5*3 + 6*4 + 1 = 2 + 6 + 15 + 24 + 1 = 48
 *   out[1][0] = 4*1 + 5*2 + 7*3 + 8*4 + 1 = 4 + 10 + 21 + 32 + 1 = 68
 *   out[1][1] = 5*1 + 6*2 + 8*3 + 9*4 + 1 = 5 + 12 + 24 + 36 + 1 = 78
 *
 * The taps are integers and the sums are small, so nothing rounds and the
 * expectation is exact.
 */
TEST(Conv2DNhwcGolden, hand_computed_single_channel) {
  std::vector<std::string> props{"filters=1", "kernel_size=2,2",
                                 "padding=valid"};
  ConvHarness nhwc(props, "nhwc", "1:3:3", 1);

  const float input[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  const float filter[2][2] = {{1, 2}, {3, 4}};
  const float bias = 1.0f;

  Tensor &in = nhwc.inputRef();
  Tensor &w = nhwc.weightRef(0);
  Tensor &b = nhwc.weightRef(1);
  ASSERT_EQ(in.getDim().channel(), 1u);
  ASSERT_EQ(w.getDim().channel(), 1u);
  ASSERT_EQ(w.getDim().height(), 2u);
  ASSERT_EQ(w.getDim().width(), 2u);
  ASSERT_EQ(b.getDim().getFeatureLen(), 1u);

  for (unsigned int h = 0; h < 3; ++h)
    for (unsigned int vw = 0; vw < 3; ++vw)
      in.setValue(0, 0, h, vw, input[h][vw]);
  for (unsigned int kh = 0; kh < 2; ++kh)
    for (unsigned int kw = 0; kw < 2; ++kw)
      w.setValue(0, 0, kh, kw, filter[kh][kw]);
  b.setValue(0, 0, 0, 0, bias);

  ASSERT_NO_THROW(nhwc.forward());

  const auto out = nhwc.readOutput();
  ASSERT_EQ(out.size(), 4u);
  const float expected[4] = {38.f, 48.f, 68.f, 78.f};
  for (unsigned int i = 0; i < 4; ++i)
    EXPECT_EQ(out[i], expected[i])
      << "out[0][0][" << i / 2 << "][" << i % 2 << "]";

  /// and the layout is what the case claims, not a channel first run
  EXPECT_EQ(nhwc.getOutputDim().getFormat(),
            ml::train::TensorDim::Format::NHWC);
}

/**
 * @brief A non square kernel is really read as height, width.
 *
 * Reading the taps in a transposed order would be invisible to a square kernel,
 * so this runs the same input and the same filter values under
 * kernel_size=3,2 and under kernel_size=2,3. The two are mirror images: the
 * first gives a 4:5 output, the second a 5:4 one, so the element counts agree
 * and the values cannot. Each is then pinned to a hand computed element, which
 * is what fixes the direction, and compared against the channel first path.
 *
 * @note Both harnesses are built with no kernel_size of their own beyond the
 * one under test: loadProperties() walks the property vector in order and
 * consumes every entry, so a key given twice is set twice and the last setting
 * is the one that stands. There is deliberately no second kernel_size here.
 *
 *   input                 filter, numbered row by row, bias 0
 *   1  2  3  4            as 3x2:  1 2 / 3 4 / 5 6
 *   5  6  7  8            as 2x3:  1 2 3 / 4 5 6
 *   9 10 11 12
 *   13 14 15 16
 *
 * padding is 1,1 and stride 1, so the tap at (kh, kw) of out[oh][ow] reads
 * input[oh + kh - 1][ow + kw - 1] and is zero outside the 4x4 input.
 *
 *   3x2, out[0][0]: only (1,1) and (2,1) land inside, 4 * 1 + 6 * 5 = 34
 *   2x3, out[0][0]: only (1,1) and (1,2) land inside, 5 * 1 + 6 * 2 = 17
 */
TEST(Conv2DNhwcKernelShape, non_square_kernel_is_height_width) {
  static const float kIn[4][4] = {
    {1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};

  auto build = [](const std::string &kernel) {
    return std::vector<std::string>{"filters=1", "padding=1,1",
                                    "kernel_size=" + kernel};
  };

  struct Side {
    std::string kernel;
    float expected_first;
  };
  const std::vector<Side> sides{{"3,2", 34.f}, {"2,3", 17.f}};

  auto load = [](ConvHarness &h) {
    Tensor &in = h.inputRef();
    ASSERT_EQ(in.getDim().channel(), 1u);
    for (unsigned int r = 0; r < 4; ++r)
      for (unsigned int vw = 0; vw < 4; ++vw)
        in.setValue(0, 0, r, vw, kIn[r][vw]);
    Tensor &w = h.weightRef(0);
    ASSERT_EQ(w.getDim().channel(), 1u);
    const unsigned int kw_cnt = w.getDim().width();
    for (unsigned int kh = 0; kh < w.getDim().height(); ++kh)
      for (unsigned int kw = 0; kw < kw_cnt; ++kw)
        w.setValue(0, 0, kh, kw, (float)(kh * kw_cnt + kw + 1));
    Tensor &b = h.weightRef(1);
    for (unsigned int f = 0; f < b.getDim().channel(); ++f)
      b.setValue(0, f, 0, 0, 0.0f);
  };

  std::vector<std::vector<float>> nhwc_out;
  for (const auto &s : sides) {
    ConvHarness h(build(s.kernel), "nhwc", "1:4:4", 1);
    ASSERT_EQ(h.getOutputDim().getFormat(), ml::train::TensorDim::Format::NHWC)
      << "kernel_size=" << s.kernel;
    load(h);
    EXPECT_NO_THROW(h.forward());
    const auto out = h.readOutput();
    /// the kernel setting is in effect on the geometry, kh and kw swapped give
    /// 4:5 and 5:4, both 20 elements
    ASSERT_EQ(out.size(), 20u) << "kernel_size=" << s.kernel;
    /// and on the value, hand computed above
    EXPECT_EQ(out.front(), s.expected_first) << "kernel_size=" << s.kernel;
    nhwc_out.push_back(out);
  }

  /// the two settings cannot produce the same output, so a kernel_size that
  /// silently did not apply would be caught here as well
  ASSERT_EQ(nhwc_out[0].size(), nhwc_out[1].size());
  EXPECT_GT(maxAbsErr(nhwc_out[0], nhwc_out[1]), kNhwcNchwMaxAbsErr)
    << "kernel_size=3,2 and kernel_size=2,3 produced the same output, so "
       "neither setting was in effect";

  /// and each direction agrees with the channel first path of its own setup
  for (size_t i = 0; i < sides.size(); ++i) {
    ConvHarness nchw(build(sides[i].kernel), "nchw", "1:4:4", 1);
    load(nchw);
    ASSERT_NO_THROW(nchw.forward());
    EXPECT_LE(maxAbsErr(nchw.readOutput(), nhwc_out[i]), kNhwcNchwMaxAbsErr)
      << "kernel_size=" << sides[i].kernel;
  }
}

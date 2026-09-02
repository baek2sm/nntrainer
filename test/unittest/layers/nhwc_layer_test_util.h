// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   nhwc_layer_test_util.h
 * @date   02 September 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Helpers to run a single layer over tensors of a given storage layout.
 *
 * The layer golden-test harness does take a format string, but it reads its
 * inputs from committed binary resources, so a new layout case would need new
 * resources. These helpers build the same contexts by hand instead, so a test
 * can state its input values and its expected output values in one place.
 *
 * Values are always given and read in logical (batch, channel, height, width)
 * order and placed with the by-index accessors, which resolve the layout. The
 * layout under test therefore only ever enters through the layer, and the same
 * test body run with NCHW inputs is an independent implementation of the
 * expected result.
 *
 * @see https://github.com/nntrainer/nntrainer
 */
#ifndef __NNTR_NHWC_LAYER_TEST_UTIL_H__
#define __NNTR_NHWC_LAYER_TEST_UTIL_H__

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <layer.h>
#include <layer_context.h>
#include <tensor.h>
#include <tensor_dim.h>
#include <tensor_wrap_specs.h>
#include <util_func.h>
#include <var_grad.h>

namespace nhwc_test {

using Shape = std::vector<unsigned int>; /**< {batch, channel, height, width} */

/**
 * @brief Finalize a layer and run one forward pass over the given layout.
 *
 * @param layer layer to run, properties already set
 * @param shapes one {batch, channel, height, width} per input
 * @param values flattened element values per input, in logical
 * (batch, channel, height, width) order regardless of the layout
 * @param format storage layout of inputs, weights and activations
 * @param training forwarded to the layer
 * @return the layer's outputs after forwarding
 */
inline std::vector<nntrainer::Tensor>
runForward(nntrainer::Layer *layer, const std::vector<Shape> &shapes,
           const std::vector<std::vector<float>> &values,
           ml::train::TensorDim::Format format, bool training = false) {
  const std::string format_str =
    format == ml::train::TensorDim::Format::NHWC ? "NHWC" : "NCHW";
  const auto ttype = nntrainer::TensorDim::TensorType(
    format, ml::train::TensorDim::DataType::FP32);

  std::vector<nntrainer::TensorDim> in_dims;
  for (const auto &s : shapes) {
    in_dims.emplace_back(s[0], s[1], s[2], s[3], ttype);
  }

  /// req_out_is_connected describes the outputs, not the inputs: every layer
  /// driven here produces a single output
  nntrainer::InitLayerContext init_context(in_dims, std::vector<bool>(1, true),
                                           false, "layout_test", "", 0.0,
                                           {format_str, "FP32", "FP32"});
  layer->finalize(init_context);
  /// mirror what the golden harness does after finalize
  for (auto &dim : init_context.getMutableInputDimensions()) {
    dim.setFormat(format);
  }

  std::vector<nntrainer::Var_Grad> inputs;
  for (size_t i = 0; i < in_dims.size(); ++i) {
    inputs.emplace_back(in_dims[i], nntrainer::Initializer::NONE, true, true,
                        "in");
    auto &t = inputs.back().getVariableRef();
    if (t.size() != values[i].size()) {
      throw std::invalid_argument(
        "layout_test: input values do not match the requested dimension");
    }
    const auto &dim = t.getDim();
    unsigned int idx = 0;
    for (unsigned int b = 0; b < dim.batch(); ++b) {
      for (unsigned int c = 0; c < dim.channel(); ++c) {
        for (unsigned int h = 0; h < dim.height(); ++h) {
          for (unsigned int w = 0; w < dim.width(); ++w) {
            t.setValue(b, c, h, w, values[i][idx++]);
          }
        }
      }
    }
  }

  std::vector<nntrainer::Var_Grad> outputs;
  for (const auto &spec : init_context.getOutSpecs()) {
    outputs.emplace_back(spec.variable_spec.dim, nntrainer::Initializer::NONE,
                         true, true, "out");
    /// an element the layer never writes must show up as NaN rather than as a
    /// plausible leftover
    outputs.back().getVariableRef().setValue(std::nanf(""));
  }

  std::vector<nntrainer::Var_Grad> tensors;
  for (const auto &spec : init_context.getTensorsSpec()) {
    tensors.emplace_back(spec, true);
  }

  std::vector<nntrainer::Var_Grad *> in_ptrs, out_ptrs, tensor_ptrs;
  for (auto &vg : inputs)
    in_ptrs.push_back(&vg);
  for (auto &vg : outputs)
    out_ptrs.push_back(&vg);
  for (auto &vg : tensors)
    tensor_ptrs.push_back(&vg);

  nntrainer::RunLayerContext run_context("layout_test", true, 0.0f, false, 1.0,
                                         nullptr, false, {}, in_ptrs, out_ptrs,
                                         tensor_ptrs);

  layer->forwarding(run_context, training);

  std::vector<nntrainer::Tensor> result;
  for (auto &vg : outputs)
    result.push_back(vg.getVariableRef());
  return result;
}

/**
 * @brief Run one forward pass over NHWC inputs.
 */
inline std::vector<nntrainer::Tensor>
runForwardNhwc(nntrainer::Layer *layer, const std::vector<Shape> &shapes,
               const std::vector<std::vector<float>> &values,
               bool training = false) {
  return runForward(layer, shapes, values, ml::train::TensorDim::Format::NHWC,
                    training);
}

/**
 * @brief Compare a tensor against expected values, element by element, in
 *        logical (batch, channel, height, width) order.
 */
inline void expectClose(const nntrainer::Tensor &actual,
                        const std::vector<float> &expected, float tol) {
  const auto &dim = actual.getDim();
  ASSERT_EQ(actual.size(), expected.size());
  unsigned int idx = 0;
  for (unsigned int b = 0; b < dim.batch(); ++b) {
    for (unsigned int c = 0; c < dim.channel(); ++c) {
      for (unsigned int h = 0; h < dim.height(); ++h) {
        for (unsigned int w = 0; w < dim.width(); ++w) {
          EXPECT_NEAR(actual.getValue(b, c, h, w), expected[idx], tol)
            << "at logical index (b " << b << ", c " << c << ", h " << h
            << ", w " << w << ")";
          ++idx;
        }
      }
    }
  }
}

/**
 * @brief Flatten a tensor in logical (batch, channel, height, width) order.
 */
inline std::vector<float> toLogical(const nntrainer::Tensor &t) {
  const auto &dim = t.getDim();
  std::vector<float> v;
  v.reserve(t.size());
  for (unsigned int b = 0; b < dim.batch(); ++b) {
    for (unsigned int c = 0; c < dim.channel(); ++c) {
      for (unsigned int h = 0; h < dim.height(); ++h) {
        for (unsigned int w = 0; w < dim.width(); ++w) {
          v.push_back(t.getValue(b, c, h, w));
        }
      }
    }
  }
  return v;
}

} /* namespace nhwc_test */

#endif /* __NNTR_NHWC_LAYER_TEST_UTIL_H__ */

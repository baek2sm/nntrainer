// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   addition_layer.cpp
 * @date   30 July 2020
 * @see    https://github.com/nntrainer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Addition Layer Class for Neural Network
 *
 */

#include <addition_layer.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

#include <layer_context.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void AdditionLayer::finalize(InitLayerContext &context) {
  if (!std::get<props::SkipPrefill>(add_props).empty())
    skip_prefill = std::get<props::SkipPrefill>(add_props).get();
  context.setOutputDimensions({context.getInputDimensions()[0]});
}

/**
 * @brief W4A8 static Q8_0: dequantize-and-sum for int8 (Q8_0_TW) residual
 * inputs into a float output.
 *
 * The residual add is a compute joiner: each producer feeding it keeps its
 * output edge int8, and calibration unifies every such producer's output scale
 * to this add's single input activation scale (add:in). So each int8 input's
 * true value is `q_i * s_in`, and the exact sum is `s_in * sum(q_i)` (plus any
 * genuinely float inputs added in their natural units). The result is written
 * to the float output edge; the add does not re-quantize (its output edge stays
 * float), so no rounding beyond the producers' own quantization is introduced.
 */
template <typename OUT>
static void additionDequantSum(RunLayerContext &context, float s_in) {
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  const size_t n = hidden_.size();
  OUT *out = hidden_.getData<OUT>();
  for (size_t i = 0; i < n; ++i)
    out[i] = static_cast<OUT>(0);

  for (unsigned int idx = 0; idx < context.getNumInputs(); ++idx) {
    const Tensor &input_ = context.getInput(idx);
    if (input_.getDataType() == ml::train::TensorDim::DataType::Q8_0_TW) {
      const int8_t *q = input_.getData<int8_t>();
      for (size_t i = 0; i < n; ++i)
        out[i] = static_cast<OUT>(static_cast<float>(out[i]) +
                                  s_in * static_cast<float>(q[i]));
#ifdef ENABLE_FP16
    } else if (input_.getDataType() == ml::train::TensorDim::DataType::FP16) {
      const _FP16 *f = input_.getData<_FP16>();
      for (size_t i = 0; i < n; ++i)
        out[i] = static_cast<OUT>(static_cast<float>(out[i]) +
                                  static_cast<float>(f[i]));
#endif
    } else {
      const float *f = input_.getData<float>();
      for (size_t i = 0; i < n; ++i)
        out[i] = static_cast<OUT>(static_cast<float>(out[i]) + f[i]);
    }
  }
}

void AdditionLayer::forwarding(RunLayerContext &context, bool training) {
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);

  // W4A8 static Q8_0: if any input arrived as an int8 (Q8_0_TW) activation
  // edge, dequantize each by the calibrated input activation scale and sum in
  // float. The output edge is float (this layer does not emit int8), so no
  // requantization is needed.
  bool any_int8_in = false;
  for (unsigned int idx = 0; idx < context.getNumInputs(); ++idx) {
    if (context.getInput(idx).getDataType() ==
        ml::train::TensorDim::DataType::Q8_0_TW) {
      any_int8_in = true;
      break;
    }
  }
  if (any_int8_in) {
    const auto &ais = std::get<props::InputActivationScale>(add_props);
    const float s_in = (!ais.empty() && ais.get() > 0.0f) ? ais.get() : 0.0f;
#ifdef ENABLE_FP16
    if (hidden_.getDataType() == ml::train::TensorDim::DataType::FP16) {
      additionDequantSum<_FP16>(context, s_in);
      return;
    }
#endif
    additionDequantSum<float>(context, s_in);
    return;
  }

  /** @todo check possibility for in-place of addition layer */
  for (unsigned int idx = 0; idx < context.getNumInputs(); ++idx) {
    const Tensor &input_ = context.getInput(idx);
    if (!idx) {
      hidden_.copy(input_);
    } else {
      hidden_.add_i(input_);
    }
  }
}

void AdditionLayer::incremental_forwarding(RunLayerContext &context,
                                           unsigned int from, unsigned int to,
                                           bool training) {
  bool is_prefill = !from;
  if (skip_prefill && is_prefill)
    return;

  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  TensorDim hidden_dim = hidden_.getDim();
  TensorDim hidden_step_dim = hidden_dim;

  hidden_step_dim.batch(1);
  hidden_step_dim.height(to - from);

  for (unsigned int b = 0; b < hidden_.batch(); ++b) {
    Tensor hidden_step = hidden_.getSharedDataTensor(
      hidden_step_dim, b * hidden_dim.getFeatureLen(), true);

    /** @todo check possibility for in-place of addition layer */
    for (unsigned int idx = 0; idx < context.getNumInputs(); ++idx) {
      const Tensor &input_ = context.getInput(idx);
      TensorDim input_dim = input_.getDim();

      TensorDim input_step_dim = input_dim;
      input_step_dim.batch(1);
      input_step_dim.height(to - from);

      Tensor input_step = input_.getSharedDataTensor(
        input_step_dim, b * input_dim.getFeatureLen(), true);
      if (!idx) {
        hidden_step.copy(input_step);
      } else {
        hidden_step.add_i(input_step);
      }
    }
  }
}

void AdditionLayer::calcDerivative(RunLayerContext &context) {

  for (unsigned int idx = 0; idx < context.getNumInputs(); ++idx) {
    /**
     * TODO: replace this with tensor assignment during optimization.
     * Tensor assignment needs to make sure that the previous connected layers
     * are not inplace
     */
    context.getOutgoingDerivative(idx).copy(
      context.getIncomingDerivative(SINGLE_INOUT_IDX));
  }
}

void AdditionLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, add_props);
  if (!remain_props.empty()) {
    std::string msg = "[AdditionLayer] Unknown Layer Properties count " +
                      std::to_string(values.size());
    throw exception::not_supported(msg);
  }
}

void AdditionLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  for (size_t i = 0; i < context.getNumInputs(); ++i) {
    context.updateInput(i, input_dimensions[0]);
  }
  context.updateOutput(SINGLE_INOUT_IDX, input_dimensions[0]);
}

} /* namespace nntrainer */

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
#include <cpu_backend.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <q8_0_tensor.h>
#include <util_func.h>

#include <layer_context.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

namespace {
/// Q8_0 residual helper: accumulate all Q8_0 input tensors into an FP32 sum
/// (row-wise dequant), then requant the sum back into the Q8_0 destination.
/// Used when the activation dtype is Q8_0, since Q8_0_Tensor has no copy/add_i.
void q8_0_residual_add(const std::vector<const Tensor *> &q8_inputs,
                       Tensor &q8_out, unsigned int H, unsigned int W) {
  ComputeOps *o = getComputeOps();
  size_t row_q8_bytes = Q8_0_SIZE * (W / QK8_0);
  size_t total = (size_t)H * W;

  // FP32 accumulation buffer for the residual sum.
  Tensor sum_fp32(TensorDim(1, 1, H, W, q8_out.getFormat(), Tdatatype::FP32),
                  true);
  float *sum_data = sum_fp32.getData<float>();
  std::fill(sum_data, sum_data + total, 0.0f);

  // dequantize_row_q8_0 overwrites (not accumulates), so dequant each input
  // into a temp and add into the sum.
  Tensor tmp_fp32(TensorDim(1, 1, H, W, q8_out.getFormat(), Tdatatype::FP32),
                  true);
  float *tmp_data = tmp_fp32.getData<float>();
  for (const Tensor *in : q8_inputs) {
    const uint8_t *in_q8 = (const uint8_t *)in->getData();
    for (unsigned int r = 0; r < H; ++r) {
      o->dequantize_row_q8_0(in_q8 + r * row_q8_bytes, tmp_data + r * W,
                             (int64_t)W);
    }
    for (size_t i = 0; i < total; ++i)
      sum_data[i] += tmp_data[i];
  }

  uint8_t *out_q8 = (uint8_t *)q8_out.getData();
  for (unsigned int r = 0; r < H; ++r) {
    o->quantize_row_q8_0(sum_data + r * W, out_q8 + r * row_q8_bytes,
                         (int64_t)W);
  }
}
} // namespace

void AdditionLayer::finalize(InitLayerContext &context) {
  if (!std::get<props::SkipPrefill>(add_props).empty())
    skip_prefill = std::get<props::SkipPrefill>(add_props).get();
  context.setOutputDimensions({context.getInputDimensions()[0]});
}

void AdditionLayer::forwarding(RunLayerContext &context, bool training) {
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);

  if (hidden_.getDataType() == Tdatatype::Q8_0) {
    std::vector<const Tensor *> ins;
    ins.reserve(context.getNumInputs());
    for (unsigned int idx = 0; idx < context.getNumInputs(); ++idx)
      ins.push_back(&context.getInput(idx));
    q8_0_residual_add(ins, hidden_, hidden_.getDim().height(),
                      hidden_.getDim().width());
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

    if (hidden_step.getDataType() == Tdatatype::Q8_0) {
      std::vector<const Tensor *> ins;
      ins.reserve(context.getNumInputs());
      for (unsigned int idx = 0; idx < context.getNumInputs(); ++idx) {
        const Tensor &input_ = context.getInput(idx);
        TensorDim input_dim = input_.getDim();
        TensorDim input_step_dim = input_dim;
        input_step_dim.batch(1);
        input_step_dim.height(to - from);
        Tensor *input_step = new Tensor(input_.getSharedDataTensor(
          input_step_dim, b * input_dim.getFeatureLen(), true));
        ins.push_back(input_step);
      }
      q8_0_residual_add(ins, hidden_step, hidden_step.getDim().height(),
                        hidden_step.getDim().width());
      for (auto *t : ins)
        delete t;
      continue;
    }

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

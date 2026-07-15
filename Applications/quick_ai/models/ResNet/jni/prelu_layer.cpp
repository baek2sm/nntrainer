// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   prelu_layer.cpp
 * @date   03 July 2026
 * @brief  custom parametric ReLU (PReLU) layer implementation
 * @see    https://github.com/nntrainer/nntrainer
 */

#include "prelu_layer.h"

namespace custom {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void PReLULayer::finalize(nntrainer::InitLayerContext &context) {
  const nntrainer::TensorDim &in_dim =
    context.getInputDimensions()[SINGLE_INOUT_IDX];

  unsigned int channels = in_dim.channel();

  weight_idx = context.requestWeight(
    nntrainer::TensorDim(1, channels, 1, 1), nntrainer::Initializer::ZEROS,
    nntrainer::WeightRegularizer::NONE, 0.0f, 0.0f, "prelu::alpha", true);

  context.setOutputDimensions(context.getInputDimensions());
}

void PReLULayer::forwarding(nntrainer::RunLayerContext &context,
                            bool training) {
  nntrainer::Tensor &output = context.getOutput(SINGLE_INOUT_IDX);
  const nntrainer::Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  const nntrainer::Tensor &alpha = context.getWeight(weight_idx);

  const float *x = input.getData<float>();
  float *y = output.getData<float>();
  const float *a = alpha.getData<float>();

  const nntrainer::TensorDim &dim = input.getDim();
  const unsigned int batch = dim.batch();
  const unsigned int channel = dim.channel();
  const unsigned int height = dim.height();
  const unsigned int width = dim.width();

  for (unsigned int b = 0; b < batch; ++b) {
    for (unsigned int c = 0; c < channel; ++c) {
      for (unsigned int h = 0; h < height; ++h) {
        for (unsigned int w = 0; w < width; ++w) {
          unsigned int idx = ((b * channel + c) * height + h) * width + w;
          float v = x[idx];
          y[idx] = v >= 0.0f ? v : a[c] * v;
        }
      }
    }
  }
}

void PReLULayer::calcDerivative(nntrainer::RunLayerContext &context) {
  nntrainer::Tensor &dx = context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  const nntrainer::Tensor &output_grad =
    context.getIncomingDerivative(SINGLE_INOUT_IDX);
  const nntrainer::Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  const nntrainer::Tensor &alpha = context.getWeight(weight_idx);

  const float *x = input.getData<float>();
  const float *dy = output_grad.getData<float>();
  const float *a = alpha.getData<float>();
  float *dx_data = dx.getData<float>();

  const nntrainer::TensorDim &dim = input.getDim();
  const unsigned int batch = dim.batch();
  const unsigned int channel = dim.channel();
  const unsigned int height = dim.height();
  const unsigned int width = dim.width();

  for (unsigned int b = 0; b < batch; ++b) {
    for (unsigned int c = 0; c < channel; ++c) {
      for (unsigned int h = 0; h < height; ++h) {
        for (unsigned int w = 0; w < width; ++w) {
          unsigned int idx = ((b * channel + c) * height + h) * width + w;
          float v = x[idx];
          dx_data[idx] = v >= 0.0f ? dy[idx] : a[c] * dy[idx];
        }
      }
    }
  }

  nntrainer::Tensor &dalpha = context.getWeightGrad(weight_idx);
  float *da = dalpha.getData<float>();

  for (unsigned int c = 0; c < channel; ++c) {
    da[c] = 0.0f;
  }

  for (unsigned int b = 0; b < batch; ++b) {
    for (unsigned int c = 0; c < channel; ++c) {
      for (unsigned int h = 0; h < height; ++h) {
        for (unsigned int w = 0; w < width; ++w) {
          unsigned int idx = ((b * channel + c) * height + h) * width + w;
          float v = x[idx];
          if (v < 0.0f) {
            da[c] += v * dy[idx];
          }
        }
      }
    }
  }
}

#ifdef PLUGGABLE

nntrainer::Layer *create_prelu_layer() { return new PReLULayer(); }

void destroy_prelu_layer(nntrainer::Layer *layer) { delete layer; }

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_prelu_layer,
                                                   destroy_prelu_layer};
}

#endif

} // namespace custom

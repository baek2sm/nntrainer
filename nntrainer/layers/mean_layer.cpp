// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   mean_layer.cpp
 * @date   07 April 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is mean layer class (operation layer)
 *
 */

#include "base_properties.h"
#include "common_properties.h"
#include <mean_layer.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <stdexcept>
#include <util_func.h>

#include <layer_context.h>
namespace nntrainer {

void MeanLayer::finalize(InitLayerContext &context) {
  axis = (int)std::get<props::Axis>(mean_props).get();
  if (axis > 3)
    throw std::invalid_argument("Axis value should be less than 4");

  TensorDim in_dim = context.getInputDimensions()[0];
  TensorDim out_dim;
  if (axis == -1) {
    out_dim = TensorDim({in_dim[0], 1, 1, 1});
  } else {
    out_dim = TensorDim(in_dim);
    out_dim[axis] = 1;
  }
  context.setOutputDimensions({out_dim});
}

void MeanLayer::forwarding_operation(const Tensor &input, Tensor &output) {
  if (axis == -1)
    input.average(output);
  else
    input.average(axis, output);
}

void MeanLayer::calcDerivative(RunLayerContext &context) {
  const Tensor &inDeriv = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &outDeriv = context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  const TensorDim &outDim = outDeriv.getDim();

  for (unsigned int i = 0; i < outDim.batch(); ++i) {
    for (unsigned int j = 0; j < outDim.channel(); ++j) {
      for (unsigned int k = 0; k < outDim.height(); ++k) {
        for (unsigned int l = 0; l < outDim.width(); ++l) {
          if (axis == -1) {
            outDeriv.addValue(
              i, j, k, l, inDeriv.getValue(i, j, k, l) / outDim.getFeatureLen(),
              1);
          } else {
            if (axis == 0)
              outDeriv.addValue(i, j, k, l,
                                inDeriv.getValue(1, j, k, l) / outDim[axis], 1);
            else if (axis == 1)
              outDeriv.addValue(i, j, k, l,
                                inDeriv.getValue(i, 1, k, l) / outDim[axis], 1);
            else if (axis == 2)
              outDeriv.addValue(i, j, k, l,
                                inDeriv.getValue(i, j, 1, l) / outDim[axis], 1);
            else if (axis == 3)
              outDeriv.addValue(i, j, k, l,
                                inDeriv.getValue(i, j, k, 1) / outDim[axis], 1);
          }
        }
      }
    }
  }
}

void MeanLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, mean_props);
  if (!remain_props.empty()) {
    std::string msg = "[MeanLayer] Unknown Layer Properties count " +
                      std::to_string(remain_props.size());
    throw exception::not_supported(msg);
  }
}

} /* namespace nntrainer */

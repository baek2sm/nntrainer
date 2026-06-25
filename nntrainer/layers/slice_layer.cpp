// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   slice_layer.cpp
 * @date   02 April 2025
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is slice layer class (operation layer)
 */

#include "common_properties.h"
#include "tensor_base.h"
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <slice_layer.h>
#include <stdexcept>
#include <util_func.h>

#include <layer_context.h>

namespace nntrainer {

void SliceLayer::finalize(InitLayerContext &context) {
  axis = std::get<props::Axis>(slice_props).get();
  start = std::get<props::StartIndex>(slice_props).get() - 1;
  unsigned int end = std::get<props::EndIndex>(slice_props).get() - 1;

  const TensorDim &in_dim = context.getInputDimensions()[0];
  TensorDim outputDim = context.getInputDimensions()[0];

  for (unsigned int i = 0; i < 4; ++i) {
    if (i == axis) {
      outputDim.setTensorDim(i, end - start);
    } else {
      outputDim.setTensorDim(i, in_dim[i]);
    }
  }

  context.setOutputDimensions({outputDim});
}

void SliceLayer::forwarding_operation(const Tensor &input, Tensor &output) {
  const TensorDim &in_dim = input.getDim();
  const TensorDim &out_dim = output.getDim();

  /// Fast bulk-copy path (general, dtype/axis-aware). For a plain contiguous
  /// FP32 NCHW tensor, slicing along any axis yields contiguous runs that can
  /// be memcpy'd instead of the per-element virtual getValue/setValue dispatch.
  /// The slice selects [start, start+out_len) along one axis; the data from
  /// that axis onward is contiguous, so each leading-index tuple maps to one
  /// contiguous run. axis=0/1/3 give one big run per leading tuple; axis=2
  /// (height) gives per-(b,c) runs of width W. The layout math uses only
  /// dim products (no getStrides dependency) so it is NCHW-contiguous-only and
  /// dtype-FP32-only today; non-contiguous / non-FP32 fall back to the scalar
  /// elementwise loop. The Q8_0-aware path is added later (W4A8 chain).
  const bool is_fp32 = in_dim.getDataType() == TensorDim::DataType::FP32 &&
                       out_dim.getDataType() == TensorDim::DataType::FP32;
  const bool contig = input.getContiguous() && output.getContiguous();
  if (is_fp32 && contig && axis <= 3) {
    /// Strides for a contiguous NCHW tensor (element units): derived here so
    /// the path does not depend on TensorDim::getStrides availability.
    const size_t W = in_dim.width();
    const size_t H = in_dim.height();
    const size_t C = in_dim.channel();
    const size_t sW = 1;
    const size_t sH = W * sW;
    const size_t sC = H * sH;
    const size_t sB = C * sC; /// == in_dim.getFeatureLen()
    const size_t strides[4] = {sB, sC, sH, sW};

    const size_t out_len = out_dim[axis];
    const size_t run_elems = out_len * strides[axis];
    const size_t run_bytes = run_elems * sizeof(float);
    /// Number of contiguous runs = product of dims strictly before the axis.
    size_t n_runs = 1;
    for (unsigned int i = 0; i < axis; ++i) {
      n_runs *= in_dim[i];
    }
    /// Selection start offset along the sliced axis (in elements).
    const size_t in_start_off = static_cast<size_t>(start) * strides[axis];
    /// Per-run source gap. Consecutive runs iterate the leading dims (strictly
    /// before `axis`) in flat order, so consecutive runs are spaced by the
    /// stride of the LAST leading dim, i.e. strides[axis-1] (== run_elems for
    /// axis==0 where there are no leading dims and n_runs==1). Using strides
    /// [axis] here corrupts every run after the first for multi-run slices.
    const size_t run_gap = (axis == 0) ? run_elems : strides[axis - 1];

    const float *in_data = input.getData<float>();
    float *out_data = output.getData<float>();
    for (size_t r = 0; r < n_runs; ++r) {
      const size_t src_off = r * run_gap + in_start_off;
      const size_t dst_off = r * run_elems;
      std::memcpy(out_data + dst_off, in_data + src_off, run_bytes);
    }
    return;
  }

  /// Scalar elementwise fallback (non-contiguous, non-FP32, or unset axis).
  /// NOTE: axis==0 (batch slice) is handled here by b_idx offset; the fast
  /// path above already covers batch slicing contiguously when applicable.
  TensorDim outputDim = output.getDim();
  for (unsigned int b = 0; b < outputDim.batch(); ++b) {
    for (unsigned int c = 0; c < outputDim.channel(); ++c) {
      for (unsigned int h = 0; h < outputDim.height(); ++h) {
        for (unsigned int w = 0; w < outputDim.width(); ++w) {
          unsigned int b_idx = (axis == 0) ? b + start : b;
          unsigned int c_idx = (axis == 1) ? c + start : c;
          unsigned int h_idx = (axis == 2) ? h + start : h;
          unsigned int w_idx = (axis == 3) ? w + start : w;
          output.setValue(b, c, h, w,
                          input.getValue(b_idx, c_idx, h_idx, w_idx));
        }
      }
    }
  }
}

void SliceLayer::calcDerivative(RunLayerContext &context) {
  const Tensor &inDeriv = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &outDeriv = context.getOutgoingDerivative(SINGLE_INOUT_IDX);

  for (unsigned int b = 0; b < inDeriv.batch(); ++b) {
    for (unsigned int c = 0; c < inDeriv.channel(); ++c) {
      for (unsigned int h = 0; h < inDeriv.height(); ++h) {
        for (unsigned int w = 0; w < inDeriv.width(); ++w) {
          unsigned int b_idx = (axis == 0) ? b + start : b;
          unsigned int c_idx = (axis == 1) ? c + start : c;
          unsigned int h_idx = (axis == 2) ? h + start : h;
          unsigned int w_idx = (axis == 3) ? w + start : w;
          outDeriv.setValue(b_idx, c_idx, h_idx, w_idx,
                            inDeriv.getValue(b, c, h, w));
        }
      }
    }
  }
}

void SliceLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, slice_props);
  if (!remain_props.empty()) {
    std::string msg = "[SliceLayer] Unknown Layer Properties count " +
                      std::to_string(values.size());
    throw exception::not_supported(msg);
  }
}

} /* namespace nntrainer */

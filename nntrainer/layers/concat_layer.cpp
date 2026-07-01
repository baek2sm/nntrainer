// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   concat_layer.cpp
 * @date   27 Oct 2020
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Concat Layer Class for Neural Network
 *
 * @todo merge concat and split layer to a common implementation
 */

#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include <concat_layer.h>
#include <layer_context.h>
#include <nntr_threads.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <tensor_dim.h>
#include <util_func.h>

namespace nntrainer {

#ifdef ENABLE_FP16
/**
 * @brief Quantize a contiguous FP16 NHWC tensor to tensor-wise Q8_0.
 *
 * Layout: uint16_t scale followed by int8_t qs[round_up(nelem,32)], with qs
 * innermost (NHWC physical order). Scale = amax/127 as FP16.
 */
static void quantize_nhwc_q8_0_tensor_fp16(const _FP16 *src, int rows,
                                           int cols, uint8_t *storage) {
  const size_t nelem = (size_t)rows * cols;
  float amax = 0.0f;
  for (size_t i = 0; i < nelem; ++i) {
    amax = std::max(amax, std::fabs(static_cast<float>(src[i])));
  }
  float scale = amax / 127.0f;
  if (scale == 0.0f)
    scale = 1.0f;
  _FP16 d = static_cast<_FP16>(scale);
  std::memcpy(storage, &d, sizeof(uint16_t));
  int8_t *qs = reinterpret_cast<int8_t *>(storage + sizeof(uint16_t));
  for (size_t i = 0; i < nelem; ++i) {
    float v = static_cast<float>(src[i]) / scale;
    v = std::max(-127.0f, std::min(127.0f, v));
    qs[i] = static_cast<int8_t>(std::round(v));
  }
  // zero padding to keep the signed range symmetric and avoid accidental -128
  const size_t padded = ((nelem + 31) / 32) * 32;
  for (size_t i = nelem; i < padded; ++i)
    qs[i] = 0;
}
#endif

ConcatLayer::ConcatLayer() : Layer(), leading_helper_dim(1) {}

static constexpr size_t SINGLE_INOUT_IDX = 0;

void ConcatLayer::finalize(InitLayerContext &context) {
  auto &concat_dimension_prop = std::get<props::ConcatDimension>(concat_props);
  /** for backward compatibility, default concat dimension will be channel */
  /// @todo this is hacky way to force concat dimension to width if channel
  /// dimension is taken, this is because recurrent realizer, return sequence
  /// exploits concat layer but have no control over where to stack/axis
  unsigned int concat_dimension =
    context.getInputDimensions().front().channel() > 1 ? 3 : 1;
  if (!concat_dimension_prop.empty())
    concat_dimension = concat_dimension_prop.get();
  concat_dimension_cache = concat_dimension;

  /**
   * The concat is only done along the axis dimension.
   * For example, consider 2 inputs a, b with dimensions [b,c,h,w] each
   * 1. concat_dimension = 1, output_dim = [b,c_a+c_b,h,w]
   * 2. concat_dimension = 2, output_dim = [b,c,h_a+h_b,w]
   * 3. concat_dimension = 3, output_dim = [b,c,h,w_a+w_b]
   */
  auto const &input_dims = context.getInputDimensions();
  const TensorDim &input_dim_0 = input_dims[SINGLE_INOUT_IDX];
  unsigned int concat_dim_val = input_dim_0.getTensorDim(concat_dimension);

  for (unsigned int idx = 1; idx < input_dims.size(); ++idx) {
    const TensorDim &dim = input_dims[idx];

    for (unsigned int i = 0; i < ml::train::TensorDim::getNumDim(); ++i) {
      if (i == concat_dimension)
        continue;
      NNTR_THROW_IF(input_dim_0[i] != dim[i], std::runtime_error)
        << "Error: concat layer requires same shape from all input layers "
           "along non-concat dimension";
    }
    concat_dim_val += dim[concat_dimension];
  }

  TensorDim output_dim = input_dim_0;
  output_dim.setTensorDim(concat_dimension, concat_dim_val);

  context.setOutputDimensions({output_dim});

  /**
   * The following helper shapes facilitate efficient concatenation and split of
   * the data.
   *
   * The helper shapes are created by consolidating all the dimensions before
   * the concat dimension to the first axis and all the remaining dimensions to
   * the last axis.
   *
   * @note This is possible since the data starting from the concat dimension to
   * the end is always continuous.
   *
   * @example the following shows how the helper dimension will look with given
   * inputs and concat dimension.
   *
   *          | cat_dim 1 | cat_dim 2 | cat_dim 3
   *  --------|-----------|-----------|-----------
   *  input0  |  2:1:2:3  |  1:2:1:3  |  1:2:2:3
   *  input1  |  2:3:2:3  |  1:2:3:3  |  1:2:2:1
   *  --------|-----------|-----------|-----------
   *  helper0 |  2:1:1:6  |  2:1:1:3  |  4:1:1:3
   *  helper1 |  2:1:1:18 |  2:1:1:9  |  4:1:1:1
   *
   */
  /// Setup output_reshape_helper (how output should be reshaped)
  output_reshape_helper.channel(1);
  output_reshape_helper.height(1);
  output_reshape_helper.width(1);
  for (unsigned int axis = concat_dimension;
       axis < ml::train::TensorDim::getNumDim(); ++axis) {
    output_reshape_helper.width(output_reshape_helper.width() *
                                output_dim.getTensorDim(axis));
  }

  /// Setup input_reshape_helper (how inputs should be reshaped)
  input_reshape_helper.resize(input_dims.size());

  for (unsigned int idx = 0; idx < input_reshape_helper.size(); idx++) {
    input_reshape_helper[idx].channel(1);
    input_reshape_helper[idx].height(1);
    input_reshape_helper[idx].width(1);

    for (unsigned int axis = concat_dimension;
         axis < ml::train::TensorDim::getNumDim(); ++axis) {

      input_reshape_helper[idx].width(input_reshape_helper[idx].width() *
                                      input_dims[idx].getTensorDim(axis));
    }
  }

  leading_helper_dim = 1;
  for (unsigned int idx = 1; idx < concat_dimension; ++idx) {
    leading_helper_dim *= output_dim.getTensorDim(idx);
  }

  setBatch(input_dims[SINGLE_INOUT_IDX].batch());

  // If any input is Q8_0 in NHWC mode, we need a contiguous FP16 staging buffer
  // to dequantize each input, concat channel-wise, and requantize to tensor-wise
  // Q8_0. Only channel-axis concat is supported for Q8_0 (matching graph usage).
  if (output_dim.getFormat() == ml::train::TensorDim::Format::NHWC &&
      concat_dimension_cache == 1) {
    for (unsigned int idx = 0; idx < input_dims.size(); ++idx) {
      if (input_dims[idx].getDataType() ==
          ml::train::TensorDim::DataType::Q8_0) {
        TensorDim scratch_dim = output_dim;
        scratch_dim.setDataType(ml::train::TensorDim::DataType::FP16);
        fp16_scratch_idx = context.requestTensor(
          scratch_dim, "fp16_concat_scratch", Initializer::NONE, false,
          TensorLifespan::MAX_LIFESPAN);
        break;
      }
    }
  }
}

void ConcatLayer::forwarding(RunLayerContext &context, bool training) {
  /**
   * Forwarding in ConcatLayer works as follows
   *
   *    in1        in2       in3                  output
   * |---0---| |----3----| |--6--|      |---0---||----3----||--6--|
   * |---1---| |----4----| |--7--|  =>  |---1---||----4----||--7--|
   * |---2---| |----5----| |--8--|      |---2---||----5----||--8--|
   *
   * @note For each input tensor, it iterates batches and copies the entire
   * width size to the corresponding output position. In the diagram above, the
   * row would be a batch, and the column would be a width. the number of each
   * block in the diagram indicates the order of copy to output.
   *
   * @todo avoid copy by creating input here as a shared_tensor of the output
   * here and then this layer can be in_place as well
   */
  Tensor &output = context.getOutput(SINGLE_INOUT_IDX);

  const TensorDim out_dim = output.getDim();

  // NHWC path: channel is physical-innermost, so a channel-axis concat (the
  // only concat YOLOv11 uses) interleaves per-pixel channel runs. The NCHW
  // reshape-helper path below assumes channel-major planes and would copy
  // into the wrong physical offsets, so handle NHWC separately. Only the
  // channel axis (logical 1) is supported here, matching the graph usage.
  if (out_dim.getFormat() == ml::train::TensorDim::Format::NHWC &&
      concat_dimension_cache == 1) {
    const bool any_q8 = [&]() {
      for (unsigned int idx = 0; idx < context.getNumInputs(); ++idx)
        if (context.getInput(idx).getDataType() ==
            TensorDim::DataType::Q8_0)
          return true;
      return false;
    }();

    if (any_q8) {
#ifdef ENABLE_FP16
      // Q8_0 channel concat: dequantize each input to the planned FP16
      // scratch, write its channel slice, then quantize the whole output.
      Tensor &scratch = context.getTensor(fp16_scratch_idx);
      _FP16 *dst = scratch.getData<_FP16>();
      const unsigned int B = out_dim.batch();
      const unsigned int H = out_dim.height();
      const unsigned int W = out_dim.width();
      const size_t HW = (size_t)H * W;
      const unsigned int Co = out_dim.channel();
      unsigned int c_offset = 0;
      for (unsigned int idx = 0; idx < context.getNumInputs(); idx++) {
        Tensor &input = context.getInput(idx);
        const unsigned int Ci = input.channel();
        if (input.getDataType() == TensorDim::DataType::FP16) {
          const _FP16 *src = input.getData<_FP16>();
          for (unsigned int b = 0; b < B; ++b) {
            const size_t base = (size_t)b * HW;
            for (size_t p = 0; p < HW; ++p) {
              std::memcpy(dst + (base + p) * Co + c_offset,
                          src + (base + p) * Ci, (size_t)Ci * sizeof(_FP16));
            }
          }
        } else if (input.getDataType() == TensorDim::DataType::Q8_0) {
          const uint8_t *storage = reinterpret_cast<const uint8_t *>(
                                     input.getData()) -
                                   sizeof(uint16_t);
          uint16_t du;
          std::memcpy(&du, storage, sizeof(uint16_t));
          float scale = static_cast<float>(
            *reinterpret_cast<const _FP16 *>(&du));
          const int8_t *src = input.getData<int8_t>();
          for (unsigned int b = 0; b < B; ++b) {
            const size_t base = (size_t)b * HW;
            for (size_t p = 0; p < HW; ++p) {
              _FP16 *d = dst + (base + p) * Co + c_offset;
              const int8_t *s = src + (base + p) * Ci;
              for (unsigned int c = 0; c < Ci; ++c)
                d[c] = static_cast<_FP16>(static_cast<float>(s[c]) * scale);
            }
          }
        } else if (input.getDataType() == TensorDim::DataType::FP32) {
          // Detect-head class logit is intentionally kept FP32 for accuracy;
          // stage it into the FP16 concat buffer so the final output can still
          // be tensor-wise Q8_0.
          const float *src = input.getData<float>();
          for (unsigned int b = 0; b < B; ++b) {
            const size_t base = (size_t)b * HW;
            for (size_t p = 0; p < HW; ++p) {
              _FP16 *d = dst + (base + p) * Co + c_offset;
              const float *s = src + (base + p) * Ci;
              for (unsigned int c = 0; c < Ci; ++c)
                d[c] = static_cast<_FP16>(s[c]);
            }
          }
        } else {
          throw std::invalid_argument(
            "ConcatLayer: unsupported input dtype for Q8_0 concat");
        }
        c_offset += Ci;
      }
      quantize_nhwc_q8_0_tensor_fp16(
        dst, (int)(B * HW), (int)Co,
        reinterpret_cast<uint8_t *>(output.getData()) - sizeof(uint16_t));
      return;
#else
      throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
    }

    const unsigned int B = out_dim.batch();
    const unsigned int H = out_dim.height();
    const unsigned int W = out_dim.width();
    const size_t HW = (size_t)H * W;
    unsigned int c_offset = 0;
    for (unsigned int idx = 0; idx < context.getNumInputs(); idx++) {
      Tensor &input = context.getInput(idx);
      const unsigned int Ci = input.channel();
      if (input.getDataType() == TensorDim::DataType::FP32) {
        const float *src = input.getData<float>();
        float *dst = output.getData<float>();
        const unsigned int Co = out_dim.channel();
        for (unsigned int b = 0; b < B; ++b) {
          const size_t base = (size_t)b * HW;
          for (size_t p = 0; p < HW; ++p) {
            const float *s = src + (base + p) * Ci;
            float *d = dst + (base + p) * Co + c_offset;
            std::memcpy(d, s, (size_t)Ci * sizeof(float));
          }
        }
      } else if (input.getDataType() == TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
        const _FP16 *src = input.getData<_FP16>();
        _FP16 *dst = output.getData<_FP16>();
        const unsigned int Co = out_dim.channel();
        for (unsigned int b = 0; b < B; ++b) {
          const size_t base = (size_t)b * HW;
          for (size_t p = 0; p < HW; ++p) {
            const _FP16 *s = src + (base + p) * Ci;
            _FP16 *d = dst + (base + p) * Co + c_offset;
            std::memcpy(d, s, (size_t)Ci * sizeof(_FP16));
          }
        }
#else
        throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
      }
      c_offset += Ci;
    }
    return;
  }

  output.reshape(output_reshape_helper);
  unsigned int output_width_offset = 0;
  TensorDim::TensorType tensor_type = output.getTensorType();

  for (unsigned int idx = 0; idx < context.getNumInputs(); idx++) {
    Tensor &input = context.getInput(idx);
    const TensorDim in_dim = input.getDim();
    auto const &irh = input_reshape_helper[idx];
    input.reshape(irh);
    unsigned int data_copy_size = irh.width();

    /** loop over the dimensions before the concat dimension */
    if (in_dim.getDataType() == TensorDim::DataType::FP32) {
      /** copy continous tensor data (reshaped width) */
      for (unsigned int batch = 0; batch < output.batch(); batch++) {
        Tensor dest_tensor = Tensor::Map<float>(
          output.getAddress<float>(batch, 0, 0, output_width_offset),
          data_copy_size * sizeof(float),
          {1, 1, 1, data_copy_size, tensor_type});
        const Tensor source_tensor =
          Tensor::Map<float>(input.getAddress<float>(batch, 0, 0, 0),
                             data_copy_size * sizeof(float),
                             {1, 1, 1, data_copy_size, tensor_type});
        dest_tensor.copy(source_tensor);
      }
    } else if (in_dim.getDataType() == TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
      /** copy continous tensor data (reshaped width) */
      for (unsigned int batch = 0; batch < output.batch(); batch++) {
        Tensor dest_tensor = Tensor::Map<_FP16>(
          output.getAddress<_FP16>(batch, 0, 0, output_width_offset),
          data_copy_size * sizeof(_FP16),
          {1, 1, 1, data_copy_size, tensor_type});
        const Tensor source_tensor =
          Tensor::Map<_FP16>(input.getAddress<_FP16>(batch, 0, 0, 0),
                             data_copy_size * sizeof(_FP16),
                             {1, 1, 1, data_copy_size, tensor_type});
        dest_tensor.copy(source_tensor);
      }
#else
      throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
    }

    output_width_offset += irh.width();
    input.reshape(in_dim);
  }

  output.reshape(out_dim);
}

void ConcatLayer::incremental_forwarding(RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) {
  /**
   * @todo avoid copy by creating input here as a shared_tensor of the output
   * here and then this layer can be in_place as well
   */
  Tensor &output = context.getOutput(SINGLE_INOUT_IDX);

  const TensorDim out_dim = output.getDim();
  output.reshape(output_reshape_helper);
  unsigned int output_height_offset = 0;
  unsigned int data_copy_size = output_reshape_helper.width();

  // @todo: this implementation is only works when axis is 3(width). Consider
  // for other axes
  unsigned int batch_channel = out_dim.batch() * out_dim.channel();

  for (unsigned int idx = 0; idx < context.getNumInputs(); idx++) {
    Tensor &input = context.getInput(idx);
    const TensorDim in_dim = input.getDim();
    auto const &irh = input_reshape_helper[idx];
    input.reshape(irh);

    /** loop over the dimensions before the concat dimension */
    for (unsigned int batch = batch_channel * from; batch < batch_channel * to;
         batch++) {
      /** loop over the concat dimension itself */
      for (unsigned int count = 0; count < irh.height(); count++) {
        Tensor dest_tensor = Tensor::Map(
          output.getAddress(batch, 0, output_height_offset + count, 0),
          data_copy_size * sizeof(float), {1, 1, 1, data_copy_size});
        const Tensor source_tensor = Tensor::Map(
          input.getAddress(batch, 0, count, 0), data_copy_size * sizeof(float),
          {1, 1, 1, data_copy_size});
        dest_tensor.copy(source_tensor);
      }
    }

    input.reshape(in_dim);
    output_height_offset += irh.height();
  }

  output.reshape(out_dim);
}

void ConcatLayer::calcDerivative(RunLayerContext &context) {
  /**
   * calcDerivative in ConcatLayer works as follows
   *
   *           output                    in1        in2       in3
   * |---0---||----3----||--6--|      |---0---| |----3----| |--6--|
   * |---1---||----4----||--7--|  =>  |---1---| |----4----| |--7--|
   * |---2---||----5----||--8--|      |---2---| |----5----| |--8--|
   *
   * @note For each input tensor, it iterates batches and copies the entire
   * input width size from the output tensor to the corresponding input. In the
   * diagram above, the row would be a batch, and the column would be a width.
   * The number of each block in the diagram indicates the order of copy to
   * inputs.
   *
   * @todo avoid copy by creating input here as a shared_tensor of the output
   * here and then this layer can be in_place as well
   */
  Tensor output = context.getIncomingDerivative(SINGLE_INOUT_IDX);

  output.reshape(output_reshape_helper);
  unsigned int output_width_offset = 0;
  TensorDim::TensorType tensor_type = output.getTensorType();

  for (unsigned int idx = 0; idx < context.getNumInputs(); idx++) {
    Tensor &input = context.getOutgoingDerivative(idx);
    const TensorDim in_dim = input.getDim();
    auto const &irh = input_reshape_helper[idx];
    input.reshape(irh);
    unsigned int data_copy_size = irh.width();

    if (in_dim.getDataType() == TensorDim::DataType::FP32) {
      /** loop over the dimensions before the concat dimension */
      for (unsigned int batch = 0; batch < output.batch(); batch++) {
        /** copy continous data (reshaped width size) in a tensor */
        const Tensor source_tensor = Tensor::Map<float>(
          output.getAddress<float>(batch, 0, 0, output_width_offset),
          data_copy_size * sizeof(float),
          {1, 1, 1, data_copy_size, tensor_type});
        Tensor dest_tensor =
          Tensor::Map<float>(input.getAddress<float>(batch, 0, 0, 0),
                             data_copy_size * sizeof(float),
                             {1, 1, 1, data_copy_size, tensor_type});
        dest_tensor.copy(source_tensor);
      }
    } else if (in_dim.getDataType() == TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
      /** loop over the dimensions before the concat dimension */
      for (unsigned int batch = 0; batch < output.batch(); batch++) {
        /** copy continous data (reshaped width size) in a tensor */
        const Tensor source_tensor = Tensor::Map<_FP16>(
          output.getAddress<_FP16>(batch, 0, 0, output_width_offset),
          data_copy_size * sizeof(_FP16),
          {1, 1, 1, data_copy_size, tensor_type});
        Tensor dest_tensor =
          Tensor::Map<_FP16>(input.getAddress<_FP16>(batch, 0, 0, 0),
                             data_copy_size * sizeof(_FP16),
                             {1, 1, 1, data_copy_size, tensor_type});
        dest_tensor.copy(source_tensor);
      }
#else
      throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
    }

    input.reshape(in_dim);
    output_width_offset += irh.width();
  }
}

void ConcatLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, concat_props);
  NNTR_THROW_IF(!remain_props.empty(), std::invalid_argument)
    << "[ConcatLayer] Unknown Layer Properties count " +
         std::to_string(values.size());
}

void ConcatLayer::exportTo(Exporter &exporter,
                           const ml::train::ExportMethods &method) const {
  Layer::exportTo(exporter, method);
  exporter.saveResult(concat_props, method, this);
}

} /* namespace nntrainer */

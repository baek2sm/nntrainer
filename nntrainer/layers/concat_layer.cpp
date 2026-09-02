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

#include <cstring>
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
ConcatLayer::ConcatLayer() : Layer(), leading_helper_dim(1) {}

static constexpr size_t SINGLE_INOUT_IDX = 0;

/// logical (batch, channel, height, width) axis indices, as the axis property
/// and TensorDim::getTensorDim() both index them this way
static constexpr unsigned int CONCAT_AXIS_CHANNEL = 1;
static constexpr unsigned int CONCAT_AXIS_HEIGHT = 2;
static constexpr unsigned int CONCAT_AXIS_WIDTH = 3;

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
  concat_axis = concat_dimension;

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
}

/**
 * @brief Concatenate the inputs into an NHWC output.
 *
 * The reshape helpers the generic path uses collapse everything from the
 * concatenated axis on into one width and copy whole leading slices, which is
 * only equivalent to the element mapping when that axis is innermost in
 * storage. In NHWC the channel axis is innermost, so the inputs interleave as
 * per-pixel channel runs and the helpers write them to the wrong offsets.
 *
 * Channels being innermost instead makes two layouts of copy contiguous, per
 * input, with a channel offset (chan_off) and a pixel offset into the output:
 * - concatenating the channel axis: every output channel slot of a pixel takes
 *   that input's C-long channel run, so the unit is one (batch, pixel) pair.
 * - concatenating a spatial axis: every input has the same channel count, so a
 *   whole input row is contiguous and lands at an offset inside an output row.
 *
 * @tparam T element type of the tensors
 * @param output output tensor, contiguous NHWC
 * @param context layer context holding the inputs
 * @param axis logical (batch, channel, height, width) axis being concatenated
 */
template <typename T>
static void concatNhwc(Tensor &output, RunLayerContext &context,
                       unsigned int axis) {
  const TensorDim out_dim = output.getDim();
  const unsigned int out_height = out_dim.height();
  const unsigned int out_width = out_dim.width();
  const unsigned int out_channels = out_dim.channel();
  T *dst = output.getData<T>();

  /// the concatenated extent already contributed to the output, so the offset
  /// of each input starts at zero and grows by the extent of the ones before it
  unsigned int chan_off = 0;
  unsigned int height_off = 0;
  unsigned int width_off = 0;

  for (unsigned int idx = 0; idx < context.getNumInputs(); idx++) {
    Tensor &input = context.getInput(idx);
    NNTR_THROW_IF(!input.getContiguous(), nntrainer::exception::not_supported)
      << "[Concat] NHWC concat requires a contiguous input";

    const TensorDim in_dim = input.getDim();
    const unsigned int in_channels = in_dim.channel();
    const unsigned int in_height = in_dim.height();
    const unsigned int in_width = in_dim.width();
    /// a row of an input, in elements, is contiguous in both tensors
    const size_t row_size = (size_t)in_width * in_channels;
    const T *src = input.getData<T>();

    /// one work unit is one (batch, input row) pair
    const unsigned int total = in_dim.batch() * in_height;
    auto job = [&](unsigned int s, unsigned int e, unsigned int, void *) {
      for (unsigned int unit = s; unit < e; ++unit) {
        const unsigned int b = unit / in_height;
        const unsigned int ih = unit % in_height;
        const T *src_row = src + ((size_t)b * in_height + ih) * row_size;
        /// an output pixel is out_channels wide even for an input that
        /// contributes fewer, so the output offsets step by out_channels
        T *dst_row =
          dst +
          ((size_t)(b * out_height + height_off + ih) * out_width + width_off) *
            out_channels +
          chan_off;
        if (axis == CONCAT_AXIS_HEIGHT) {
          /// out_width is the input width here, so the row is one contiguous
          /// run of the output
          std::memcpy(dst_row, src_row, row_size * sizeof(T));
        } else {
          /**
           * Channel and width axis: the output row is shared with the other
           * inputs, so the row is cut into pixels to step over the channels
           * they own. For the channel axis the stride step is the full output
           * channel count while only this input's channels are copied.
           */
          for (unsigned int iw = 0; iw < in_width; ++iw)
            std::memcpy(dst_row + (size_t)iw * out_channels,
                        src_row + (size_t)iw * in_channels,
                        in_channels * sizeof(T));
        }
      }
    };

    bool ran_parallel = false;
    if (total > 1) {
      auto workers = ParallelBatch(job, total, nullptr);
      if (workers.getNumWorkers() > 1) {
        workers.run();
        ran_parallel = true;
      }
    }
    if (!ran_parallel)
      job(0, total, 0, nullptr);

    switch (axis) {
    case CONCAT_AXIS_CHANNEL:
      chan_off += in_channels;
      break;
    case CONCAT_AXIS_HEIGHT:
      height_off += in_height;
      break;
    case CONCAT_AXIS_WIDTH:
      width_off += in_width;
      break;
    default:
      break;
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

  /**
   * The reshape helpers below copy whole leading slices, which is only correct
   * when the concatenated axis is the innermost one. In NHWC the channel axis
   * is innermost, so the inputs interleave as per-pixel channel runs and the
   * helpers would write them into the wrong offsets.
   */
  if (out_dim.getFormat() == TensorDim::Format::NHWC) {
    NNTR_THROW_IF(concat_axis != CONCAT_AXIS_CHANNEL &&
                    concat_axis != CONCAT_AXIS_HEIGHT &&
                    concat_axis != CONCAT_AXIS_WIDTH,
                  nntrainer::exception::not_supported)
      << "[Concat] NHWC concat supports the channel, height and width axes, "
         "got axis="
      << concat_axis;
    NNTR_THROW_IF(!output.getContiguous(), nntrainer::exception::not_supported)
      << "[Concat] NHWC concat requires a contiguous output";

    switch (output.getDataType()) {
    case TensorDim::DataType::FP32:
      concatNhwc<float>(output, context, concat_axis);
      return;
#ifdef ENABLE_FP16
    case TensorDim::DataType::FP16:
      concatNhwc<_FP16>(output, context, concat_axis);
      return;
#endif
    default:
      throw std::runtime_error("Unsupported datatype");
    }
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

// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   conv2d_layer.h
 * @date   02 June 2020
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Convolution Layer Class for Neural Network
 *
 */
#include <algorithm>
#include <cstring>
#include <limits>
#include <string>

#include <conv2d_layer.h>
#include <cpu_backend.h>
#include <layer_context.h>
#include <lazy_tensor.h>
#include <nntr_threads.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <profiler.h>
#include <tensor_dim.h>
#include <thread>
#include <util_func.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

namespace detail {

/**
 * @brief Dimension a channel last quantized conv2d weight is requested with.
 *
 * A PER_CHANNEL_AFFINE QINT8 tensor sizes its scale vector to width(), so the
 * only spelling that yields one scale per output channel is one whose width is
 * the output channel count -- which the classical
 * (filter_size, in_ch, kh, kw) request is not, because there width() is the
 * kernel width. This spelling carries every tap as well, since its four request
 * axes are the four axes of the kernel: batch() kh, channel() in_ch, height()
 * kw and width() filter_size, which is how @a dequantizeKernel reads it back.
 *
 * @note The words "channel last" name the axis convention the per-channel scale
 * vector follows -- the output channel is the axis the scales key on -- and say
 * nothing about which axis varies fastest in the tensor's storage; a reader
 * should not infer a byte order from the spelling of the request.
 */
TensorDim channelLastKernelDim(unsigned int k_height, unsigned int in_ch,
                               unsigned int k_width, unsigned int filter_size,
                               const TensorDim::TensorType &t_type) {
  return TensorDim(k_height, in_ch, k_width, filter_size, t_type);
}

/**
 * @brief Dequantize a channel last QINT8 weight into the operand of the dot.
 *
 * The dot in Conv2DLayer::forwarding() consumes the classical
 * (filter_size, in_ch, kh, kw) operand -- the operand im2col builds its columns
 * against, and the dimension a float weight of this layer has. So the quantized
 * weight is turned into exactly that operand, and the result is the operand the
 * layer would have dotted had the weight been stored in floating point: the
 * weight dtype is a storage decision, not a change of computation.
 *
 * @param[in] weight weight as requested, see @a channelLastKernelDim
 * @param[in] dtype activation data type to dequantize into, i.e. the type of
 * the tensors the dot is given
 * @return Tensor classical operand holding every tap scaled by the scale of its
 *         output channel
 *
 * @note Both sides are addressed through the logical accessors, so each
 * tensor's own format resolves its physical order and the operand comes out in
 * the format of the weight it was built from. The one assumption is the axis
 * convention the request establishes: width() is the output channel, which is
 * also the index the per-channel scale vector is keyed on.
 *
 * @note The weight codes are read as int8_t. Read as the default float of
 * Tensor::getValue, getData<float>() would reinterpret four int8 taps as one
 * float and every tap of the operand would be garbage.
 */
Tensor dequantizeKernel(const Tensor &weight, TensorDim::DataType dtype) {
  const float *scales = weight.getScale<float>();

  TensorDim kernel_dim(weight.width(), weight.channel(), weight.batch(),
                       weight.height(),
                       TensorDim::TensorType(weight.getFormat(), dtype));
  Tensor kernel(kernel_dim);

  for (unsigned int f = 0; f < kernel_dim.batch(); ++f) {
    /// the per-channel scale vector is one scale per output channel
    const float scale = scales[f];
    for (unsigned int c = 0; c < kernel_dim.channel(); ++c) {
      for (unsigned int h = 0; h < kernel_dim.height(); ++h) {
        for (unsigned int w = 0; w < kernel_dim.width(); ++w) {
          /// the weight is (kh, in_ch, kw, filter_size): its batch is the
          /// kernel row, its height the kernel column
          kernel.setValue(f, c, h, w,
                          weight.getValue<int8_t>(h, c, w, f) * scale);
        }
      }
    }
  }
  return kernel;
}

} // namespace detail

namespace {

static TensorDim calcCol2ImOutputDim(const TensorDim &out,
                                     const TensorDim &kdim) {

  return TensorDim({kdim.getFeatureLen(), out.width() * out.height()},
                   out.getTensorType());
}

/**
 * @brief     reconstruct image data from 2d column matrix
 *
 * @param[in] in input data
 * @param[in] kdim kernel dimesion for define number of row
 * @param[in] padding padding information
 * @param[in] mstride stride value : x, y direction
 * @param[in] dilation kernel dilation factor : x, y each
 * @param[out] image image tensor to put
 */
static void col2im(const Tensor &col_matrix, const TensorDim &kdim,
                   const std::array<unsigned, 4> &padding,
                   const std::array<props::Stride, CONV2D_DIM> &mstride,
                   const std::array<props::Dilation, CONV2D_DIM> &dilation,
                   Tensor &image) {

  auto pt = padding[0];
  auto pb = padding[1];
  auto pl = padding[2];
  auto pr = padding[3];

  unsigned k_height = kdim.height();
  unsigned k_width = kdim.width();

  /// effective kernel height considering dilation
  unsigned eff_k_height = (k_height - 1) * dilation[0] + 1;
  /// effective kernel width considering dilation
  unsigned eff_k_width = (k_width - 1) * dilation[1] + 1;

  unsigned im_channel = image.channel();
  int im_height = image.height();
  int im_width = image.width();

  unsigned hstride = mstride[0];
  unsigned wstride = mstride[1];

  unsigned hdilation = dilation[0];
  unsigned wdilation = dilation[1];

  /// image considering padding
  unsigned im_eff_height = im_height + pt + pb;
  unsigned im_eff_width = im_width + pl + pr;
  image.setZero();

  int h_stride_end = im_eff_height - eff_k_height - pt;
  int w_stride_end = im_eff_width - eff_k_width - pl;

  /** @todo We need to implement way to use this kind of function to work inside
   * of Tensor. Then we could remove to access the getData or getValue which has
   * dependecy of data type.
   */
  auto apply_data = [&](auto *val) {
    using T = std::decay_t<decltype(*val)>;
    unsigned col_w = 0;
    for (int hs = -(int)pt; hs <= h_stride_end; hs += hstride) {
      for (int ws = -(int)pl; ws <= w_stride_end; ws += wstride) {
        unsigned col_h = 0;
        int patch_height_end = hs + eff_k_height;
        int patch_width_end = ws + eff_k_width;
        for (unsigned c = 0; c < im_channel; c++) {
          for (int h = hs; h < patch_height_end; h += hdilation) {
            if (h < 0 || im_height <= h) {
              col_h += k_width;
              continue;
            }
            for (int w = ws; w < patch_width_end; w += wdilation) {
              if (w < 0 || im_width <= w) {
                col_h++;
                continue;
              }

              val = image.getAddress<T>(0, c, h, w);
              *val += col_matrix.getValue<T>(0, 0, col_h, col_w);
              col_h++;
            }
          }
        }
        col_w++;
      }
    }
  };

  if (image.getDataType() == nntrainer::Tdatatype::FP32) {
    float val;
    apply_data(&val);
  }
#ifdef ENABLE_FP16
  else if (image.getDataType() == nntrainer::Tdatatype::FP16) {
    _FP16 val;
    apply_data(&val);
  }
#endif
  else {
    throw std::runtime_error("Not supported datatype");
  }
}

/**
 * @brief     reform the data to 2d matrix
 * a region is sampled considering @a padding, @a mstride of unit @a kdim
 * Each region is mapped to one column,
 * if channel mode, kernel channel is considered part of kernel feature
 * if not, kernel channel is consider part of output dimension
 *
 * @param[in] in input data
 * @param[in] kdim kernel dimesion for define number of row
 * @param[in] padding padding information
 * @param[in] mstride stride value : x, y direction
 * @param[in] dilation kernel dilation factor : x, y each
 * @param[out] out out tensor, padding set each time for now
 * @note if out is initialized tensor, setting padding is skipped.
 */
static void im2col(const Tensor &in, const TensorDim &kdim,
                   const std::array<unsigned int, 4> &padding,
                   const std::array<props::Stride, CONV2D_DIM> &mstride,
                   const std::array<props::Dilation, CONV2D_DIM> &dilation,
                   Tensor &out) {
  /// for channel last mode, this is deprecated for now, leaving here on
  /// purpose.
  /** @code
  //   ================ initialize part ====================
  //   out_height -= 2;
  //   out =
  //     Tensor(k_height * k_width, in.channel() * (out_height) *
  //     (out_width));
  //   unsigned int im_w = 0;
  //   ================ loop part ====================
  //   if (eff_k_height > height || eff_k_width > width)
  //     throw std::runtime_error("Kernel shape bigger than input shape");

  //   for (unsigned int c = 0; c < channel; ++c) {
  //     for (unsigned int hs = 0; hs <= height - eff_k_height; hs +=
  //     mstride[0]) {
  //       for (unsigned int ws = 0; ws <= width - eff_k_width; ws +=
  //       mstride[1]) {
  //         unsigned int im_h = 0;
  //         unsigned int patch_height_end = eff_k_height + hs;
  //         unsigned int patch_width_end = eff_k_width + ws;

  //         for (unsigned int h = hs; h < patch_height_end; h += dilation[0]) {
  //           if (h < ph || in_height + ph <= h) {
  //             im_h += k_width;
  //             continue;
  //           }

  //           for (unsigned int w = ws; w < patch_width_end; w += dilation[1])
  //           {
  //             if (w < pw || in_width + pw <= w) {
  //               im_h++;
  //               continue;
  //             }

  //             float val = in.getValue(0, c, h - ph, w - pw);
  //             out.setValue(0, 0, im_h, im_w, val);
  //             im_h++;
  //           }
  //         }
  //         im_w++;
  //       }
  //     }
  //   }
  */

  auto pt = padding[0];
  auto pb = padding[1];
  auto pl = padding[2];
  auto pr = padding[3];

  unsigned int channel = in.channel();
  int in_height = in.height();
  int in_width = in.width();
  unsigned int height = in_height + pt + pb;
  unsigned int width = in_width + pl + pr;
  unsigned int k_height = kdim.height();
  unsigned int k_width = kdim.width();

  /// effective kernel height considering dilation
  unsigned int eff_k_height = (k_height - 1) * dilation[0] + 1;
  /// effective kernel width considering dilation
  unsigned int eff_k_width = (k_width - 1) * dilation[1] + 1;

  unsigned int out_height = (height - eff_k_height) / mstride[0] + 1;
  unsigned int out_width = (width - eff_k_width) / mstride[1] + 1;

  out.reshape(
    TensorDim({out_height * out_width, in.channel() * k_height * k_width},
              in.getTensorType()));
  // float *out_data = out.getData();

  auto apply_data = [&](auto *out_data) {
    using T = std::decay_t<decltype(*out_data)>;
    int h_stride_end = height - eff_k_height - pt;
    int w_stride_end = width - eff_k_width - pl;

    /// get a patch, size of kernel
    /// hs is height_strided, ws is width_strided
    unsigned int owidth = out.width();
    unsigned int base_im_w = 0;
    for (int hs = -(int)pt; hs <= h_stride_end; hs += mstride[0]) {
      unsigned int base_im_h = 0;
      int patch_height_end = eff_k_height + hs;
      /// map the patch to a single line looping through channel
      // We need to optimize this padding & copy. May be use multi threads, or
      // SIMD
      for (unsigned int c = 0; c < channel; ++c) {
        for (int h = hs; h < patch_height_end; h += dilation[0]) {
          if (h < 0 || in_height <= h) {
            base_im_h += k_width;
            continue;
          }

          unsigned int im_w = base_im_w;
          for (int ws = -(int)pl; ws <= w_stride_end; ws += mstride[1]) {
            unsigned int im_h = base_im_h;
            int patch_width_end = eff_k_width + ws;

            for (int w = ws; w < patch_width_end; w += dilation[1]) {
              if (w < 0 || in_width <= w) {
                im_h++;
                continue;
              }
              out_data[im_w * owidth + im_h] = in.getValue<T>(0, c, h, w);
              im_h++;
            }
            im_w++;
          }
          base_im_h += k_width;
        }
      }
      base_im_w += out_width;
    }
  };

  if (out.getDataType() == nntrainer::Tdatatype::FP32) {
    float *out_data = out.getData<float>();
    apply_data(out_data);
  }
#ifdef ENABLE_FP16
  else if (out.getDataType() == nntrainer::Tdatatype::FP16) {
    _FP16 *out_data = out.getData<_FP16>();
    apply_data(out_data);
  }
#endif
  else {
    throw std::runtime_error("Not supported datatype");
  }
}
} // namespace

enum ConvParams { weight, bias };

Conv2DLayer::Conv2DLayer(
  const std::array<unsigned int, CONV2D_DIM * 2> &padding_) :
  LayerImpl(),
  padding(padding_),
  conv_props(props::FilterSize(), std::array<props::KernelSize, CONV2D_DIM>(),
             std::array<props::Stride, CONV2D_DIM>(), props::Padding2D(),
             std::array<props::Dilation, CONV2D_DIM>()) {
  wt_idx.fill(std::numeric_limits<unsigned>::max());
}

void Conv2DLayer::finalize(InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "Convolution layer takes only one input";

  const TensorDim &in_dim = context.getInputDimensions()[0];

  auto &weight_regularizer =
    std::get<props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<props::WeightRegularizerConstant>(*layer_impl_props);
  auto &weight_initializer =
    std::get<props::WeightInitializer>(*layer_impl_props);
  auto &weight_decay = std::get<props::WeightDecay>(*layer_impl_props);
  auto &bias_decay = std::get<props::BiasDecay>(*layer_impl_props);
  auto &bias_initializer = std::get<props::BiasInitializer>(*layer_impl_props);
  auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);

  unsigned int filter_size = std::get<props::FilterSize>(conv_props);
  auto &kernel_size =
    std::get<std::array<props::KernelSize, CONV2D_DIM>>(conv_props);
  auto &stride = std::get<std::array<props::Stride, CONV2D_DIM>>(conv_props);
  auto &dilation =
    std::get<std::array<props::Dilation, CONV2D_DIM>>(conv_props);

  auto weight_data_type = context.getWeightDataType();
  auto in_t_type = in_dim.getTensorType();
  in_t_type.data_type = weight_data_type;

  // A quantized weight is only well defined when its per-channel scale vector
  // is keyed on the output channel, and that fixes the shape it has to be
  // requested in. QINT8 under PER_CHANNEL_AFFINE has scale_size() == width(),
  // so the requirement is exactly a channel last weight; requested channel
  // first as (filter_size, in_ch, kh, kw), width() is kw and the scale vector
  // silently describes the kernel instead of the output channels. The QINT16
  // (height()) and QINT4 (height() * width() / 32) layouts key on a kernel axis
  // in either format, so they stay rejected.
  //
  // What the weight is stored in and what computes it are separate questions,
  // answered in two places. Storage is answered here: a QINT8 weight is stored
  // quantized -- one int8 per tap plus one fp32 scale per output channel,
  // instead of one fp32 per tap -- and that saving does not depend on which
  // kernel eventually reads it. Computation is answered at the top of
  // forwarding(), where a channel last weight is refused until the int8 kernel
  // lands. Admitting the weight here pins the storage without claiming the
  // computation.
  //
  // FP16 is accepted, but forwarding() only runs it against an FP16 activation
  // (W16A16) — a pre-existing constraint unrelated to this guard.
  const bool is_float_weight = weight_data_type == TensorDim::DataType::FP32 ||
                               weight_data_type == TensorDim::DataType::FP16;
  // The weight follows the activation layout, so a channel last activation
  // dimension is what lets the weight be requested channel last. The scheme
  // clause is implicit: a conv2d weight is always requested through the weight
  // pool, which builds tensors PER_CHANNEL_AFFINE, and per-channel affine is
  // the only scheme defined for a weight.
  const bool store_weight_quantized =
    weight_data_type == TensorDim::DataType::QINT8 &&
    in_dim.getTensorType().format == TensorDim::Format::NHWC;
  NNTR_THROW_IF(!is_float_weight && !store_weight_quantized,
                std::invalid_argument)
    << "[Conv2D] a quantized conv2d weight is supported only as a QINT8 weight "
       "of a channel last model: requested as (filter, in_ch, kh, kw) the "
       "per-channel scale axis is the kernel width and not the output channel. "
       "Use an FP32/FP16 weight, or a QINT8 weight with a channel last (NHWC) "
       "model.";

  // Requesting the weight channel last is what keys its per-channel scale
  // vector on the output channel: scale_size() for QINT8 is width(), so only a
  // spelling whose width is filter_size gives one scale per output channel.
  // (in_ch, kh, kw, filter_size) also carries every tap, which a
  // (1, kh, kw, filter_size) spelling could not for more than one input
  // channel. The dot below still consumes the classical operand, because that
  // is the operand im2col builds its columns against, so kernel_dim keeps that
  // shape and forwarding() converts the stored weight into it.
  TensorDim kernel_dim(filter_size, in_dim.channel(), kernel_size[0],
                       kernel_size[1], in_t_type);

  // A quantized weight has no fp32 mirror for an optimizer to update, and
  // CharTensor accepts no weight initializer at all, so a training model cannot
  // even allocate one. Rejected here rather than as an initializer error from
  // the tensor pool, and rather than letting calcGradient() write int8 storage.
  const bool qint8_weight_in_training =
    store_weight_quantized &&
    context.getExecutionMode() != ml::train::ExecutionMode::INFERENCE;
  NNTR_THROW_IF(qint8_weight_in_training, std::invalid_argument)
    << "[Conv2D] a QINT8 conv2d weight is inference only: it has no fp32 "
       "mirror for an optimizer to update. Run the model in inference mode, "
       "or use an FP32/FP16 weight.";

  // The bias is added straight to the activation, so it cannot follow the
  // weight into a quantized dtype. Following FullyConnectedLayer's convention
  // for a quantized weight (fc_layer.cpp, "Bias Dimension"), it is requested as
  // FP32, the dtype it is stored in on disk, so an FP16 activation does not
  // reinterpret those bytes. A float weight keeps the dtype it already had.
  TensorDim bias_dim = TensorDim(1, filter_size, 1, 1, in_t_type);
  if (store_weight_quantized) {
    bias_dim.setDataType(TensorDim::DataType::FP32);
  }

  padding = std::get<props::Padding2D>(conv_props)
              .compute(in_dim, kernel_dim, {stride[0], stride[1]},
                       {dilation[0], dilation[1]});

  wt_idx[ConvParams::weight] = context.requestWeight(
    store_weight_quantized
      ? detail::channelLastKernelDim(kernel_size[0], in_dim.channel(),
                                     kernel_size[1], filter_size, in_t_type)
      : kernel_dim,
    // A quantized weight is not trainable: an int8 weight has no fp32 mirror
    // for an optimizer to update, so it must not be requested as if it had. It
    // is unreachable here -- the training mode check above already refused this
    // combination -- but the invariant holds rather than rests on that order.
    weight_initializer, weight_regularizer, weight_regularizer_constant,
    weight_decay, "filter", !store_weight_quantized, 0);

  if (disable_bias.empty() || disable_bias.get() == false) {
    wt_idx[ConvParams::bias] =
      context.requestWeight(bias_dim, bias_initializer, WeightRegularizer::NONE,
                            1.0f, bias_decay, "bias", true, 0);
  }

  // this output_dim must be the same with dimension of hidden
  unsigned int eff_in_height = in_dim.height() + padding[0] + padding[1];
  unsigned int eff_in_width = in_dim.width() + padding[2] + padding[3];

  unsigned int eff_k_height = (kernel_size[0] - 1) * dilation[0] + 1;
  unsigned int eff_k_width = (kernel_size[1] - 1) * dilation[1] + 1;

  TensorDim out_dim;
  out_dim.batch(in_dim.batch());
  out_dim.channel(filter_size);
  out_dim.height((eff_in_height - eff_k_height) / stride[0] + 1);
  out_dim.width((eff_in_width - eff_k_width) / stride[1] + 1);

  out_dim.setTensorType(in_dim.getTensorType());

  context.setOutputDimensions({out_dim});

  NNTR_THROW_IF(eff_in_height < kernel_size[0] || eff_in_width < kernel_size[1],
                std::invalid_argument)
    << "Failed to initialize: in size + padding is smaller than effective "
       "kernel";

  unsigned int IM = std::numeric_limits<int>::max();

  NNTR_THROW_IF(eff_in_height - padding[0] - kernel_size[0] > IM ||
                  eff_in_width - padding[2] - kernel_size[1] > IM,
                std::invalid_argument)
    << "Failed to initialize: Calculated patch end is over int max";
}

void Conv2DLayer::forwarding(RunLayerContext &context, bool training) {
  int status = ML_ERROR_NONE;

  unsigned int filter_size = std::get<props::FilterSize>(conv_props);
  auto &stride = std::get<std::array<props::Stride, CONV2D_DIM>>(conv_props);
  auto &dilation =
    std::get<std::array<props::Dilation, CONV2D_DIM>>(conv_props);

  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);

  Tensor &filter_kernel = context.getWeight(wt_idx[ConvParams::weight]);

  // Two ways this weight can be unusable, refused in the order they are
  // discovered, so that each reports its own reason rather than the first one
  // swallowing the other.
  //
  // First, a quantized weight whose per-channel scales are all zero. A zero
  // scale is not something quantize() produces -- it is what it refuses to
  // divide by -- so a weight that arrives that way is not a quantized weight,
  // and multiplying every tap by zero would hand the layer an output of all
  // zeros that nothing else marks as wrong. One way to arrive here is a reader
  // that does not know the block layout: a quantized weight is written as the
  // quantization scheme, then the int8 taps, then one fp32 scale per output
  // channel, and a reader that reads only the taps' worth of bytes from the
  // front of the block stops before the scale area. What it leaves there is
  // whatever the allocation held, so this check catches that case when the area
  // reads as zero; the second refusal below catches it otherwise. Either way a
  // weight that was never given its scales does not reach the dot.
  if (filter_kernel.getDataType() == TensorDim::DataType::QINT8) {
    const float *scales = filter_kernel.getScale<float>();
    const size_t num_scales = filter_kernel.scale_size();
    const bool all_scales_zero =
      num_scales != 0 && std::all_of(scales, scales + num_scales,
                                     [](float scale) { return scale == 0.0f; });
    NNTR_THROW_IF(all_scales_zero, std::runtime_error)
      << "[Conv2D] QINT8 conv2d weight '"
      << context.getWeightName(wt_idx[ConvParams::weight])
      << "' has no scale factors: a quantized weight is stored with the "
         "quantization scheme and one scale per output channel around its int8 "
         "taps, and with every scale zero every tap of this layer would "
         "dequantize to zero.";
  }

  // Second, a layout nothing below computes. A QINT8 weight is defined to be
  // stored channel last -- finalize() requests it so, because only there does
  // the per-channel scale vector key on the output channel -- and neither path
  // here handles that: the channel first path below builds its columns for a
  // channel first kernel, and the channel last path has no dtype branch at all,
  // so it would hand the int8 taps to the floating point dot as if they were
  // floats. Storing the weight is this layer's business; computing it channel
  // last belongs to the int8 kernel that reads the taps and their scales
  // directly, which is what the @todo below waits on.
  NNTR_THROW_IF(filter_kernel.getDataType() == TensorDim::DataType::QINT8 &&
                  input_.getDim().getFormat() ==
                    ml::train::TensorDim::Format::NHWC,
                std::runtime_error)
    << "[Conv2D] channel last QINT8 weight computation is not implemented yet: "
       "the weight is stored (kh, in_ch, kw, filter) so that its per-channel "
       "scale vector keys on the output channel, and no path here computes a "
       "channel last kernel. An int8 kernel is the follow up.";

  /** Calculate Convolution 2D
   *
   * This is the 2D Matrix Shape [ height ] x [ width ]
   *   . Height : filter_size
   *   . Width  : Input Channel * Kernel_size[0] * Kernel_size[1]
   *
   *                              imKernel
   *                        +------|------|------+
   *                        |------|------|------|
   * [filter_size (height)] |------|------|------|
   *                        |------|------|------|
   *                        +------|------|------+
   *                     [Input Channel * Kernel_size[0]
   *                       * Kernel_size[1] (width)]
   *
   *
   * After im2Col with channel_mode true (in : input)
   *
   * This is the 2D Matrix Shape [ height ] x [ width ]
   *   . Height : Input Channel * Kernel_size[0] * Kernel_size[1]
   *   . Width  : output_dim.height * output_dim.width
   *
   *                      +-|-|-|-|      |-|-|-|-+
   *   [Input Channel     | | | | |      | | | | |
   *   * Kernel_size[0]   |_|_|_|_|      |_|_|_|_|
   *  * Kenel_size[1]     | | | | | .... | | | | |
   *    (height)]         |_|_|_|_|      |_|_|_|_|
   *                      | | | | |      | | | | |
   *                      +_|_|_|_|      |_|_|_|_+
   *                     [ output_dim.height
   *                      * output_dim.width (width) ]
   *
   * Output Dimention
   *   -> [Channel ( = filter_size = output_dim.channel )]
   *       x [output_dim.height x output_dim.width]
   */
  const TensorDim &in_dim = input_.getDim();
  const TensorDim &out_dim = hidden_.getDim();

  // The dot below runs in floating point and consumes the classical
  // (filter_size, in_ch, kh, kw) operand, so a quantized weight is turned into
  // that operand first. This is FullyConnectedLayer::forwarding()'s arrangement
  // — dequantize the weight, then dot — with the one repack the two weight
  // layouts differ by. The operand is built here and dropped when this call
  // returns rather than kept alongside the weight, so between calls the
  // quantized weight is the only stored copy of the weights; during a call this
  // operand is a second, floating point, copy of this one kernel.
  //
  ///@todo dequantizing per step is the cost of not having an int8 kernel yet;
  /// this moves into the kernel once the dot can consume the weight and its
  /// scales directly.
  // Where this stands until then: finalize() admits a QINT8 weight only for a
  // channel last model, and the guard at the top of this function refuses to
  // compute one, so this branch is not reached by any model that initializes
  // today. It is kept because dequantizing is what this weight is computed by
  // -- the channel last kernel that removes the refusal reads the same operand
  // -- and its mapping from stored taps and per-channel scales to that operand
  // is pinned tap by tap against hand computed values, which a path no model
  // reaches could not be tested through.
  Tensor dequantized;
  if (filter_kernel.getDataType() == TensorDim::DataType::QINT8) {
    dequantized =
      detail::dequantizeKernel(filter_kernel, hidden_.getDataType());
  }
  Tensor &kernel = dequantized.empty() ? filter_kernel : dequantized;

  const TensorDim &filter_dim = kernel.getDim();
  TensorDim filter_dim_squeezed{kernel.batch(),
                                kernel.getDim().getFeatureLen()};

  filter_dim_squeezed.setTensorType(kernel.getTensorType());

  kernel.reshape(filter_dim_squeezed);

  /**
   * Below sets the pad area values to zero
   * it is faster to do this way than seting selective area to zero
   */
  auto forwarding_job = [&](unsigned int s, unsigned int e, unsigned int pid,
                            void *user_data) {
    Tensor result = Tensor(calcCol2ImOutputDim(out_dim, filter_dim));
    result.setZero();
    for (unsigned int b = s; b < e; ++b) {
      Tensor out = hidden_.getBatchSlice(b, 1);
      out.reshape({filter_size, out_dim.width() * out_dim.height()});
      Tensor in_sub = input_.getBatchSlice(b, 1);

      im2col(in_sub, filter_dim, padding, stride, dilation, result);
      // filter kernel is (K, CRS), result is (CRS, OH*OW)
      kernel.dot(result, out, false, true);
    }
    result.deallocate();
  };

  auto workers = ParallelBatch(forwarding_job, in_dim.batch(), nullptr);

  if (workers.getNumWorkers() > 1) {
    workers.run();
  } else {
    forwarding_job(0, in_dim.batch(), 0, nullptr);
  }

  kernel.reshape(filter_dim);
  if (auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);
      disable_bias.empty() || disable_bias.get() == false) {
    Tensor &bias_kernel = context.getWeight(wt_idx[ConvParams::bias]);
    // The bias of a quantized weight is stored FP32 while the activation it is
    // added to keeps its own dtype, so the two can differ -- under an FP16
    // activation the fp32 bytes have to be cast, not reinterpreted. Not reached
    // today: it is keyed on a dequantized weight, which the guard at the top of
    // this function keeps out of this path, and it becomes live exactly when
    // that refusal is lifted.
    if (!dequantized.empty() &&
        bias_kernel.getDataType() != hidden_.getDataType()) {
      Tensor bias_cast = bias_kernel.clone(hidden_.getDataType());
      status = hidden_.add_i(bias_cast);
    } else {
      status = hidden_.add_i(bias_kernel);
    }
    if (status != ML_ERROR_NONE) {
      throw std::invalid_argument("[Conv2D] adding bias failed");
    }
  }
}

void Conv2DLayer::calcDerivative(RunLayerContext &context) {
  unsigned int filter_size = std::get<props::FilterSize>(conv_props);
  auto &stride = std::get<std::array<props::Stride, CONV2D_DIM>>(conv_props);
  auto &dilation =
    std::get<std::array<props::Dilation, CONV2D_DIM>>(conv_props);

  const Tensor &derivative = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &input_derivative = context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  Tensor &filter_kernel = context.getWeight(wt_idx[ConvParams::weight]);

  TensorDim filter_dim = filter_kernel.getDim();
  TensorDim filter_dim_squeezed{filter_kernel.batch(),
                                filter_kernel.getDim().getFeatureLen()};

  filter_kernel.reshape(filter_dim_squeezed);

  /// for each batch
  /// filter_kernel^T X derivaitive  -> column matrix
  /// col2im(column matrix) to reconstruct the original image

  auto compute_derivative = [&](unsigned int s, unsigned int e,
                                unsigned int pid, void *user_data) {
    Tensor result =
      Tensor(calcCol2ImOutputDim(derivative.getDim(), filter_dim));

    for (unsigned int b = s; b < e; ++b) {
      Tensor deriv_sub = derivative.getBatchSlice(b, 1);
      Tensor in_deriv_sub = input_derivative.getBatchSlice(b, 1);
      deriv_sub.reshape(
        {filter_size, derivative.width() * derivative.height()});
      // filter_kernel is (K, CRS), deriv_sub is (K, OH*OW), result is (CRS,
      // OH*OW)
      filter_kernel.dot(deriv_sub, result, true, false);
      col2im(result, filter_dim, padding, stride, dilation, in_deriv_sub);
      // in_derv_sub is (C,H,W)
    }
    result.deallocate();
  };

  auto workers = ParallelBatch(compute_derivative, derivative.batch(), nullptr);

  if (workers.getNumWorkers() > 1) {
    workers.run();
  } else {
    compute_derivative(0, derivative.batch(), 0, nullptr);
  }

  filter_kernel.reshape(filter_dim);
}

void Conv2DLayer::calcGradient(RunLayerContext &context) {
  unsigned int filter_size = std::get<props::FilterSize>(conv_props);
  auto &stride = std::get<std::array<props::Stride, CONV2D_DIM>>(conv_props);
  auto &dilation =
    std::get<std::array<props::Dilation, CONV2D_DIM>>(conv_props);

  const Tensor &derivative = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);

  Tensor &delK = context.getWeightGrad(wt_idx[ConvParams::weight]);
  delK.setZero();

  TensorDim filter_dim = delK.getDim();
  TensorDim filter_dim_squeezed{filter_dim.batch(), filter_dim.getFeatureLen()};

  delK.reshape(filter_dim_squeezed);

  /**
   * no need to set zero for im2col_result, as its lifespan is ITERATION,
   * so its zero padded values will still be zero
   */

  TensorDim out_dim_squeezed{filter_size,
                             derivative.width() * derivative.height()};
  auto workers = ParallelBatch(input_.batch());
  /// input -(im2col)-> column_matrix -> filter x (column_matrix) = output
  /// so delK = dy x column_matrix ^ T;
  if (workers.getNumWorkers() > 1) {

    TensorDim delK_ext = filter_dim_squeezed;
    delK_ext.batch(input_.batch());

    Tensor delK_par = Tensor(delK_ext);
    delK_par.setZero();

    auto calc_grad_job = [&](unsigned int s, unsigned int e, unsigned int pid,
                             void *user_data) {
      Tensor result =
        Tensor(calcCol2ImOutputDim(derivative.getDim(), filter_dim));
      result.setZero();
      for (unsigned int b = s; b < e; ++b) {
        Tensor deriv_sub = derivative.getBatchSlice(b, 1);
        Tensor delK_sub = delK_par.getBatchSlice(b, 1);
        deriv_sub.reshape(out_dim_squeezed);

        Tensor in_sub = input_.getBatchSlice(b, 1);

        /**
         * @todo this result can be cached from the forward iteration at the
         * expense of memory. In this case, memory of im2col_result must be
         * saved for the whole batch. try this while benchmarking.
         */
        // deriv_sub is (K, OH*OW) and result is (CRS, OH*OW)
        im2col(in_sub, filter_dim, padding, stride, dilation, result);
        deriv_sub.dot(result, delK_sub, false, false);
      }
      result.deallocate();
    };

    workers.setCallback(calc_grad_job, nullptr);

    workers.run();

    for (unsigned int b = 0; b < input_.batch(); ++b) {
      Tensor delK_sub = delK_par.getBatchSlice(b, 1);
      delK.add_i(delK_sub);
    }

  } else {
    Tensor result =
      Tensor(calcCol2ImOutputDim(derivative.getDim(), filter_dim));
    result.setZero();

    for (unsigned int b = 0; b < input_.batch(); ++b) {
      Tensor deriv_sub = derivative.getBatchSlice(b, 1);
      deriv_sub.reshape(out_dim_squeezed);

      Tensor in_sub = input_.getBatchSlice(b, 1);

      /**
       * @todo this result can be cached from the forward iteration at the
       * expense of memory. In this case, memory of im2col_result must be saved
       * for the whole batch. try this while benchmarking.
       */
      im2col(in_sub, filter_dim, padding, stride, dilation, result);
      deriv_sub.dot(result, delK, false, false, b == 0 ? 0.0f : 1.0f);
    }
    result.deallocate();
  }
  delK.reshape(filter_dim);
  if (auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);
      disable_bias.empty() || disable_bias.get() == false) {
    Tensor &delBias = context.getWeightGrad(wt_idx[ConvParams::bias]);
    delBias.setZero();
    derivative.sum({0, 2, 3}, delBias);
  }
}

void Conv2DLayer::exportTo(Exporter &exporter,
                           const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(conv_props, method, this);
}

void Conv2DLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, conv_props);
  LayerImpl::setProperty(remain_props);
}

} /* namespace nntrainer */

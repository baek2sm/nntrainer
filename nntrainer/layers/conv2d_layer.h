// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   conv2d_layer.h
 * @date   01 June 2020
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Convolution 2D Layer Class for Neural Network
 *
 */

#ifndef __CONV2D_LAYER_H_
#define __CONV2D_LAYER_H_
#ifdef __cplusplus

#include <memory.h>

#include <common_properties.h>
#include <layer_impl.h>

namespace nntrainer {

constexpr const unsigned int CONV2D_DIM = 2;

/**
 * @class   Convolution 2D Layer
 * @brief   Convolution 2D Layer
 */
class Conv2DLayer : public LayerImpl {
public:
  /**
   * @brief     Constructor of Conv 2D Layer
   */
  Conv2DLayer(const std::array<unsigned int, CONV2D_DIM * 2> &padding_ = {
                0, 0, 0, 0});

  /**
   * @brief     Destructor of Conv 2D Layer
   */
  ~Conv2DLayer() override;

  /**
   *  @brief  Move constructor of Conv 2D Layer.
   *  @param[in] Conv2dLayer &&
   */
  Conv2DLayer(Conv2DLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs Conv2DLayer to be moved.
   */
  Conv2DLayer &operator=(Conv2DLayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(InitLayerContext &context) override;

  /**
   * @copydoc Layer::initialize(RunLayerContext &context)
   */
  void initialize(RunLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(RunLayerContext &context) override;

  /**
   * @copydoc Layer::calcGradient(RunLayerContext &context)
   */
  void calcGradient(RunLayerContext &context) override;

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return Conv2DLayer::type; };

  /**
   * @copydoc Layer::supportBackwarding()
   */
  bool supportBackwarding() const override { return true; }

  using Layer::setProperty;

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @brief Load Q4_0 quantized weights from external buffer
   * @param bytes pointer to Q4_0 byte buffer
   * @param size size of the buffer in bytes
   */
  void loadQ40Weights(const uint8_t *bytes, size_t size);

  /**
   * @brief Get reference to Q4_0 buffer (const)
   * @return const reference to the Q4_0 buffer
   */
  const std::vector<uint8_t> &getQ40Buffer() const {
    return filter_q4_0_buffer;
  }

  /**
   * @brief Get original filter dimension (for Q4_0_PRELOAD mode)
   * @return TensorDim of the original filter shape
   */
  TensorDim getFilterOrigDim() const { return filter_q4_0_orig_dim; }

  /* TO DO : support keras type of padding */
  /* enum class PaddingType { */
  /*   full = 0, */
  /*   same = 1, */
  /*   valid = 2, */
  /*   unknown = 3, */
  /* }; */

  static constexpr const char *type = "conv2d";

private:
  /**
   * @brief Extract one im2col row and quantize to Q8_0 in-place.
   *
   * @param input  FP32 input tensor data (single batch, NCHW layout)
   * @param dst_row  destination buffer: (K/32) block_q8_0 blocks (34 B each)
   * @param m  output spatial position index (row in the im2col matrix)
   * @param c_in  number of input channels
   * @param h_in  input height
   * @param w_in  input width
   * @param kH  kernel height
   * @param kW  kernel width
   * @param stride_h  vertical stride
   * @param stride_w  horizontal stride
   * @param pad_h  top padding
   * @param pad_w  left padding
   * @param K  total patch length = c_in * kH * kW (must be divisible by 32)
   */
  static void extract_im2col_row_q8_0(const float *input, void *dst_row, int m,
                                      int c_in, int h_in, int w_in, int kH,
                                      int kW, int stride_h, int stride_w,
                                      int pad_h, int pad_w, int K);

  /**
   * @brief Conv2D-private property for Q4_0 weight quantization.
   *
   * key = "conv_weight_quant" (distinct from framework's "weight_dtype" to
   * avoid layer_node intercepting it and trying to allocate a Q4_0 tensor for
   * a 4D conv filter, which the framework does not support).
   * Accepted values: "FP32" (default) or "Q4_0".
   */
  struct Conv2DWeightQuant final : public nntrainer::Property<std::string> {
    using prop_tag = str_prop_tag;
    static constexpr const char *key = "conv_weight_quant";
  };

  std::array<unsigned int, CONV2D_DIM * 2> padding;
  std::tuple<props::FilterSize, std::array<props::KernelSize, CONV2D_DIM>,
             std::array<props::Stride, CONV2D_DIM>, props::Padding2D,
             std::array<props::Dilation, CONV2D_DIM>, props::ConvGroups,
             Conv2DWeightQuant>
    conv_props;

  std::array<unsigned int, 5> wt_idx; /**< indices of the weights and tensors */

  /**
   * @brief Q4_0 quantized weight buffer (owned by conv2d layer when
   * weight_dtype=Q4_0)
   */
  std::vector<uint8_t> filter_q4_0_buffer;
  unsigned int filter_q4_0_M{0}; /**< M dim for repacked filter */
  unsigned int filter_q4_0_N{0}; /**< N dim for repacked filter */
  bool use_q4_0_gemm{false};     /**< flag to use Q4_0 GEMM path */

  /**
   * @brief Q4_0_PRELOAD mode flag — when true, FP32 weight allocation is
   * skipped and weights are loaded via loadQ40Weights()
   */
  bool use_q4_0_preload{false};

  /**
   * @brief Original filter shape for Q4_0_PRELOAD mode (used for quantizer)
   */
  TensorDim filter_q4_0_orig_dim;
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __CONV2D_LAYER_H__ */

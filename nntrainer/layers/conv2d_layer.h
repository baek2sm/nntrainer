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
   * @brief Conv2D-private property for Q4_0 weight quantization.
   *
   * key = "conv_weight_quant" (distinct from framework's "weight_dtype" to
   * avoid layer_node intercepting it and trying to allocate a Q4_0 tensor for
   * a 4D conv filter, which the framework does not support).
   * Accepted values: "FP32" (default) or "Q4_0".
   */
  struct Conv2DWeightQuant final
    : public nntrainer::Property<std::string> {
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
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __CONV2D_LAYER_H__ */

// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   conv2d_layer.h
 * @date   01 June 2020
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Convolution Layer Class for Neural Network
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
  ~Conv2DLayer() = default;

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
   * @copydoc Layer::setBatch(RunLayerContext &context, unsigned int batch)
   */
  void setBatch(RunLayerContext &context, unsigned int batch) override;

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::save(std::ofstream &file, RunLayerContext &run_context,
   * bool opt_var, ml::train::ExecutionMode mode, bool trainable,
   * TensorDim::DataType dtype, ml::train::ISA target_isa)
   *
   * @note Overridden so a conv filter can be quantized to Q4_0 as a matmul
   * weight. The FP32 filter is stored [out_ch, in_ch, kh, kw], which is already
   * row-major [out_ch, CRS] (CRS = in_ch*kh*kw) = [N, K] with N=out_ch rows of
   * K=CRS — exactly the layout quantize_q4_0 wants, so (unlike the FC path in
   * the base class) it needs no transpose. Ineligible filters (out_ch or CRS
   * not 32-aligned) and the bias stay FP32.
   */
  void save(
    std::ofstream &file, RunLayerContext &run_context, bool opt_var,
    ml::train::ExecutionMode mode, bool trainable,
    ml::train::TensorDim::DataType dtype = ml::train::TensorDim::DataType::NONE,
    ml::train::ISA target_isa = ml::train::ISA::DEFAULT) const override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return Conv2DLayer::type; };

  /**
   * @copydoc Layer::supportBackwarding()
   */
  bool supportBackwarding() const override { return true; }

  /**
   * @copydoc Layer::supportInt8ActInput()
   * @note W4A8 (U5b, "1x1 first" scope): conv2d consumes an int8 (Q8_0_TW)
   * activation edge *directly* — the flat int8 payload is repacked into the
   * SMMLA block_q8_0x4 layout and fed straight to the int8xint4 GEMM, with no
   * FP16 dequant/re-quantize round trip. Only the 1x1 stride-1 quantized GEMM
   * path can do this today, so int8 input capability is restricted to a 1x1
   * stride-1 kernel whose filter is Q4_0-quantized (out_ch %32==0 — otherwise
   * forwarding uses the FP im2col GEMM, which cannot consume a Q8_0_TW input);
   * 3x3 / depthwise / non-32-aligned convs stay FP16. The edge is still only
   * promoted to int8 when the producer is int8-out-capable and the static scale
   * is registered (§5.7 conditions 1&3); capability alone forms no int8 edge.
   */
  bool supportInt8ActInput() const override;

  /**
   * @copydoc Layer::supportInt8ActOutput()
   * @note W4A8 (U5b, "1x1 first" scope): conv2d emits an int8 (Q8_0_TW)
   * activation edge via the §5.2 output requant epilogue, but only for a 1x1
   * stride-1 kernel whose output-channel count is a multiple of 32 (block_q8_0
   * granularity). This keeps every int8 edge channel-count %32==0, so the
   * consumer's block_q8_0x4 GEMM is always available (no unpackable edge). See
   * supportInt8ActInput() for the promotion conditions (a registered scale is
   * still required, §5.7 condition 3).
   */
  bool supportInt8ActOutput() const override;

  /**
   * @copydoc Layer::hasActivationScale()
   * @note W4A8 (U5b): true when a positive props::ActivationScale was injected
   * for this conv's output edge (the calibration scale table). §5.7 condition 3
   * gates int8 promotion on this.
   */
  bool hasActivationScale() const override;

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
  std::array<unsigned int, CONV2D_DIM * 2> padding;
  std::tuple<props::FilterSize, std::array<props::KernelSize, CONV2D_DIM>,
             std::array<props::Stride, CONV2D_DIM>, props::Padding2D,
             std::array<props::Dilation, CONV2D_DIM>, props::ConvGroups,
             props::FusedActivation, props::ActivationScale, props::PreactScale,
             props::InputActivationScale>
    conv_props;

  std::array<unsigned int, 5> wt_idx; /**< indices of the weights and tensors */
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __CONV2D_LAYER_H__ */

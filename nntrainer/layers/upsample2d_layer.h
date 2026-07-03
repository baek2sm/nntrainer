// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 heka1024 <heka1024@gmail.com>
 *
 * @file   upsample2d_layer.h
 * @date   15 June 2024
 * @brief  This is Upsample2d Layer Class of Neural Network
 * @see    https://github.com/nntrainer/nntrainer
 * @author heka1024 <heka1024@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __UPSAMPLE2D_LAYER_H__
#define __UPSAMPLE2D_LAYER_H__
#ifdef __cplusplus

#include <common_properties.h>
#include <layer_impl.h>

#include <node_exporter.h>

namespace nntrainer {

constexpr const unsigned int UPSAMPLE2D_DIM = 2;

/**
 * @class   Upsample2dLayer
 * @brief   Upsamle 2d layer
 */
class Upsample2dLayer : public Layer {
public:
  /**
   * @brief Construct a new Upsample layer object
   *
   */
  Upsample2dLayer();

  /**
   * @brief Destroy the Upsample layer object
   *
   */
  ~Upsample2dLayer() {}

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(nntrainer::InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(nntrainer::RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc bool supportBackwarding() const
   */
  bool supportBackwarding() const override { return true; };

  /**
   * @copydoc Layer::supportInt8ActInput()
   * @note W4A8 static Q8_0: nearest upsample is a pure passthrough — every
   * output element is a verbatim copy of an input element, so on the int8
   * (Q8_0_TW) payload the replicated codes stay valid under the same per-tensor
   * scale (calibration keeps it unchanged across the upsample) with no
   * dequant/requant. Bilinear interpolation would need a real DQ->Q, so int8 is
   * only advertised for the nearest mode.
   */
  bool supportInt8ActInput() const override;

  /**
   * @copydoc Layer::supportInt8ActOutput()
   */
  bool supportInt8ActOutput() const override;

  /**
   * @copydoc Layer::isActivationPassthrough()
   */
  bool isActivationPassthrough() const override;

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ExportMethods method)
   */
  void exportTo(nntrainer::Exporter &exporter,
                const ml::train::ExportMethods &method) const override{};

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return Upsample2dLayer::type; };

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  static constexpr const char *type = "upsample2d";

private:
  std::tuple<props::UpsampleMode, std::array<props::KernelSize, UPSAMPLE2D_DIM>>
    upsample2d_props; /* mode, size of kernel */
};
} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __UPSAMPLE2D_LAYER_H__ */

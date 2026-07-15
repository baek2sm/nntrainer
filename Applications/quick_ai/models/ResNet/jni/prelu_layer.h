// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   prelu_layer.h
 * @date   03 July 2026
 * @brief  custom parametric ReLU (PReLU) layer
 * @see    https://github.com/nntrainer/nntrainer
 */

#ifndef __PRELU_LAYER_H__
#define __PRELU_LAYER_H__

#include <layer_context.h>
#include <layer_devel.h>
#include <node_exporter.h>

namespace custom {

/**
 * @brief PReLU activation layer with per-channel learnable negative slope.
 *
 * Forward: f(x) = x for x >= 0, f(x) = alpha * x for x < 0.
 * Alpha has one parameter per input channel.
 */
class PReLULayer final : public nntrainer::Layer {
public:
  /**
   * @brief Construct a new PReLU Layer object
   */
  PReLULayer() : Layer(), weight_idx(0) {}

  /**
   * @brief Destroy the PReLU Layer object
   */
  ~PReLULayer() {}

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
   * @copydoc Layer::exportTo(Exporter &exporter, ExportMethods method)
   */
  void exportTo(nntrainer::Exporter &exporter,
                const ml::train::ExportMethods &method) const override {}

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return PReLULayer::type; };

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override {}

  static constexpr const char *type = "prelu";

private:
  unsigned int weight_idx; /**< index of per-channel alpha weight */
};

} // namespace custom

#endif /* __PRELU_LAYER_H__ */

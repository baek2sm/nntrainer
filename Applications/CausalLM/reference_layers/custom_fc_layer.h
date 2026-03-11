// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Hyeonseok Lee <hs89.lee@samsung.com>
 *
 * @file   custom_fc_layer.h
 * @date   17 Nov 2023
 * @brief  Implementation of custom fc layer
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __CUSTOM_FC_LAYER_H__
#define __CUSTOM_FC_LAYER_H__

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <custom_common_properties.h>
#include <layer_context.h>
#include <layer_impl.h>
#include <node_exporter.h>
#include <utility>

namespace custom {

/**
 * @brief A Custom FC layer for llama.
 *
 */
WIN_EXPORT class CustomFCLayer : public nntrainer::LayerImpl {
public:
  /**
   * @brief Construct a new Custom FC layer object
   *
   */
  WIN_EXPORT CustomFCLayer();

  /**
   * @brief Destroy the Custom FC layer object
   *
   */
  WIN_EXPORT ~CustomFCLayer() {}

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::incremental_forwarding(RunLayerContext &context, unsigned
   * int from, unsigned int to, bool training)
   */
  WIN_EXPORT void incremental_forwarding(nntrainer::RunLayerContext &context,
                              unsigned int from, unsigned int to,
                              bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc Layer::calcGradient(RunLayerContext &context)
   */
  //   void calcGradient(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc bool supportBackwarding() const
   */
  WIN_EXPORT bool supportBackwarding() const override { return true; };

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ExportMethods method)
   */
  WIN_EXPORT void exportTo(nntrainer::Exporter &exporter,
                const ml::train::ExportMethods &method) const override{};

  /**
   * @copydoc Layer::getType()
   */
  WIN_EXPORT const std::string getType() const override { return CustomFCLayer::type; };

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "custom_fully_connected";

private:
  std::tuple<nntrainer::props::Unit, props::SmartReply> custom_fc_props; /**< fc layer properties :
                                              unit - number of output neurons */
  std::array<unsigned int, 2> weight_idx; /**< indices of the weights */
};

} // namespace custom

#endif /* __CUSTOM_FC_LAYER_H__ */

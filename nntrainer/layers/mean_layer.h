// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   mean_layer.h
 * @date   07 April 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is cast layer class (operation layer)
 */

#ifndef __MEAN_LAYER_H__
#define __MEAN_LAYER_H__
#ifdef __cplusplus

#include <common_properties.h>
#include <layer_devel.h>
#include <operation_layer.h>

namespace nntrainer {

/**
 * @class Mean Layer
 * @brief Mean Layer
 */
class MeanLayer : public UnaryOperationLayer {
public:
  /**
   * @brief Constructor of Mean Layer
   */
  MeanLayer() :
    UnaryOperationLayer(),
    mean_props(props::Print(), props::Axis()),
    support_backwarding(true),
    axis(-1) {}

  /**
   * @brief Destructor of Mean Layer
   */
  ~MeanLayer(){};

  /**
   *  @brief  Move constructor of Mean Layer.
   *  @param[in] MeanLayer &&
   */
  MeanLayer(MeanLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs MeanLayer to be moved.
   */
  MeanLayer &operator=(MeanLayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(InitLayerContext &context) final;

  /**
   * @brief forwarding operation for mean
   *
   * @param input input tensor
   * @param hidden tensor to store the result value
   */
  void forwarding_operation(const Tensor &input, Tensor &hidden) final;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(RunLayerContext &context) final;

  /**
   * @copydoc bool supportBackwarding() const
   */
  bool supportBackwarding() const final { return support_backwarding; };

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const final {}

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) final;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const final { return MeanLayer::type; };

  std::tuple<props::Print, props::Axis> mean_props;
  bool support_backwarding;
  int axis;

  static constexpr const char *type = "mean";
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __MEAN_LAYER_H__ */

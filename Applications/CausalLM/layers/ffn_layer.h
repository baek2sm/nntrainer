// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   fc_layer.h
 * @date   14 May 2020
 * @brief  This is Fully Connected Layer Class of Neural Network
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __FFN_LAYER_H__
#define __FFN_LAYER_H__
#ifdef __cplusplus

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <causallm_common_properties.h>
#include <common_properties.h>
#include <layer_impl.h>

namespace causallm {

/**
 * @class   FullyConnecedLayer
 * @brief   fully connected layer
 */
WIN_EXPORT class FFNLayer : public nntrainer::LayerImpl {
public:
  /**
   * @brief     Constructor of Fully Connected Layer
   */
  WIN_EXPORT FFNLayer();

  /**
   * @brief     Destructor of Fully Connected Layer
   */
  WIN_EXPORT ~FFNLayer() = default;

  /**
   *  @brief  Move constructor.
   *  @param[in] FullyConnected &&
   */
  WIN_EXPORT FFNLayer(FFNLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs FFNLayer to be moved.
   */
  WIN_EXPORT FFNLayer &operator=(FFNLayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override;

  /**
￼   * @copydoc Layer::incremental_forwarding(RunLayerContext &context, unsigned
￼   * int from, unsigned int to, bool training)
￼   */
  WIN_EXPORT void incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc Layer::calcGradient(RunLayerContext &context)
   * @note
   * [note for LoRA] implicit calcDerivative is implicitly applied.
   * The weight is already updated with the LoRA's (W = W + W_lora)
   */
  WIN_EXPORT void calcGradient(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::getType()
   */
  WIN_EXPORT const std::string getType() const override {
    return FFNLayer::type;
  };

  /**
   * @copydoc Layer::supportBackwarding()
   */
  WIN_EXPORT bool supportBackwarding() const override { return true; }

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::setBatch(RunLayerContext &context, unsigned int batch)
   */
  WIN_EXPORT void setBatch(nntrainer::RunLayerContext &context,
                           unsigned int batch) override;

  WIN_EXPORT void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  /**
   * @copydoc Layer::read()
   */
  WIN_EXPORT void read(std::ifstream &file, nntrainer::RunLayerContext &context,
                       bool opt_var, ml::train::ExecutionMode mode,
                       bool trainable,
                       nntrainer::TensorDim::DataType definedWeightDataType,
                       bool fsu = false, size_t start_offset = 0,
                       bool read_from_offset = false, int file_id = -1);

  /**
   * @copydoc Layer::read()
   */
  WIN_EXPORT void read(nntrainer::ReadSource src,
                       nntrainer::RunLayerContext &context, bool opt_var,
                       ml::train::ExecutionMode mode, bool trainable,
                       nntrainer::TensorDim::DataType definedWeightDataType,
                       bool fsu = false, size_t start_offset = 0,
                       bool read_from_offset = false) override;

  inline static const std::string type = "custom_ffn";

private:
  float lora_scaling;
  std::tuple<nntrainer::props::Unit, nntrainer::props::LoraRank,
             nntrainer::props::LoraAlpha,
             std::vector<nntrainer::props::TensorDataType>,
             causallm::props::LoraEnable, props::SkipPrefill>
    fc_props;                             /**< fc layer properties :
                                                unit - number of output neurons,
                                                lora_rank - rank of lora (optional)
                                                lora_scaling - scaling factor of LoRA apply, i.e.,
                                             lora_scaling = alpha / lora_rank */
  std::array<unsigned int, 2> weight_idx; /**< indices of the weights */
  std::array<unsigned int, 2> lora_idx;   /**< indices of the lora weights */
  bool skip_prefill = false;
};

} // namespace causallm

#endif /* __cplusplus */
#endif /* __FC_LAYER_H__ */

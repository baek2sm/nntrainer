// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   custom_tie_word_embedding_layer.h
 * @date   21 May 2025
 * @brief  This is Tie_Word_Embedding Layer Class of Neural Network
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __CUSTOM_TIE_WORD_EMBEDDING_LAYER_H__
#define __CUSTOM_TIE_WORD_EMBEDDING_LAYER_H__
#ifdef __cplusplus

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <common_properties.h>
#include <custom_common_properties.h>
#include <layer_devel.h>
#include <layer_impl.h>

namespace custom {

/**
 * @class   CustomTieWordEmbeddingLayer
 * @brief   CustomTieWordEmbeddingLayer
 * @todo    Support setBatch for CustomTieWordEmbeddingLayer
 */
WIN_EXPORT class CustomTieWordEmbeddingLayer : public nntrainer::LayerImpl {
public:
  /**
   * @brief     Constructor of Embedding Layer
   */
  WIN_EXPORT CustomTieWordEmbeddingLayer();

  /**
   * @brief     Destructor of Embedding Layer
   */
  WIN_EXPORT ~CustomTieWordEmbeddingLayer() = default;

  /**
   *  @brief  Move constructor.
   *  @param[in] CustomTieWordEmbeddingLayer &&
   */
  WIN_EXPORT CustomTieWordEmbeddingLayer(
    CustomTieWordEmbeddingLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs CustomTieWordEmbeddingLayer to be moved.
   */
  WIN_EXPORT CustomTieWordEmbeddingLayer &
  operator=(CustomTieWordEmbeddingLayer &&rhs) = default;

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
    return CustomTieWordEmbeddingLayer::type;
  };

  /**
   * @copydoc Layer::supportBackwarding()
   */
  WIN_EXPORT bool supportBackwarding() const override { return false; }

  WIN_EXPORT void updateTensorsByInputDimensions(nntrainer::RunLayerContext &context,
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


  /**
   * @copydic Layer::save()
   */
  WIN_EXPORT void
  save(std::ofstream &file, nntrainer::RunLayerContext &run_context,
       bool opt_var, ml::train::ExecutionMode mode, bool trainable,
       nntrainer::TensorDim::DataType definedWeightDataType) const;

  using Layer::setProperty;

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "custom_tie_word_embedding";

private:
  std::tuple<nntrainer::props::InDim, nntrainer::props::OutDim,
             props::SmartReply, nntrainer::props::Unit,
             props::UseVocabSelection, props::LshChoices>
    custom_embedding_props;
  enum mode { embedding, lm_head };
  enum mode mode_;
  std::array<unsigned int, 4> weight_idx; /**< indices of the weights */

  WIN_EXPORT void finalize_embedding(nntrainer::InitLayerContext &context);
  WIN_EXPORT void finalize_lmhead(nntrainer::InitLayerContext &context);
  WIN_EXPORT void
  incremental_forwarding_embedding(nntrainer::RunLayerContext &context,
                                   unsigned int from, unsigned int to,
                                   bool training);
  WIN_EXPORT void
  incremental_forwarding_lmhead(nntrainer::RunLayerContext &context,
                                unsigned int from, unsigned int to,
                                bool training);
};
} // namespace custom

#endif /* __cplusplus */
#endif /* __CUSTOM_TIE_WORD_EMBEDDING_LAYER_H__ */

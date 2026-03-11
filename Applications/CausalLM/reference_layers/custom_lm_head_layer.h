// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Hyeonseok Lee <hs89.lee@samsung.com>
 *
 * @file   custom_lm_head_layer.h
 * @date   13 Feb 2024
 * @brief  Implementation of custom lm head layer
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __CUSTOM_LM_HEAD_LAYER_H__
#define __CUSTOM_LM_HEAD_LAYER_H__

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <custom_common_properties.h>
#ifdef ENABLE_FP16
#include <custom_vocab_selection.h>
#endif
#include <layer_context.h>
#include <layer_impl.h>
#include <node_exporter.h>
#include <utility>

namespace custom {

/**
 * @brief A Custom LM Head layer for llama.
 *
 */
WIN_EXPORT class CustomLMHeadLayer : public nntrainer::LayerImpl {
public:
  /**
   * @brief Construct a new Custom LM Head layer object
   *
   */
  WIN_EXPORT CustomLMHeadLayer();

  /**
   * @brief Destroy the Custom LM Head layer object
   *
   */
  WIN_EXPORT ~CustomLMHeadLayer() {}

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;

  WIN_EXPORT void initialize(nntrainer::RunLayerContext &context) override {
#ifdef ENABLE_FP16
    auto use_vocab_selection =
      std::get<props::UseVocabSelection>(custom_lm_head_props).get();

    if (use_vocab_selection) {
      auto lsh_choices =
        std::get<props::LshChoices>(custom_lm_head_props).get();
      initVocabSelection(LshType::ORTHOSIMHASH, lsh_choices, context);
    }
#endif
  }

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override;

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
  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override{};

  /**
   * @copydoc Layer::getType()
   */
  WIN_EXPORT const std::string getType() const override {
    return CustomLMHeadLayer::type;
  };

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "custom_lm_head";

#ifdef ENABLE_FP16
  void initVocabSelection(LshType lshType, int lshChoices,
                          nntrainer::RunLayerContext &context);

  std::shared_ptr<VocabSelection> vocabSelection;
#endif

private:
  std::tuple<nntrainer::props::Unit, props::UseVocabSelection,
             props::LshChoices, props::SmartReply>
    custom_lm_head_props;
  std::array<unsigned int, 4> weight_idx; /**< indices of the weights */
  std::unique_ptr<nntrainer::Tensor>
    weight_T; // temporary weight. will be removed
};

} // namespace custom

#endif /* __CUSTOM_LM_HEAD_LAYER_H__ */

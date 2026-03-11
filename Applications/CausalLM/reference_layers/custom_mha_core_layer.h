// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   custom_mha_core_layer.h
 * @date   02 September 2024
 * @see    https://github.com/nnstreamer/nntrainer
 *         https://arxiv.org/abs/1706.03762
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is custom_mha_core layer supports
 *         the work of multi_head_attention.
 * @note   Unlike custom_multi_head_attention_layer,
 *         which works all of the attention operations
 *         in a layer, this layer is attached after Q / K / V
 *         fully connected layer to post-process them
 *         including KV-Cache.
 *         For inference, incremental_forwarding is called,
 *         which takes inputs of seq_len = 1 via `from` / `to` param.
 *         For training, forwarding is called,
 *         which takes all input seqences at once.
 *
 */

#ifndef __CUSTOM_MHA_CORE_LAYER_H__
#define __CUSTOM_MHA_CORE_LAYER_H__

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <complex>

#include <acti_func.h>
#include <custom_common_properties.h>
#include <layer_impl.h>
#include <util_simd.h>

#include <utility>

namespace custom {

/**
 * @class MHA Core Layer
 * @brief Part of Multi-Head-Attention Layer.
 *        It should be attached after Q / K / V fc layers and before O fc layer.
 *        custom_mha_core_layer computes attention, while updating KV-cache for
 *        inference mode.
 *
 *    [ Q ]    [ K ]    [ V ]
 *      |        |        |
 *   [ custom_mha_core_layer ]   <=>  [ custom_multi_head_attention_layer ]
 *               |
 *             [ O ]
 *
 */
WIN_EXPORT class CustomMHACoreLayer : public nntrainer::LayerImpl {
public:
  /**
   * @brief Constructor of MhaCore Layer
   */
  WIN_EXPORT CustomMHACoreLayer();

  /**
   * @brief Destructor of MhaPost Layer
   */
  WIN_EXPORT ~CustomMHACoreLayer();

  /**
   *  @brief  Move constructor of CustomMultiHeadAttentionLayer.
   *  @param[in] CustomMultiHeadAttentionLayer &&
   */
  WIN_EXPORT CustomMHACoreLayer(CustomMHACoreLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs CustomMultiHeadAttentionLayer to be moved.
   */
  CustomMHACoreLayer &operator=(CustomMHACoreLayer &&rhs) = default;

  /**
   * @brief Finalize funciton of MhaCore Layer
   */
  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;

  /**
   * @brief forwarding function of MhaCore Layer
   *        Please note that forwarding function is used only for training.
   */
  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::incremental_forwarding
   */
  WIN_EXPORT void one_batch_incremental_forwarding(
    const unsigned int batch, const unsigned int _from, const unsigned int from,
    const unsigned int to, 
    nntrainer::Tensor &query, nntrainer::Tensor &key,
    nntrainer::Tensor &value, nntrainer::Tensor &output,
    nntrainer::Tensor &cache_key, nntrainer::Tensor &cache_value, 
    nntrainer::Tensor &projected_key, nntrainer::Tensor &projected_value, 
    nntrainer::Tensor &attention_weight, 
    ml::train::TensorDim &query_dim,
    ml::train::TensorDim &query_step_dim,
    ml::train::TensorDim &key_dim,
    ml::train::TensorDim &value_dim,
    ml::train::TensorDim &cache_key_dim,
    ml::train::TensorDim &cache_key_step_dim,
    ml::train::TensorDim &cache_value_dim,
    ml::train::TensorDim &cache_value_step_dim,
    ml::train::TensorDim &attention_weight_dim,
    ml::train::TensorDim &attention_weight_step_dim,
    ml::train::TensorDim &output_dim,
    ml::train::TensorDim &output_step_dim);

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
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
  WIN_EXPORT void calcGradient(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc bool supportBackwarding() const
   * @note In current version, we do not support backwarding yet.
   * It will be updated ASAP.
   */
  WIN_EXPORT bool supportBackwarding() const override { return true; };

  /**
   * @copydoc Layer::setBatch(RunLayerContext &context, unsigned int batch)
   */
  WIN_EXPORT void exportTo(nntrainer::Exporter &exporter, const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::getType()
   */
  WIN_EXPORT const std::string getType() const override {
    return CustomMHACoreLayer::type;
  };

  /**
   * @copydoc Layer::setBatch(RunLayerContext &context, unsigned int batch)
   */
  WIN_EXPORT void setBatch(nntrainer::RunLayerContext &context,
                unsigned int batch) override;

  WIN_EXPORT void updateTensorsByInputDimensions(nntrainer::RunLayerContext &context,
                std::vector<nntrainer::TensorDim> input_dimensions) override;

  inline static const std::string type = "custom_mha_core";

private:
  std::tuple<nntrainer::props::NumHeads, props::NumHeads_KV,
             nntrainer::props::ProjectedKeyDim,
             nntrainer::props::ProjectedValueDim, nntrainer::props::OutputShape,
             nntrainer::props::DropOutRate,
             nntrainer::props::ReturnAttentionWeight,
             nntrainer::props::AverageAttentionWeight,
             nntrainer::props::MaxTimestep, props::SmartReply,
             props::LocalWindowSize, props::MaxNewTokens>
    mha_core_props; /**< mha_core layer properties */

  /** softmax activation operation */
  nntrainer::ActiFunc sm;

  float epsilon;            /** to avoid overflow */
  unsigned int cache_index; /** idx of kv cache */

  /** intermal info */
  unsigned int num_heads_Q;
  unsigned int num_heads_KV;
  unsigned int head_dim;
  bool cache_shift;

enum INOUT_INDEX {
  /** input index */
  QUERY = 0,
  KEY = 1,    
  VALUE = 2,
  MASK = 3,

  /** output index */
  OUTPUT = 0,
  RETURN_ATTENTION_WEIGHT = 1,
};


  /**< indices of the weights and tensors */
  enum AttentionParams {
    cache_key,
    cache_value,
    projected_key,
    projected_value,
    /** intended comment for later use of attention_mask */
    // attention_mask,
    attention_weight,
    dropout_mask,
    attention_output,
  };
  std::array<unsigned int, 7> weight_idx;


  /****************** ROTARY EMBEDDING *****************/
  /** static variable - they are all expected to be initialized once */
  inline static std::vector<std::vector<float>> *freqs_cos = {};
  inline static std::vector<std::vector<float>> *freqs_sin = {};
  inline static std::vector<float> thetas;

  /**
   * @brief pre_compute frequencies for Rotary Embedding.
   * @note it is expected to be called only once at the finalize.
   * @param[in] head_dim dimension of head
   * @param[in] seq_len sequence length
   * @param[in] theta base of theta (default = 10000)
   */
  void precompute_freqs(int head_dim, unsigned int seq_len,
                        float theta = 10000.0);

  /**
   * @brief     apply rotary embedding
   * @param[in] in input tensor
   * @param[in] dim hidden dim size
   * @param[in] from sequence order
   */
  void apply_rotary_emb_tensor(nntrainer::Tensor &in, unsigned int dim,
                               unsigned int from);
  /************** END OF  ROTARY EMBEDDING *************/

  /**
   * @brief calculate common derivative
   * @param context Context of the layer
   */
  void calcCommonDerivative(nntrainer::RunLayerContext &context);

}; // end of class CustomMHACoreLayer
} // end of namespace custom

#endif

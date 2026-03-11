// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungBaek Hong
 *
 * @file   cross_mha_core.h
 * @date   11 March 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @brief  Simplified cross-attention core with static KV cache.
 */

#ifndef __CROSS_MHA_CORE_H__
#define __CROSS_MHA_CORE_H__

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <array>
#include <tuple>

#include <common_properties.h>
#include <cpu_backend.h>
#include <layer_impl.h>

namespace causallm {

namespace props {

/**
 * @brief Number of KV heads for grouped-query attention.
 */
class NumHeads_KV : public nntrainer::PositiveIntegerProperty {
public:
  NumHeads_KV(unsigned int value = 1) { set(value); };
  static constexpr const char *key = "num_heads_KV";
  using prop_tag = nntrainer::uint_prop_tag;
};

/**
 * @brief Number of new tokens used to refresh max_timestep in dynamic shape.
 */
class MaxNewTokens : public nntrainer::Property<unsigned int> {
public:
  MaxNewTokens(unsigned int value = 1) { set(value); };
  static constexpr const char *key = "max_new_tokens";
  using prop_tag = nntrainer::uint_prop_tag;
};

}; // namespace props

/**
 * @brief Cross-attention core with one-time KV cache initialization.
 *
 * First incremental step (`from == 0`) stores key/value into static cache.
 * Following steps only use query with the cached key/value tensors.
 */
WIN_EXPORT class CrossMHACoreLayer : public nntrainer::LayerImpl {
public:
  WIN_EXPORT CrossMHACoreLayer();
  WIN_EXPORT ~CrossMHACoreLayer();

  WIN_EXPORT CrossMHACoreLayer(CrossMHACoreLayer &&rhs) noexcept = default;
  WIN_EXPORT CrossMHACoreLayer &
  operator=(CrossMHACoreLayer &&rhs) noexcept = default;

  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;
  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override;
  WIN_EXPORT void incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) override;
  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override;
  WIN_EXPORT void calcGradient(nntrainer::RunLayerContext &context) override;
  WIN_EXPORT bool supportBackwarding() const override { return true; };
  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override;
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;
  WIN_EXPORT const std::string getType() const override {
    return CrossMHACoreLayer::type;
  };
  WIN_EXPORT void setBatch(nntrainer::RunLayerContext &context,
                           unsigned int batch) override;
  WIN_EXPORT void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  inline static const std::string type = "cross_mha_core";

private:
  std::tuple<
    nntrainer::props::NumHeads, props::NumHeads_KV,
    nntrainer::props::ProjectedKeyDim, nntrainer::props::ProjectedValueDim,
    nntrainer::props::OutputShape, nntrainer::props::DropOutRate,
    nntrainer::props::ReturnAttentionWeight,
    nntrainer::props::AverageAttentionWeight, nntrainer::props::MaxTimestep,
    props::MaxNewTokens>
    mha_core_props;

  size_t num_heads_Q;
  size_t num_heads_KV;
  size_t head_dim;
  bool cache_initialized;
  unsigned int cached_seq_len;

  enum INOUT_INDEX {
    QUERY = 0,
    KEY = 1,
    VALUE = 2,
    MASK = 3,
    OUTPUT = 0,
    RETURN_ATTENTION_WEIGHT = 1,
  };

  enum AttentionParams {
    cache_key,
    cache_value,
  };

  std::array<unsigned int, 2> tensor_idx;

  void one_batch_incremental_forwarding(
    const unsigned int batch, nntrainer::Tensor &query_step,
    nntrainer::Tensor &attention_output_step, nntrainer::Tensor &cache_key,
    nntrainer::Tensor &cache_value,
    const ml::train::TensorDim &cache_key_dim,
    const ml::train::TensorDim &cache_value_dim, unsigned int context_len);

  void compute_kcaches(nntrainer::Tensor &in, nntrainer::Tensor &cache,
                       nntrainer::Tensor &out, unsigned int context_len,
                       unsigned int sequence_len, unsigned int num_head,
                       unsigned int group_size, unsigned int head_dim);

  void softmax_rows(nntrainer::Tensor &qk_out, unsigned int row,
                    unsigned int context_len, unsigned int num_head);

  void compute_vcaches(nntrainer::Tensor &in, nntrainer::Tensor &vcache,
                       nntrainer::Tensor &output, unsigned int context_len,
                       unsigned int sequence_len, int num_cache_head,
                       int gqa_size, int head_dim);
};
} // namespace causallm

#endif

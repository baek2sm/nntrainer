// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2022 hyeonseok Lee <hs89.lee@samsung.com>
 *
 * @file   multi_head_attention_layer.h
 * @date   08 July 2022
 * @see    https://github.com/nnstreamer/nntrainer
 *         https://arxiv.org/abs/1706.03762
 * @author hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is MultiHeadAttention Layer Class for Neural Network
 *
 */

#ifndef __MULTI_HEAD_ATTENTION_LAYER_H__
#define __MULTI_HEAD_ATTENTION_LAYER_H__

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <complex>

#include <acti_func.h>
#include <complex>
#include <custom_common_properties.h>
#include <layer_impl.h>
#include <util_simd.h>

#include <utility>

namespace custom {

/**
 * @class   Multi Head Attention Layer
 * @brief   Implementation of multi head attention which is described in paper
 * "Attention is all you need"
 */
WIN_EXPORT class CustomMultiHeadAttentionLayer : public nntrainer::LayerImpl {
public:
  /**
   * @brief     Constructor of MultiHeadAttention Layer
   */
  WIN_EXPORT CustomMultiHeadAttentionLayer();

  /**
   * @brief     Destructor of MultiHeadAttention Layer
   */
  WIN_EXPORT ~CustomMultiHeadAttentionLayer();

  /**
   *  @brief  Move constructor of CustomMultiHeadAttentionLayer.
   *  @param[in] CustomMultiHeadAttentionLayer &&
   */
  WIN_EXPORT CustomMultiHeadAttentionLayer(CustomMultiHeadAttentionLayer &&rhs) noexcept =
    default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs CustomMultiHeadAttentionLayer to be moved.
   */
  CustomMultiHeadAttentionLayer &
  operator=(CustomMultiHeadAttentionLayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context, bool training) override;

  WIN_EXPORT void one_batch_incremental_forwarding(const unsigned int batch, const unsigned int _from, const unsigned int from, const unsigned int to, const unsigned int num_heads, const unsigned int projected_query_dim_prop, const unsigned int projected_key_dim_prop, const unsigned int projected_value_dim_prop, const bool disable_bias, const bool cache_shift, nntrainer::Tensor &query, nntrainer::Tensor &key, nntrainer::Tensor &value, nntrainer::Tensor &output, nntrainer::Tensor &query_fc_weight, nntrainer::Tensor &query_fc_bias, nntrainer::Tensor &key_fc_weight, nntrainer::Tensor &key_fc_bias, nntrainer::Tensor &value_fc_weight, nntrainer::Tensor &value_fc_bias, nntrainer::Tensor &fc_weight, nntrainer::Tensor &fc_bias, nntrainer::Tensor &projected_query, nntrainer::Tensor &projected_key, nntrainer::Tensor &projected_value, nntrainer::Tensor &cache_key, nntrainer::Tensor &cache_value, nntrainer::Tensor &attention_weight, nntrainer::Tensor &attention_output, ml::train::TensorDim &query_dim, ml::train::TensorDim &query_step_dim, ml::train::TensorDim &key_dim, ml::train::TensorDim &key_step_dim, ml::train::TensorDim &value_dim, ml::train::TensorDim &value_step_dim, ml::train::TensorDim &output_dim, ml::train::TensorDim &output_step_dim, ml::train::TensorDim &projected_query_dim, ml::train::TensorDim &projected_query_step_dim, ml::train::TensorDim &projected_key_dim, ml::train::TensorDim &projected_key_step_dim, ml::train::TensorDim &cache_key_dim, ml::train::TensorDim &cache_key_step_dim, ml::train::TensorDim &cached_key_dim, ml::train::TensorDim &projected_value_dim, ml::train::TensorDim &projected_value_step_dim, ml::train::TensorDim &cache_value_dim, ml::train::TensorDim &cache_value_step_dim, ml::train::TensorDim &cached_value_dim, ml::train::TensorDim &attention_weight_dim, ml::train::TensorDim &attention_weight_step_dim, ml::train::TensorDim &attention_output_dim, ml::train::TensorDim &attention_output_step_dim);

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
  WIN_EXPORT void calcGradient(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc bool supportBackwarding() const
   */
  WIN_EXPORT bool supportBackwarding() const override { return true; };

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  WIN_EXPORT void exportTo(nntrainer::Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::getType()
   */
  WIN_EXPORT const std::string getType() const override {
    return CustomMultiHeadAttentionLayer::type;
  };

  /**
   * @copydoc Layer::setBatch(RunLayerContext &context, unsigned int batch)
   */
  WIN_EXPORT void setBatch(nntrainer::RunLayerContext &context,
                unsigned int batch) override;

  inline static const std::string type = "custom_multi_head_attention";

private:
  std::tuple<
    nntrainer::props::NumHeads, nntrainer::props::ProjectedKeyDim,
    nntrainer::props::ProjectedValueDim, nntrainer::props::OutputShape,
    nntrainer::props::DropOutRate, nntrainer::props::ReturnAttentionWeight,
    nntrainer::props::AverageAttentionWeight, nntrainer::props::MaxTimestep, props::SmartReply>
    multi_head_attention_props; /**< multi_head_attention layer properties */

  nntrainer::ActiFunc sm; /** softmax activation operation */
  std::array<unsigned int, 16>
    weight_idx; /**< indices of the weights and tensors */

  /**
   * @brief     to protect overflow
   */
  float epsilon;

  unsigned int cache_index;

  inline static std::vector<std::vector<float>> *freqs_cos = {};
  inline static std::vector<std::vector<float>> *freqs_sin = {};
  inline static std::vector<float> freqs;

  /**
   * @brief     compute frequency for rotary embedding
   * @param[in] dim hidden dim size
   * @param[in] seq_len sequency length
   * @param[in] theta rotary angle
   */
  void precompute_freqs(int dim, unsigned int seq_len, float theta = 10000.0) {
    if (freqs_cos == nullptr) {
      unsigned int half_ = dim / 2;
      for (unsigned int i = 0; i < half_; ++i) {
        freqs.push_back(1.0 /
                        (std::pow(theta, (2 * i) / static_cast<float>(dim))));
      }

      auto cos = new std::vector<std::vector<float>>();
      cos->assign(seq_len, std::vector<float>(dim, 0));

      auto sin = new std::vector<std::vector<float>>();
      sin->assign(seq_len, std::vector<float>(dim, 0));

      for (unsigned int i = 0; i < seq_len; ++i) {
#ifdef USE_NEON
        nntrainer::calc_trigonometric_vals_dup(half_, freqs.data(), (*cos)[i].data(),
                                    (*sin)[i].data(), i);
#else
        for (unsigned int j = 0; j < half_; ++j) {
          float angle = i * freqs[j];
          (*cos)[i][j] = std::cos(angle);
          (*cos)[i][j + half_] = std::cos(angle); // repeated 2 times

          (*sin)[i][j] = std::sin(angle);
          (*sin)[i][j + half_] = std::sin(angle); // repeated 2 times
        }
#endif
      }

      freqs_cos = cos;
      freqs_sin = sin;
    }
  }

  /**
   * @brief     apply rotary embedding
   * @param[in] in input tensor
   * @param[in] dim hidden dim size
   * @param[in] from sequence order
   */
  void apply_rotary_emb_tensor(nntrainer::Tensor &in, unsigned int dim,
                               unsigned int from) {
    nntrainer::Tensor out(in.getDim());
    float value = 0;
    float transformed_value = 0.0;
    unsigned int half_ = dim / 2;
    unsigned int max_timestep =
      std::get<nntrainer::props::MaxTimestep>(multi_head_attention_props).get();

    std::vector<float> *cos_ = nullptr;
    std::vector<float> *sin_ = nullptr;

    if (from >= max_timestep) {
      cos_ = new std::vector<float>(dim);
      sin_ = new std::vector<float>(dim);
#ifdef USE_NEON
      nntrainer::calc_trigonometric_vals_dup(half_, freqs.data(), cos_->data(),
                                  sin_->data(), from);
#else
      for (unsigned int i = 0; i < half_; ++i) {
        float angle = from * freqs[i];
        (*cos_)[i] = std::cos(angle);
        (*cos_)[i + half_] = std::cos(angle); // repeated 2 times

        (*sin_)[i] = std::sin(angle);
        (*sin_)[i + half_] = std::sin(angle); // repeated 2 times
      }
#endif
    }

    if (in.getDataType() == ml::train::TensorDim::DataType::FP32) {
      for (unsigned int b = 0; b < in.batch(); b++) {
        for (unsigned int c = 0; c < in.channel(); c++) {
          for (unsigned int h = 0; h < in.height(); h++) {
            if (from < max_timestep) {
              cos_ = &(*freqs_cos)[from + h];
              sin_ = &(*freqs_sin)[from + h];
            }

            for (unsigned int w = 0; w < in.width(); w = w + dim) {
              for (unsigned int k = 0; k < dim; k++) {
                unsigned int span = w + k;
                value = in.getValue<float>(b, c, h, span);

                if (k < half_) {
                  transformed_value =
                    -1.0 * in.getValue<float>(b, c, h, span + half_);
                } else {
                  transformed_value = in.getValue<float>(b, c, h, span - half_);
                }
                value = value * (*cos_)[k] + transformed_value * (*sin_)[k];
                out.setValue(b, c, h, span, value);
              }
            }
          }
        }
      }
    } else if (in.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
      for (unsigned int b = 0; b < in.batch(); b++) {
        for (unsigned int c = 0; c < in.channel(); c++) {
          for (unsigned int h = 0; h < in.height(); h++) {
            if (from < max_timestep) {
              cos_ = &(*freqs_cos)[from + h];
              sin_ = &(*freqs_sin)[from + h];
            }
            for (unsigned int w = 0; w < in.width(); w = w + dim) {
#ifdef USE_NEON
              nntrainer::compute_rotary_embedding_value(
                dim, half_, w, in.getData<_FP16>() + in.getIndex(b, c, h, 0),
                out.getData<_FP16>() + out.getIndex(b, c, h, 0), cos_->data(),
                sin_->data());
#else
              for (unsigned int k = 0; k < dim; k++) {
                unsigned int span = w + k;
                value = static_cast<float>(in.getValue<_FP16>(b, c, h, span));

                if (k < half_) {
                  transformed_value =
                    -1.0 * static_cast<float>(
                             in.getValue<_FP16>(b, c, h, half_ + span));
                } else {
                  transformed_value = static_cast<float>(
                    in.getValue<_FP16>(b, c, h, span - half_));
                }
                out.setValue(
                  b, c, h, span,
                  static_cast<_FP16>(value * (*cos_)[k] +
                                     transformed_value * (*sin_)[k]));
              }
#endif
            }
          }
        }
      }
#else
      throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
    }

    if (from >= max_timestep) {
      delete cos_;
      delete sin_;
    }
    in.copy(out);
  }

  /**
   * @brief calculate common derivative
   * @param context Context of the layer
   */
  void calcCommonDerivative(nntrainer::RunLayerContext &context);
};

} // namespace custom

#endif /* __MULTI_HEAD_ATTENTION_LAYER_H__ */

// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungBaek Hong
 *
 * @file   cross_mha_core.cpp
 * @date   11 March 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @brief  Simplified cross-attention core with static KV cache.
 */

#include <algorithm>
#include <cmath>
#include <limits>

#include <cross_mha_core.h>
#include <layer_context.h>
#include <nntrainer_error.h>
#include <node_exporter.h>

namespace causallm {

#define tile_size 4

CrossMHACoreLayer::CrossMHACoreLayer() :
  mha_core_props(
    nntrainer::props::NumHeads(), props::NumHeads_KV(),
    nntrainer::props::ProjectedKeyDim(), nntrainer::props::ProjectedValueDim(),
    nntrainer::props::OutputShape(), nntrainer::props::DropOutRate(),
    nntrainer::props::ReturnAttentionWeight(),
    nntrainer::props::AverageAttentionWeight(), nntrainer::props::MaxTimestep(),
    props::MaxNewTokens()),
  num_heads_Q(0),
  num_heads_KV(0),
  head_dim(0),
  cache_initialized(false),
  cached_seq_len(0) {
  tensor_idx.fill(std::numeric_limits<unsigned>::max());
}

CrossMHACoreLayer::~CrossMHACoreLayer() {}

void CrossMHACoreLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() < 3 || context.getNumInputs() > 4,
                std::invalid_argument)
    << "cross_mha_core layer needs 3 or 4 inputs. (query, key, value and mask "
       "is optional)";

  const std::vector<ml::train::TensorDim> &input_dims =
    context.getInputDimensions();
  const ml::train::TensorDim &query_dim = input_dims[INOUT_INDEX::QUERY];
  const ml::train::TensorDim &key_dim = input_dims[INOUT_INDEX::KEY];
  const ml::train::TensorDim &value_dim = input_dims[INOUT_INDEX::VALUE];

  NNTR_THROW_IF(key_dim.height() != value_dim.height(), std::invalid_argument)
    << "cross attention requires key/value sequence lengths to match";
  NNTR_THROW_IF(key_dim.width() != value_dim.width(), std::invalid_argument)
    << "cross attention requires key/value hidden widths to match";

  const unsigned int batch_size = query_dim.batch();
  const unsigned int key_height = key_dim.height();
  const unsigned int query_width = query_dim.width();
  const unsigned int key_width = key_dim.width();

  num_heads_Q = static_cast<size_t>(
    std::get<nntrainer::props::NumHeads>(mha_core_props).get());
  num_heads_KV =
    std::get<props::NumHeads_KV>(mha_core_props).empty()
      ? num_heads_Q
      : static_cast<size_t>(std::get<props::NumHeads_KV>(mha_core_props).get());

  NNTR_THROW_IF(num_heads_Q == 0 || num_heads_KV == 0, std::invalid_argument)
    << "num_heads and num_heads_KV must be greater than zero";
  NNTR_THROW_IF(query_width % num_heads_Q != 0, std::invalid_argument)
    << "query width must be divisible by num_heads";
  NNTR_THROW_IF(key_width % num_heads_KV != 0, std::invalid_argument)
    << "key width must be divisible by num_heads_KV";

  head_dim = static_cast<size_t>(query_width) / num_heads_Q;
  NNTR_THROW_IF(head_dim != key_width / num_heads_KV, std::invalid_argument)
    << "num_heads and num_heads_KV are not properly given";

#ifdef ENABLE_FP16
  ml::train::TensorDim cache_key_dim(
    {batch_size, 1, key_height, num_heads_KV * head_dim},
    {context.getFormat(), ml::train::TensorDim::DataType::FP16});
  ml::train::TensorDim cache_value_dim(
    {batch_size, 1, key_height, num_heads_KV * head_dim},
    {context.getFormat(), ml::train::TensorDim::DataType::FP16});
#else
  ml::train::TensorDim cache_key_dim(
    {batch_size, 1, key_height, num_heads_KV * head_dim},
    {context.getFormat(), ml::train::TensorDim::DataType::UINT16});
  ml::train::TensorDim cache_value_dim(
    {batch_size, 1, key_height, num_heads_KV * head_dim},
    {context.getFormat(), ml::train::TensorDim::DataType::UINT16});
#endif

  tensor_idx[AttentionParams::cache_key] = context.requestTensor(
    cache_key_dim, "cache_key", nntrainer::Initializer::NONE, false,
    nntrainer::TensorLifespan::MAX_LIFESPAN);
  tensor_idx[AttentionParams::cache_value] = context.requestTensor(
    cache_value_dim, "cache_value", nntrainer::Initializer::NONE, false,
    nntrainer::TensorLifespan::MAX_LIFESPAN);

  std::vector<nntrainer::TensorDim> output_dims(1);
  output_dims[0] = query_dim;
  output_dims[0].width(head_dim * num_heads_Q);
  output_dims[0].setTensorType(
    {context.getFormat(), context.getActivationDataType()});
  context.setOutputDimensions(output_dims);

  cache_initialized = false;
  cached_seq_len = 0;
}

void CrossMHACoreLayer::forwarding(nntrainer::RunLayerContext &context,
                                   bool training) {
  (void)context;
  (void)training;
}

void CrossMHACoreLayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                               unsigned int from,
                                               unsigned int to, bool training) {
  (void)to;
  (void)training;

  nntrainer::Tensor &query = context.getInput(INOUT_INDEX::QUERY);
  nntrainer::Tensor &key = context.getInput(INOUT_INDEX::KEY);
  nntrainer::Tensor &value = context.getInput(INOUT_INDEX::VALUE);
  nntrainer::Tensor &output = context.getOutput(INOUT_INDEX::OUTPUT);
  nntrainer::Tensor &cache_key =
    context.getTensor(tensor_idx[AttentionParams::cache_key]);
  nntrainer::Tensor &cache_value =
    context.getTensor(tensor_idx[AttentionParams::cache_value]);

  ml::train::TensorDim query_dim = query.getDim();
  ml::train::TensorDim key_dim = key.getDim();
  ml::train::TensorDim value_dim = value.getDim();
  ml::train::TensorDim output_dim = output.getDim();
  ml::train::TensorDim cache_key_dim = cache_key.getDim();
  ml::train::TensorDim cache_value_dim = cache_value.getDim();

  NNTR_THROW_IF(query_dim.width() != num_heads_Q * head_dim, std::invalid_argument)
    << "query width does not match layer head configuration";
  NNTR_THROW_IF(query_dim.batch() > cache_key_dim.batch(), std::invalid_argument)
    << "query batch cannot exceed cache batch size";

  const bool refresh_cache = (from == 0) || !cache_initialized;
  if (refresh_cache) {
    NNTR_THROW_IF(key_dim.height() != value_dim.height(), std::invalid_argument)
      << "cross attention requires key/value sequence lengths to match";
    NNTR_THROW_IF(key_dim.width() != value_dim.width(), std::invalid_argument)
      << "cross attention requires key/value hidden widths to match";
    NNTR_THROW_IF(key_dim.width() != num_heads_KV * head_dim, std::invalid_argument)
      << "key/value width does not match layer head configuration";
    NNTR_THROW_IF(key_dim.height() > cache_key_dim.height(), std::invalid_argument)
      << "key/value sequence length exceeds cache capacity";
    NNTR_THROW_IF(key_dim.batch() > cache_key_dim.batch(), std::invalid_argument)
      << "key/value batch cannot exceed cache batch size";

    ml::train::TensorDim key_step_dim = key_dim;
    key_step_dim.batch(1);
    ml::train::TensorDim value_step_dim = value_dim;
    value_step_dim.batch(1);
    ml::train::TensorDim cache_key_step_dim = cache_key_dim;
    cache_key_step_dim.batch(1);
    cache_key_step_dim.height(key_dim.height());
    ml::train::TensorDim cache_value_step_dim = cache_value_dim;
    cache_value_step_dim.batch(1);
    cache_value_step_dim.height(value_dim.height());

    for (unsigned int batch = 0; batch < key_dim.batch(); ++batch) {
      nntrainer::Tensor key_step = key.getSharedDataTensor(
        key_step_dim, batch * key_dim.getFeatureLen(), true);
      nntrainer::Tensor value_step = value.getSharedDataTensor(
        value_step_dim, batch * value_dim.getFeatureLen(), true);
      nntrainer::Tensor cache_key_step = cache_key.getSharedDataTensor(
        cache_key_step_dim, batch * cache_key_dim.getFeatureLen(), true);
      nntrainer::Tensor cache_value_step = cache_value.getSharedDataTensor(
        cache_value_step_dim, batch * cache_value_dim.getFeatureLen(), true);

      cache_key_step.copyData(key_step);
      cache_value_step.copyData(value_step);
    }

    if (key_dim.batch() == 1 && cache_key_dim.batch() > 1) {
      nntrainer::Tensor cache_key_0 =
        cache_key.getSharedDataTensor(cache_key_step_dim, 0, true);
      nntrainer::Tensor cache_value_0 =
        cache_value.getSharedDataTensor(cache_value_step_dim, 0, true);
      for (unsigned int batch = 1; batch < cache_key_dim.batch(); ++batch) {
        nntrainer::Tensor cache_key_n = cache_key.getSharedDataTensor(
          cache_key_step_dim, batch * cache_key_dim.getFeatureLen(), true);
        nntrainer::Tensor cache_value_n = cache_value.getSharedDataTensor(
          cache_value_step_dim, batch * cache_value_dim.getFeatureLen(), true);
        cache_key_n.copyData(cache_key_0);
        cache_value_n.copyData(cache_value_0);
      }
    }

    cache_initialized = true;
    cached_seq_len = key_dim.height();
  }

  NNTR_THROW_IF(!cache_initialized || cached_seq_len == 0, std::invalid_argument)
    << "cross-attention cache is not initialized";

  ml::train::TensorDim query_step_dim = query_dim;
  query_step_dim.batch(1);
  ml::train::TensorDim output_step_dim = output_dim;
  output_step_dim.batch(1);

  for (unsigned int batch = 0; batch < query_dim.batch(); ++batch) {
    nntrainer::Tensor query_step = query.getSharedDataTensor(
      query_step_dim, batch * query_dim.getFeatureLen(), true);
    nntrainer::Tensor output_step = output.getSharedDataTensor(
      output_step_dim, batch * output_dim.getFeatureLen(), true);

    one_batch_incremental_forwarding(batch, query_step, output_step, cache_key,
                                     cache_value, cache_key_dim,
                                     cache_value_dim, cached_seq_len);
  }
}

void CrossMHACoreLayer::one_batch_incremental_forwarding(
  const unsigned int batch, nntrainer::Tensor &query_step,
  nntrainer::Tensor &attention_output_step, nntrainer::Tensor &cache_key,
  nntrainer::Tensor &cache_value, const ml::train::TensorDim &cache_key_dim,
  const ml::train::TensorDim &cache_value_dim, unsigned int context_len) {

  ml::train::TensorDim cached_key_dim = cache_key_dim;
  cached_key_dim.batch(1);
  cached_key_dim.height(context_len);
  ml::train::TensorDim cached_value_dim = cache_value_dim;
  cached_value_dim.batch(1);
  cached_value_dim.height(context_len);

  nntrainer::Tensor b_cached_key = cache_key.getSharedDataTensor(
    cached_key_dim, batch * cache_key_dim.getFeatureLen(), true);
  nntrainer::Tensor b_cached_value = cache_value.getSharedDataTensor(
    cached_value_dim, batch * cache_value_dim.getFeatureLen(), true);

  const unsigned int q_seq_len = query_step.height();
  const unsigned int gqa_size = num_heads_Q / num_heads_KV;

  nntrainer::Tensor attention_score(
    1, 1, q_seq_len * context_len, num_heads_Q, query_step.getTensorType());

  compute_kcaches(query_step, b_cached_key, attention_score, context_len,
                  q_seq_len, num_heads_Q, gqa_size, head_dim);
  softmax_rows(attention_score, q_seq_len, context_len, num_heads_Q);
  compute_vcaches(attention_score, b_cached_value, attention_output_step,
                  context_len, q_seq_len, num_heads_KV, gqa_size, head_dim);
}

void CrossMHACoreLayer::compute_kcaches(nntrainer::Tensor &in,
                                        nntrainer::Tensor &cache,
                                        nntrainer::Tensor &out,
                                        unsigned int context_len,
                                        unsigned int sequence_len,
                                        unsigned int num_head,
                                        unsigned int group_size,
                                        unsigned int dim_per_head) {
  if (in.getDataType() == ml::train::TensorDim::DataType::FP32) {
    const uint16_t *cache_data = cache.getData<uint16_t>();
    for (unsigned int i = 0; i < sequence_len; ++i) {
      const float *input_addr = in.getData<float>() + num_head * dim_per_head * i;
      float *output_addr = out.getData<float>() + i * context_len * num_head;
      nntrainer::compute_kcaches<uint16_t>(
        input_addr, cache_data, output_addr, context_len, num_head / group_size,
        dim_per_head, group_size, tile_size, context_len);
    }
  } else if (in.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    const _FP16 *cache_data = cache.getData<_FP16>();
    for (unsigned int i = 0; i < sequence_len; ++i) {
      const _FP16 *input_addr =
        in.getData<_FP16>() + num_head * dim_per_head * i;
      _FP16 *output_addr = out.getData<_FP16>() + i * context_len * num_head;
      nntrainer::compute_kcaches(input_addr, cache_data, output_addr,
                                 context_len, num_head / group_size,
                                 dim_per_head, group_size, tile_size,
                                 context_len);
    }
#else
    NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
#endif
  }
}

void CrossMHACoreLayer::softmax_rows(nntrainer::Tensor &qk_out, unsigned int row,
                                     unsigned int context_len,
                                     unsigned int num_head) {
  if (qk_out.getDataType() == ml::train::TensorDim::DataType::FP32) {
    float *qk_out_ = qk_out.getData<float>();
    for (unsigned int i = 0; i < row; ++i) {
      const size_t start_row = i * context_len;
      const size_t end_row = (i + 1) * context_len;
      nntrainer::softmax_row(qk_out_, start_row, end_row, num_head);
    }
  } else if (qk_out.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    _FP16 *qk_out_ = qk_out.getData<_FP16>();
    for (unsigned int i = 0; i < row; ++i) {
      const size_t start_row = i * context_len;
      const size_t end_row = (i + 1) * context_len;
      nntrainer::softmax_row_inplace(qk_out_, start_row, end_row, num_head);
    }
#else
    NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
#endif
  }
}

void CrossMHACoreLayer::compute_vcaches(nntrainer::Tensor &in,
                                        nntrainer::Tensor &vcache,
                                        nntrainer::Tensor &output,
                                        unsigned int context_len,
                                        unsigned int sequence_len,
                                        int num_cache_head, int gqa_size,
                                        int dim_per_head) {
  const int row_num = static_cast<int>(context_len) - 1;

  if (in.getDataType() == ml::train::TensorDim::DataType::FP32) {
    const uint16_t *vcache_data = vcache.getData<uint16_t>();
    for (unsigned int i = 0; i < sequence_len; ++i) {
      const size_t start_idx = i * context_len;
      const float *input =
        in.getData<float>() + start_idx * num_cache_head * gqa_size;
      float *out =
        output.getData<float>() + i * (num_cache_head * gqa_size * dim_per_head);
      nntrainer::compute_fp16vcache_fp32_transposed(
        row_num, input, vcache_data, out, num_cache_head, gqa_size, dim_per_head,
        context_len);
    }
  } else if (in.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    const _FP16 *vcache_data = vcache.getData<_FP16>();
    for (unsigned int i = 0; i < sequence_len; ++i) {
      const size_t start_idx = i * context_len;
      const _FP16 *input =
        in.getData<_FP16>() + start_idx * num_cache_head * gqa_size;
      _FP16 *out =
        output.getData<_FP16>() + i * (num_cache_head * gqa_size * dim_per_head);
      nntrainer::compute_fp16vcache_transposed(
        row_num, input, vcache_data, out, num_cache_head, gqa_size, dim_per_head,
        context_len);
    }
#else
    NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
#endif
  }
}

void CrossMHACoreLayer::setBatch(nntrainer::RunLayerContext &context,
                                 unsigned int batch) {
  context.updateTensor(tensor_idx[AttentionParams::cache_key], batch);
  context.updateTensor(tensor_idx[AttentionParams::cache_value], batch);
  cache_initialized = false;
  cached_seq_len = 0;
}

void CrossMHACoreLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  NNTR_THROW_IF(input_dimensions.size() < 3, std::invalid_argument)
    << "cross attention requires query/key/value input dimensions";

  const ml::train::TensorDim &query_dim = input_dimensions[INOUT_INDEX::QUERY];
  const ml::train::TensorDim &key_dim = input_dimensions[INOUT_INDEX::KEY];
  const ml::train::TensorDim &value_dim = input_dimensions[INOUT_INDEX::VALUE];

  NNTR_THROW_IF(key_dim.height() != value_dim.height(), std::invalid_argument)
    << "cross attention requires key/value sequence lengths to match";
  NNTR_THROW_IF(key_dim.width() != value_dim.width(), std::invalid_argument)
    << "cross attention requires key/value hidden widths to match";

  unsigned int &max_timestep =
    std::get<nntrainer::props::MaxTimestep>(mha_core_props).get();
  unsigned int &max_new_tokens =
    std::get<props::MaxNewTokens>(mha_core_props).get();
  max_timestep = query_dim.height() + max_new_tokens;

  ml::train::TensorDim kv_cache_dim = key_dim;
#ifdef ENABLE_FP16
  kv_cache_dim.setDataType(ml::train::TensorDim::DataType::FP16);
#else
  kv_cache_dim.setDataType(ml::train::TensorDim::DataType::UINT16);
#endif

  context.updateInput(INOUT_INDEX::QUERY, query_dim);
  context.updateInput(INOUT_INDEX::KEY, key_dim);
  context.updateInput(INOUT_INDEX::VALUE, value_dim);
  context.updateOutput(INOUT_INDEX::OUTPUT, query_dim);
  context.updateTensor(tensor_idx[AttentionParams::cache_key], kv_cache_dim);
  context.updateTensor(tensor_idx[AttentionParams::cache_value], kv_cache_dim);

  cache_initialized = false;
  cached_seq_len = 0;
}

void CrossMHACoreLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  (void)context;
}

void CrossMHACoreLayer::calcGradient(nntrainer::RunLayerContext &context) {
  (void)context;
}

void CrossMHACoreLayer::exportTo(nntrainer::Exporter &exporter,
                                 const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(mha_core_props, method, this);
}

void CrossMHACoreLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, mha_core_props);
  LayerImpl::setProperty(remain_props);
}

#ifdef PLUGGABLE

nntrainer::Layer *create_cross_mha_core_layer() {
  auto layer = new CrossMHACoreLayer();
  return layer;
}

void destroy_cross_mha_core_layer(nntrainer::Layer *layer) { delete layer; }

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_cross_mha_core_layer,
                                                   destroy_cross_mha_core_layer};
}

#endif

} // namespace causallm

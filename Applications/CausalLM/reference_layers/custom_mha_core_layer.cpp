// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 hyeonseok Lee <hs89.lee@samsung.com>
 * Copyright (C) 2024 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   custom_mha_core_layer.cpp
 * @date   02 September 2024
 * @see    https://github.com/nnstreamer/nntrainer
 *         https://arxiv.org/abs/1706.03762
 * @author hyeonseok Lee <hs89.lee@samsung.com>
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This code is based on custom_multi_head_attention_layer.cpp.
 *         This code is a part of the break down version of the mha layer.
 *
 */
#include <algorithm>
#include <cmath>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <chrono>
#endif

#include <custom_common_properties.h>
#include <custom_mha_core_layer.h>
#include <layer_context.h>
#include <nntrainer_error.h>
#include <node_exporter.h>

namespace custom {

/************************************************************** */

/**
 * @brief constructor of CustomMHACoreLayer
 */
CustomMHACoreLayer::CustomMHACoreLayer() :
  mha_core_props(
    nntrainer::props::NumHeads(), props::NumHeads_KV(),
    nntrainer::props::ProjectedKeyDim(), nntrainer::props::ProjectedValueDim(),
    nntrainer::props::OutputShape(), nntrainer::props::DropOutRate(),
    nntrainer::props::ReturnAttentionWeight(),
    nntrainer::props::AverageAttentionWeight(), nntrainer::props::MaxTimestep(),
    props::SmartReply(), props::LocalWindowSize(), props::MaxNewTokens()),
  sm(nntrainer::ActivationType::ACT_SOFTMAX),
  epsilon(1e-3),
  cache_index(0),
  num_heads_Q(0),
  num_heads_KV(0),
  head_dim(0),
  cache_shift(false) {
  weight_idx.fill(std::numeric_limits<unsigned>::max());
}

CustomMHACoreLayer::~CustomMHACoreLayer() {}

/************************************************************** */

void CustomMHACoreLayer::finalize(nntrainer::InitLayerContext &context) {

  NNTR_THROW_IF(context.getNumInputs() < 3 || context.getNumInputs() > 4,
                std::invalid_argument)
    << "Multi head Attention layer needs 3 or 4 inputs. (query, key, value and "
       "mask is optional)";
  const bool provide_attention_mask = context.getNumInputs() == 4;

  ml::train::TensorDim::TensorType activation_type = {
    context.getFormat(), context.getActivationDataType()};
  ml::train::TensorDim empty_dim(activation_type);

  const std::vector<ml::train::TensorDim> &input_dims =
    context.getInputDimensions();
  const ml::train::TensorDim &query_dim = input_dims[INOUT_INDEX::QUERY];
  const ml::train::TensorDim &key_dim = input_dims[INOUT_INDEX::KEY];

  /** max time step of this model */
  const unsigned int max_timestep =
    std::get<nntrainer::props::MaxTimestep>(mha_core_props).get();

  /** query_dim = (B, 1, seq_len, H_Q * Head_Dim ) */
  const unsigned int batch_size = query_dim.batch();
  const unsigned int query_height = query_dim.height();
  const unsigned int query_width = query_dim.width();
  /** key_dim = (B, 1, max_seq_len, H_KV * Head_Dim ) */
  const unsigned int key_height = max_timestep;

  /**
   *  @note If NumHeads_KV is set, then use the value. Otherwise,
   *        we initialize num_heads_KV with num_heads_Q.
   */
  num_heads_Q = std::get<nntrainer::props::NumHeads>(mha_core_props).get();
  num_heads_KV = std::get<props::NumHeads_KV>(mha_core_props).empty()
                   ? num_heads_Q
                   : std::get<props::NumHeads_KV>(mha_core_props).get();

  // head_dim
  head_dim = query_width / num_heads_Q;
  // NNTR_THROW_IF(head_dim != key_width / num_heads_KV,
  // std::invalid_argument)
  // << "num_heads_Q and num_heads_KV are not properly given. Please check the "
  //  "num_heads_* are set correctly so that the `head_dim`s are all same for "
  //  "query / key / value";

  /** Tensor for KV-Cache */
  ml::train::TensorDim cache_key_dim(
    {batch_size, 1, max_timestep, num_heads_KV * head_dim}, activation_type);
  ml::train::TensorDim cache_value_dim(
    {batch_size, 1, max_timestep, num_heads_KV * head_dim}, activation_type);
  weight_idx[AttentionParams::cache_key] = context.requestTensor(
    cache_key_dim, "cache_key", nntrainer::Initializer::NONE, false,
    nntrainer::TensorLifespan::MAX_LIFESPAN);
  weight_idx[AttentionParams::cache_value] = context.requestTensor(
    cache_value_dim, "cache_value", nntrainer::Initializer::NONE, false,
    nntrainer::TensorLifespan::MAX_LIFESPAN);
  /** Tensor for projected_key */
  weight_idx[AttentionParams::projected_key] = context.requestTensor(
    cache_key_dim, "projected_key", nntrainer::Initializer::NONE, false,
    nntrainer::TensorLifespan::MAX_LIFESPAN);
  weight_idx[AttentionParams::projected_value] = context.requestTensor(
    cache_value_dim, "projected_value", nntrainer::Initializer::NONE, false,
    nntrainer::TensorLifespan::MAX_LIFESPAN);

  /**
   * @note
   *  Attention mask will be applied for training phase
   *  for the inference version, the attention_mask-related part will be skiped.
   *  The code below is commented out intentionally.
   */
  if (provide_attention_mask) {
    // ml::train::TensorDim attention_mask_dim(
    //   {batch_size, num_heads, query_height, key_height});
    // weight_idx[AttentionParams::attention_mask] = context.requestTensor(
    //   attention_mask_dim, "attention_mask",
    //   nntrainer::Initializer::NONE, false,
    //   nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
  }

  /** Tensor for Attention Weight */
  // @todo check num_heads_Q is correct.
  ml::train::TensorDim attention_weight_dim(
    {batch_size, num_heads_Q, query_height, key_height}, activation_type);
  weight_idx[AttentionParams::attention_weight] = context.requestTensor(
    attention_weight_dim, "attention_weight", nntrainer::Initializer::NONE,
    false, nntrainer::TensorLifespan::ITERATION_LIFESPAN);

  /** Tensor for Attention Output */
  ml::train::TensorDim attention_output_dim(
    {batch_size, 1, query_height, num_heads_Q * head_dim}, activation_type);
  weight_idx[AttentionParams::attention_output] = context.requestTensor(
    attention_output_dim, "attention_output", nntrainer::Initializer::NONE,
    false, nntrainer::TensorLifespan::ITERATION_LIFESPAN);

  /** precompute_freqs will be invoked only once */
  if (freqs_cos == nullptr)
    precompute_freqs(head_dim, max_timestep, 500000);

  /** set Output dimension! - one output */
  std::vector<nntrainer::TensorDim> output_dims(1);
  output_dims[0] = input_dims[0];
  output_dims[0].width(head_dim * num_heads_Q);
  output_dims[0].setTensorType(
    {context.getFormat(), context.getActivationDataType()});
  context.setOutputDimensions(output_dims);
}

/************************************************************** */

/**
 * @note This forwarding function is used for training mode.
 *       This will be implemented ASAP.
 * @date 2024-09-02
 */
void CustomMHACoreLayer::forwarding(nntrainer::RunLayerContext &context,
                                    bool training) {}

/**
 * @note This incremental_forwarding method is invoked for inference mode.
 *       Please note that Transformer Decoder's MHA takes only one sequence at a
 * step. Incremental forwarding function is used for this.
 */
void CustomMHACoreLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int _from, unsigned int _to,
  bool training) {

  if (_from && (_to - _from != 1)) {
    throw std::invalid_argument(
      "if it is not initial forwarding, then step size(difference between to "
      "and from) should be 1");
  }

  unsigned int max_timestep =
    std::get<nntrainer::props::MaxTimestep>(mha_core_props).get();

  unsigned int from = _from;
  unsigned int to = _to;

  if (to >= max_timestep) {
    // initial forwarding
    if (!_from) {
      throw std::invalid_argument(
        "to shouldn't greater than max_timestep for initial forwarding");
    } else {
      // exceeds the kv_cache size
      // KV_cache is shifted!
      cache_shift = true;
      from = max_timestep - 1;
      to = max_timestep;
    }
  }

  // util fn to compute tensor dimension for one step.
  auto get_step_dim = [to, from](const ml::train::TensorDim &dim) {
    auto step_dim = dim;
    step_dim.batch(1);
    step_dim.height(to - from); // One is expected.
    return step_dim;
  };

  /** incremental forwarding for each batch */
  nntrainer::Tensor &query =
    context.getInput(INOUT_INDEX::QUERY); // projected query
  nntrainer::Tensor &key = context.getInput(INOUT_INDEX::KEY); // projected key
  nntrainer::Tensor &value =
    context.getInput(INOUT_INDEX::VALUE); // projected value
  nntrainer::Tensor &output =
    context.getOutput(INOUT_INDEX::OUTPUT); // output to be projected

  nntrainer::Tensor &cache_key =
    context.getTensor(weight_idx[AttentionParams::cache_key]);
  nntrainer::Tensor &cache_value =
    context.getTensor(weight_idx[AttentionParams::cache_value]);
  nntrainer::Tensor &projected_key =
    context.getTensor(weight_idx[AttentionParams::projected_key]);
  nntrainer::Tensor &projected_value =
    context.getTensor(weight_idx[AttentionParams::projected_value]);

  nntrainer::Tensor &attention_weight =
    context.getTensor(weight_idx[AttentionParams::attention_weight]);

  const unsigned int num_heads_Q =
    std::get<nntrainer::props::NumHeads>(mha_core_props).get();
  head_dim = query.width() / num_heads_Q;

  ml::train::TensorDim query_dim =
    query.getDim(); // (B, 1, seq_len, n_heads_Q * head_dim)
  ml::train::TensorDim key_dim =
    key.getDim(); // (B, 1, seq_len, n_heads_KV * head_dim)
  ml::train::TensorDim value_dim =
    value.getDim(); // (B, 1, seq_len, n_heads_KV * head_dim)
  ml::train::TensorDim output_dim =
    output.getDim(); // (B, 1, seq_len, n_heads_Q * head_dim)
  ml::train::TensorDim cache_key_dim =
    cache_key.getDim(); // (B, 1, max_seq_len, n_heads_KV * head_dim)
  ml::train::TensorDim cache_value_dim =
    cache_value.getDim(); // (B, 1, max_seq_len, n_heads_KV * head_dim)
  ml::train::TensorDim attention_weight_dim =
    attention_weight.getDim(); // (B, n_heads_Q, seq_len, seq_len)

  ml::train::TensorDim query_step_dim =
    get_step_dim(query_dim); // (B, 1, from-to, n_heads_Q * head_dim)
  ml::train::TensorDim output_step_dim =
    get_step_dim(output_dim); // (B, 1, from-to, n_heads_Q * head_dim)
  ml::train::TensorDim cache_key_step_dim =
    get_step_dim(cache_key_dim); // (B, 1, from-to, n_heads_KV * head_dim)
  ml::train::TensorDim cache_value_step_dim =
    get_step_dim(cache_value_dim); // (B, 1, from-to, n_heads_KV * head_dim)
  ml::train::TensorDim attention_weight_step_dim =
    get_step_dim(attention_weight_dim); // (B, n_heads_Q, from-to, seq_len)

  attention_weight_step_dim.width(to); // (B, n_heads_Q, 1, to)

  bool smart_reply = std::get<props::SmartReply>(mha_core_props).get();
  unsigned int batch_size = (smart_reply && _from) ? 1 : query_dim.batch();
  // do the incremental forwarding
  for (unsigned int batch = 0; batch < batch_size; ++batch) {
    one_batch_incremental_forwarding(
      batch, _from, from, to, query, key, value, output, cache_key, cache_value,
      projected_key, projected_value, attention_weight, query_dim,
      query_step_dim, key_dim, value_dim, cache_key_dim, cache_key_step_dim,
      cache_value_dim, cache_value_step_dim, attention_weight_dim,
      attention_weight_step_dim, output_dim, output_step_dim);
  }

  // update KV cache
  if (!_from) {
    batch_size = query_dim.batch();
    nntrainer::Tensor cache_key_0_step =
      cache_key.getSharedDataTensor(cache_key_step_dim, 0, true);
    nntrainer::Tensor cache_value_0_step =
      cache_value.getSharedDataTensor(cache_value_step_dim, 0, true);

    for (unsigned int batch = 1; batch < batch_size; ++batch) {
      nntrainer::Tensor cache_key_nth_step = cache_key.getSharedDataTensor(
        cache_key_step_dim,
        batch * cache_key_dim.getFeatureLen() + from * cache_key_dim.width(),
        true);
      nntrainer::Tensor cache_value_nth_step =
        cache_key.getSharedDataTensor(cache_value_step_dim,
                                      batch * cache_value_dim.getFeatureLen() +
                                        from * cache_value_dim.width(),
                                      true);

      cache_key_nth_step.copyData(cache_key_0_step);
      cache_key_nth_step.copyData(cache_value_0_step);
    }
  }
}
void CustomMHACoreLayer::one_batch_incremental_forwarding(
  const unsigned int batch, const unsigned int _from, const unsigned int from,
  const unsigned int to, nntrainer::Tensor &query, nntrainer::Tensor &key,
  nntrainer::Tensor &value, nntrainer::Tensor &output,
  nntrainer::Tensor &cache_key, nntrainer::Tensor &cache_value,
  nntrainer::Tensor &projected_key, nntrainer::Tensor &projected_value,
  nntrainer::Tensor &attention_weight, ml::train::TensorDim &query_dim,
  ml::train::TensorDim &query_step_dim, ml::train::TensorDim &key_dim,
  ml::train::TensorDim &value_dim, ml::train::TensorDim &cache_key_dim,
  ml::train::TensorDim &cache_key_step_dim,
  ml::train::TensorDim &cache_value_dim,
  ml::train::TensorDim &cache_value_step_dim,
  ml::train::TensorDim &attention_weight_dim,
  ml::train::TensorDim &attention_weight_step_dim,
  ml::train::TensorDim &output_dim, ml::train::TensorDim &output_step_dim) {

  /**
   *  cache_key
   *  +--------+                        ->
   *  |        |                        ->
   *  |        |                        ->
   *  |........| from                   ->
   *  |........| to -> b_cache_key_step -> b_cached_key
   *  |        |
   *  +--------+
   *
   */

  /** 1. Load Input Tensors of this batch : b_ denotes a Tensor for this batch
   * **/
  // projected_query_step of this batch
  // (1, 1, 1, n_heads_Q * head_dim)
  nntrainer::Tensor b_projected_query_step = query.getSharedDataTensor(
    query_step_dim, batch * query_dim.getFeatureLen(), true);

  // cache_key_step : internal cache for step of key of this batch
  // (1, 1, from  - to, n_heads_Q * head_dim)
  // copy incoming projected key into the b_cache_key_step
  nntrainer::Tensor b_cache_key_step = cache_key.getSharedDataTensor(
    cache_key_step_dim,
    batch * cache_key_dim.getFeatureLen() + from * cache_key_dim.width(), true);
  b_cache_key_step.copyData(key.getSharedDataTensor(
    cache_key_step_dim, batch * value_dim.getFeatureLen(), true));

  // cache_value_step : internal cache for step of value of this batch
  // (1, 1, from - to, n_heads_Q * head_dim)
  // copy incoming projected key into the b_cache_key_step
  nntrainer::Tensor b_cache_value_step = cache_value.getSharedDataTensor(
    cache_value_step_dim,
    batch * cache_value_dim.getFeatureLen() + from * cache_value_dim.width(),
    true);
  b_cache_value_step.copyData(value.getSharedDataTensor(
    cache_value_step_dim, batch * cache_value_dim.getFeatureLen(), true));

  // part of cached key/value for this batch
  // cached_key : (1, 1, to, n_heads_Q * head_dim)
  // cached_value : (1, 1, to, n_heads_Q * head_dim)
  ml::train::TensorDim cached_key_dim = cache_key_dim;
  ml::train::TensorDim cached_value_dim = cache_value_dim;
  cached_key_dim.height(to);
  cached_value_dim.height(to);
  nntrainer::Tensor b_cached_key = cache_key.getSharedDataTensor(
    cached_key_dim, batch * cache_key_dim.getFeatureLen(), true);
  nntrainer::Tensor b_cached_value = cache_value.getSharedDataTensor(
    cached_value_dim, batch * cache_value_dim.getFeatureLen(), true);

  nntrainer::Tensor b_projected_key_step = projected_key.getSharedDataTensor(
    cached_key_dim, batch * cache_key_dim.getFeatureLen(), true);
  nntrainer::Tensor b_projected_value_step =
    projected_value.getSharedDataTensor(
      cached_value_dim, batch * cache_value_dim.getFeatureLen(), true);

  // attention weight & output's step tensor
  nntrainer::Tensor attention_weight_step =
    attention_weight.getSharedDataTensor(
      attention_weight_step_dim, batch * attention_weight_dim.getFeatureLen(),
      true);
  nntrainer::Tensor attention_output_step = output.getSharedDataTensor(
    output_step_dim, batch * output_dim.getFeatureLen(), true);

  /**
   * 2. Apply Rotary Embedding  for query and key
   * */
  apply_rotary_emb_tensor(b_projected_query_step, head_dim, _from);
  apply_rotary_emb_tensor(b_cache_key_step, head_dim, _from);

  /**
   * 3. Reshape & Transpose the Tensors
   * */
  b_projected_query_step.reshape(
    ml::train::TensorDim({1, to - from, num_heads_Q, head_dim}));
  b_cached_key.reshape(ml::train::TensorDim({1, to, num_heads_KV, head_dim}));
  b_cached_value.reshape(ml::train::TensorDim({1, to, num_heads_KV, head_dim}));

  if (to - from != 1)
    b_projected_query_step.transpose("1:0:2", b_projected_query_step);

  b_cached_key.transpose("1:0:2", b_projected_key_step);
  b_cached_value.transpose("1:0:2", b_projected_value_step);

  b_projected_query_step.reshape(
    ml::train::TensorDim({1 * num_heads_Q, 1, to - from, head_dim}));
  b_projected_key_step.reshape(
    ml::train::TensorDim({1 * num_heads_KV, 1, to, head_dim}));
  b_projected_value_step.reshape(
    ml::train::TensorDim({1 * num_heads_KV, 1, to, head_dim}));

  attention_weight_step.reshape(
    ml::train::TensorDim({1 * num_heads_Q, 1, to - from, to}));
  attention_output_step.reshape(
    ml::train::TensorDim({1 * num_heads_Q, 1, to - from, head_dim}));

  /** scaled dot product attention */
  b_projected_query_step.dotBatched(b_projected_key_step, attention_weight_step,
                                    false, true);
  attention_weight_step.multiply_i(1 / sqrt((float)head_dim));

  if (!from) {
    unsigned int mask_size = attention_weight_step.getDim().width();
    unsigned int mask_dim_height = mask_size;
    unsigned int mask_dim_width = mask_size;

    nntrainer::Tensor causal_mask(ml::train::TensorDim{
      1, 1, mask_size, mask_size, attention_weight_step.getTensorType()});

    causal_mask.setZero();

#ifdef ENABLE_FP16
#define _MASK_NUM -1e4
#else
#define _MASK_NUM -3.4028e38
#endif

    for (unsigned int i = 0; i < mask_dim_height; ++i) {
      for (unsigned int j = i + 1; j < mask_dim_width; ++j) {
        causal_mask.setValue(0, 0, i, j, _MASK_NUM);
      }
    }

    size_t local_window_size =
      std::get<props::LocalWindowSize>(mha_core_props).get();

    for (unsigned int i = 0; i < mask_dim_height; ++i) {
      for (unsigned int j = 0; j + local_window_size <= i; ++j) {
        causal_mask.setValue(0, 0, i, j, _MASK_NUM);
      }
    }

    attention_weight_step.add_i(causal_mask);
  } else {
    size_t local_window_size =
      std::get<props::LocalWindowSize>(mha_core_props).get();

    if (local_window_size != UINT_MAX) {
      unsigned int mask_size = attention_weight_step.getDim().width();
      unsigned int mask_dim_height = mask_size;
      unsigned int mask_dim_width = mask_size;

      nntrainer::Tensor causal_mask(ml::train::TensorDim{
        1, 1, 1, mask_size, attention_weight_step.getTensorType()});

      causal_mask.setZero();

#ifdef ENABLE_FP16
#define _MASK_NUM -1e4
#else
#define _MASK_NUM -3.4028e38
#endif

      for (unsigned int j = 0; j + local_window_size < mask_dim_width; ++j) {
        causal_mask.setValue(0, 0, 0, j, _MASK_NUM);
      }

      attention_weight_step.add_i(causal_mask);
    }
  }

  sm.run_fn(attention_weight_step, attention_weight_step);
  attention_weight_step.dotBatched(b_projected_value_step,
                                   attention_output_step);

  if (to - from != 1) {
    attention_output_step.reshape(
      ml::train::TensorDim({1, num_heads_Q, to - from, head_dim}));
    attention_output_step.transpose("1:0:2", attention_output_step);
  }
  attention_output_step.reshape(
    ml::train::TensorDim({(to - from), 1, 1, head_dim * num_heads_Q}));

  if (cache_shift) {
    if (cache_key.getDataType() == ml::train::TensorDim::DataType::FP32) {
      float *buf = cache_key.getAddress<float>(batch, 0, 1, 0);
      float *dbuf = cache_key.getAddress<float>(batch, 0, 0, 0);
      memcpy(dbuf, buf,
             (cache_key.getDim().getFeatureLen() - cache_key.width()) *
               sizeof(float));
      buf = cache_value.getAddress<float>(batch, 0, 1, 0);
      dbuf = cache_value.getAddress<float>(batch, 0, 0, 0);
      memcpy(dbuf, buf,
             (cache_value.getDim().getFeatureLen() - cache_value.width()) *
               sizeof(float));
    } else if (cache_key.getDataType() ==
               ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16

      _FP16 *buf = cache_key.getAddress<_FP16>(batch, 0, 1, 0);
      _FP16 *dbuf = cache_key.getAddress<_FP16>(batch, 0, 0, 0);
      memcpy(dbuf, buf,
             (cache_key.getDim().getFeatureLen() - cache_key.width()) *
               sizeof(_FP16));
      buf = cache_value.getAddress<_FP16>(batch, 0, 1, 0);
      dbuf = cache_value.getAddress<_FP16>(batch, 0, 0, 0);
      memcpy(dbuf, buf,
             (cache_key.getDim().getFeatureLen() - cache_value.width()) *
               sizeof(_FP16));
#else
      throw std::invalid_argument("enable-fp16 is not set");
#endif
    }
  }
}

/************************************************************** */

/**
 * @brief rotary embedding-related member function
 */
void CustomMHACoreLayer::precompute_freqs(int head_dim, unsigned int seq_len,
                                          float theta) {
  // compute the freqs only when it is the first time to call this function
  if (freqs_cos != nullptr)
    return;

  // theta_i = 10000^(-2(i-1)/dim) for i = [1, 2, ... , dim/2]
  // head_dim should be divisible by 2
  unsigned int half_ = head_dim / 2;
  for (unsigned int i = 0; i < half_; ++i) {
    thetas.push_back(1.0 /
                     (std::pow(theta, (2 * i) / static_cast<float>(head_dim))));
  }

  // cos / sin
  auto cos = new std::vector<std::vector<float>>();
  cos->assign(seq_len, std::vector<float>(head_dim, 0));
  auto sin = new std::vector<std::vector<float>>();
  sin->assign(seq_len, std::vector<float>(head_dim, 0));

  // update cos / sin frequency
  for (unsigned int i = 0; i < seq_len; ++i) {

#ifdef USE_NEON
    nntrainer::calc_trigonometric_vals_dup(
      half_, thetas.data(), (*cos)[i].data(), (*sin)[i].data(), i);
#else
    for (unsigned int j = 0; j < half_; ++j) {
      float angle = i * thetas[j];
      (*cos)[i][j] = std::cos(angle);
      (*cos)[i][j + half_] = std::cos(angle); // repeated 2 times

      (*sin)[i][j] = std::sin(angle);
      (*sin)[i][j + half_] = std::sin(angle); // repeated 2 times
    }
#endif
  }
  freqs_cos = cos;
  freqs_sin = sin;
};

/**
 * @brief rotary embedding-related member function
 */
void CustomMHACoreLayer::apply_rotary_emb_tensor(nntrainer::Tensor &in,
                                                 unsigned int dim,
                                                 unsigned int from) {
  nntrainer::Tensor out(in.getDim());
  float value = 0;
  float transformed_value = 0.0;
  unsigned int half_ = dim / 2;
  unsigned int max_timestep =
    std::get<nntrainer::props::MaxTimestep>(mha_core_props).get();

  std::vector<float> *cos_ = nullptr;
  std::vector<float> *sin_ = nullptr;

  if (from >= max_timestep) {
    cos_ = new std::vector<float>(dim);
    sin_ = new std::vector<float>(dim);
#ifdef USE_NEON
    nntrainer::calc_trigonometric_vals_dup(half_, thetas.data(), cos_->data(),
                                           sin_->data(), from);
#else
    for (unsigned int i = 0; i < half_; ++i) {
      float angle = from * thetas[i];
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
                  -1.0 *
                  static_cast<float>(in.getValue<_FP16>(b, c, h, half_ + span));
              } else {
                transformed_value =
                  static_cast<float>(in.getValue<_FP16>(b, c, h, span - half_));
              }
              out.setValue(b, c, h, span,
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

void CustomMHACoreLayer::setBatch(nntrainer::RunLayerContext &context,
                                  unsigned int batch) {

  const float dropout_rate =
    std::get<nntrainer::props::DropOutRate>(mha_core_props).get();
  context.updateTensor(weight_idx[AttentionParams::cache_key], batch);
  context.updateTensor(weight_idx[AttentionParams::cache_value], batch);
  context.updateTensor(weight_idx[AttentionParams::attention_weight], batch);
  if (dropout_rate > epsilon) {
    context.updateTensor(weight_idx[AttentionParams::dropout_mask], batch);
  }
}

void CustomMHACoreLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  unsigned int height = input_dimensions[0].height();
  unsigned int &max_timestep =
    std::get<nntrainer::props::MaxTimestep>(mha_core_props).get();
  unsigned int &max_new_tokens =
    std::get<props::MaxNewTokens>(mha_core_props).get();
  max_timestep = height + max_new_tokens;

  ml::train::TensorDim kv_dim = input_dimensions[0];
  kv_dim.width(kv_dim.width() / (num_heads_Q / num_heads_KV));

  ml::train::TensorDim kv_cache_dim = kv_dim;
  kv_cache_dim.height(max_timestep);

  ml::train::TensorDim attention_weight_dim =
    context.getTensor(weight_idx[AttentionParams::attention_weight]).getDim();
  attention_weight_dim.height(max_timestep);
  attention_weight_dim.width(max_timestep);

  precompute_freqs(head_dim, max_timestep, 500000);

  context.updateInput(INOUT_INDEX::QUERY, input_dimensions[0]);
  context.updateInput(INOUT_INDEX::KEY, kv_dim);
  context.updateInput(INOUT_INDEX::VALUE, kv_dim);
  context.updateOutput(0, input_dimensions[0]);

  unsigned int height_axis =
    context.getTensor(weight_idx[AttentionParams::cache_key]).getFormat() ==
        nntrainer::Tformat::NCHW
      ? 2
      : 1;
  unsigned int width_axis =
    context.getTensor(weight_idx[AttentionParams::cache_key]).getFormat() ==
        nntrainer::Tformat::NCHW
      ? 3
      : 2;

  context.updateTensor(weight_idx[AttentionParams::cache_key], kv_cache_dim);
  context.updateTensor(weight_idx[AttentionParams::cache_value], kv_cache_dim);
  context.updateTensor(weight_idx[AttentionParams::projected_key],
                       kv_cache_dim);
  context.updateTensor(weight_idx[AttentionParams::projected_value],
                       kv_cache_dim);
  context.updateTensor(weight_idx[AttentionParams::attention_weight],
                       attention_weight_dim);
  context.updateTensor(weight_idx[AttentionParams::attention_output],
                       kv_cache_dim);

  // const float dropout_rate =
  //   std::get<nntrainer::props::DropOutRate>(mha_core_props).get();
  // if (dropout_rate > epsilon) {
  //   ml::train::TensorDim dropout_mask_dim = input_dimensions[0];
  //   context.updateTensor(weight_idx[AttentionParams::dropout_mask],
  //   kv_cache_dim);
  // }
}

void CustomMHACoreLayer::calcDerivative(nntrainer::RunLayerContext &context) {}

void CustomMHACoreLayer::calcGradient(nntrainer::RunLayerContext &context) {}

void CustomMHACoreLayer::exportTo(
  nntrainer::Exporter &exporter, const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(mha_core_props, method, this);
}

void CustomMHACoreLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, mha_core_props);
  LayerImpl::setProperty(remain_props);
}

#ifdef PLUGGABLE

nntrainer::Layer *create_mha_core_layer() {
  auto layer = new CustomMHACoreLayer();
  return layer;
}

void destroy_custom_mha_core_layer(nntrainer::Layer *layer) { delete layer; }

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{
  create_mha_core_layer, destroy_custom_mha_core_layer};
}

#endif

} // end of namespace custom

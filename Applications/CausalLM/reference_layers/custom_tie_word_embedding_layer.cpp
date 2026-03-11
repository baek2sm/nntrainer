// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   custom_tie_word_embedding_layer.cpp
 * @date   21 May 2025
 * @brief  This is Embedding Layer Class of Neural Network
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <cpu_backend.h>
#include <custom_dot_wrapper.h>
#include <custom_tie_word_embedding_layer.h>
#include <int4_utils.h>
#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <tensor.h>
#include <tensor_dim.h>
#include <util_func.h>

#include <iostream>

namespace custom {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum TieWordEmbeddingParams {
  weight,
  bias,
  candidate_weight,
  candidate_hidden_step
};

CustomTieWordEmbeddingLayer::CustomTieWordEmbeddingLayer() :
  LayerImpl(),
  custom_embedding_props(nntrainer::props::InDim(), nntrainer::props::OutDim(),
                         props::SmartReply(), nntrainer::props::Unit(),
                         props::UseVocabSelection(), props::LshChoices()) {

  weight_idx.fill(std::numeric_limits<unsigned>::max());
}

void CustomTieWordEmbeddingLayer::finalize(
  nntrainer::InitLayerContext &context) {

  mode_ = std::get<nntrainer::props::Unit>(custom_embedding_props).empty()
            ? mode::embedding
            : mode::lm_head;

  if (mode_ == mode::embedding)
    finalize_embedding(context);
  else if (mode_ == mode::lm_head)
    finalize_lmhead(context);
  else
    throw std::invalid_argument("Invalid mode");
}

void CustomTieWordEmbeddingLayer::finalize_embedding(
  nntrainer::InitLayerContext &context) {

  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "Embedding layer takes only one input";

  const nntrainer::TensorDim &input_dim =
    context.getInputDimensions()[SINGLE_INOUT_IDX];
  NNTR_THROW_IF(input_dim.channel() != 1, std::invalid_argument)
    << "Embedding layer takes only one for channel size";

  NNTR_THROW_IF(input_dim.getDataType() != nntrainer::TensorDim::DataType::FP32,
                std::invalid_argument)
    << "Embedding layer takes only FP32 input data";

  auto &weight_regularizer =
    std::get<nntrainer::props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<nntrainer::props::WeightRegularizerConstant>(*layer_impl_props);
  //  auto &weight_initializer =
  //    std::get<nntrainer::props::WeightInitializer>(*layer_impl_props);
  auto weight_initializer = nntrainer::props::InitializerInfo::Enum::NONE;
  auto &weight_decay =
    std::get<nntrainer::props::WeightDecay>(*layer_impl_props);

  unsigned int in_dim =
    std::get<nntrainer::props::InDim>(custom_embedding_props);
  unsigned int out_dim =
    std::get<nntrainer::props::OutDim>(custom_embedding_props);

  nntrainer::TensorDim output_dim = input_dim;

  // output_dim expected as hidden x num input (batch size)
  output_dim.height(input_dim.width());
  output_dim.width(out_dim);
  output_dim.setTensorType(
    {context.getFormat(), context.getActivationDataType()});
  context.setOutputDimensions({output_dim});

  nntrainer::TensorDim dim = output_dim;

  dim.setTensorType({context.getFormat(), context.getWeightDataType()});

  /// @note it is recommend to have a custom_embedding_int4_layer
  if (context.getWeightDataType() == nntrainer::TensorDim::DataType::QINT4) {
    dim.height(nntrainer::align(in_dim, 32));
  } else {
    dim.height(in_dim);
  }
  dim.width(out_dim);
  dim.batch(1);

  weight_idx[TieWordEmbeddingParams::weight] = context.requestWeight(
    dim, weight_initializer, weight_regularizer, weight_regularizer_constant,
    weight_decay, "Embedding", true);
}

void CustomTieWordEmbeddingLayer::finalize_lmhead(
  nntrainer::InitLayerContext &context) {
  auto &weight_regularizer =
    std::get<nntrainer::props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<nntrainer::props::WeightRegularizerConstant>(*layer_impl_props);
  auto weight_initializer = nntrainer::props::InitializerInfo::Enum::NONE;
  // auto &weight_initializer =
  //   std::get<nntrainer::props::WeightInitializer>(*layer_impl_props);
  auto &weight_decay =
    std::get<nntrainer::props::WeightDecay>(*layer_impl_props);
  auto &bias_decay = std::get<nntrainer::props::BiasDecay>(*layer_impl_props);
  auto &bias_initializer =
    std::get<nntrainer::props::BiasInitializer>(*layer_impl_props);
  auto &disable_bias =
    std::get<nntrainer::props::DisableBias>(*layer_impl_props);

  auto unit = std::get<nntrainer::props::Unit>(custom_embedding_props).get();

  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "lm head layer takes only one input";

  std::vector<ml::train::TensorDim> output_dims(1);

  /// @todo fc actaully supports multidimensions. EffDimFlag shouldn't be fixed
  /// like this.
  context.setEffDimFlagInputDimension(0, 0b1001);
  context.setDynDimFlagInputDimension(0, 0b1000);

  bool is_nchw = (context.getFormat() == nntrainer::Tformat::NCHW);
  /** set output dimensions */
  auto const &in_dim = context.getInputDimensions()[0];
  output_dims[0] = in_dim;
  is_nchw ? output_dims[0].width(unit) : output_dims[0].channel(unit);

  output_dims[0].setTensorType(
    {context.getFormat(), context.getActivationDataType()});

  context.setOutputDimensions(output_dims);

  /** set weight specifications */
  // @todo : This NCHW format setting is just temporal, it needs to be set by
  // global configuration
  ml::train::TensorDim bias_dim(
    1, is_nchw ? 1 : unit, 1, is_nchw ? unit : 1,
    ml::train::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()),
    is_nchw ? 0b0001 : 0b0100);

  ///@note TieWordEmbedding layer's tensor dim is transposed dim of user-defined
  /// dim
  /// so it can reuse embedding layer.
  ml::train::TensorDim weight_dim(
    1, is_nchw ? 1 : in_dim.channel(), is_nchw ? unit : 1,
    is_nchw ? in_dim.width() : unit,
    ml::train::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()),
    is_nchw ? 0b0011 : 0b0101);

  weight_idx[TieWordEmbeddingParams::weight] = context.requestWeight(
    weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "Embedding", true);

  if (disable_bias.empty() || disable_bias.get() == false) {
    weight_idx[TieWordEmbeddingParams::bias] = context.requestWeight(
      bias_dim, bias_initializer, nntrainer::WeightRegularizer::NONE, 1.0f,
      bias_decay, "bias", true);
  }

  auto use_vocab_selection =
    std::get<props::UseVocabSelection>(custom_embedding_props).get();

  if (use_vocab_selection) {
    auto lsh_choices =
      std::get<props::LshChoices>(custom_embedding_props).get();

    ml::train::TensorDim candidate_weight_dim(
      1, is_nchw ? 1 : lsh_choices, is_nchw ? lsh_choices : in_dim.channel(),
      is_nchw ? in_dim.width() : 1,
      ml::train::TensorDim::TensorType(context.getFormat(),
                                       context.getWeightDataType()));

    weight_idx[TieWordEmbeddingParams::candidate_weight] =
      context.requestTensor(candidate_weight_dim, "candidate_weight",
                            nntrainer::Initializer::NONE, false,
                            nntrainer::TensorLifespan::ITERATION_LIFESPAN);

    ml::train::TensorDim candidate_hidden_step_dim(
      1, 1, 1, lsh_choices,
      ml::train::TensorDim::TensorType(context.getFormat(),
                                       context.getWeightDataType()));

    weight_idx[TieWordEmbeddingParams::candidate_hidden_step] =
      context.requestTensor(candidate_hidden_step_dim, "candidate_hidden_step",
                            nntrainer::Initializer::NONE, false,
                            nntrainer::TensorLifespan::ITERATION_LIFESPAN);
  }
}

void CustomTieWordEmbeddingLayer::setProperty(
  const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, custom_embedding_props);
  LayerImpl::setProperty(remain_props);
}

void CustomTieWordEmbeddingLayer::forwarding(
  nntrainer::RunLayerContext &context, bool training) {}

void CustomTieWordEmbeddingLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {

  if (mode_ == mode::embedding)
    incremental_forwarding_embedding(context, from, to, training);
  else if (mode_ == mode::lm_head)
    incremental_forwarding_lmhead(context, from, to, training);
  else
    throw std::invalid_argument("lm_head is not supported yet");
}

void CustomTieWordEmbeddingLayer::incremental_forwarding_embedding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  /// @todo get input and output dimension from input_ and hidden itself
  unsigned int in_dim =
    std::get<nntrainer::props::InDim>(custom_embedding_props);
  unsigned int out_dim =
    std::get<nntrainer::props::OutDim>(custom_embedding_props);

  unsigned int _from = from;

  if (from) {
    NNTR_THROW_IF(to - from != 1, std::invalid_argument)
      << "incremental step size is not 1";
    from = 0;
    to = 1;
  }

  nntrainer::Tensor &weight =
    context.getWeight(weight_idx[TieWordEmbeddingParams::weight]);
  nntrainer::Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);

  nntrainer::TensorDim out_tensor_dim =
    nntrainer::TensorDim({1, 1, 1, out_dim}, hidden_.getTensorType());

  bool smart_reply = std::get<props::SmartReply>(custom_embedding_props);

  unsigned int b_size = input_.batch();

  if (smart_reply && !_from) {
    b_size = 1;
  }

  for (unsigned int b = 0; b < b_size; ++b) {
    float *in_data =
      input_.getAddress<float>(b * input_.getDim().getFeatureLen());

    nntrainer::Tensor batchsliced_hidden = hidden_.getBatchSlice(b, 1);

#pragma omp parallel for
    for (int i = from; i < to; ++i) {
      unsigned int embed_idx = static_cast<unsigned int>(in_data[i]);
      if (embed_idx >= in_dim) {
        throw std::invalid_argument("input word index is greater than in_dim");
      }

      nntrainer::Tensor cur_weight =
        weight.getSharedDataTensor(out_tensor_dim, out_dim * embed_idx);

      nntrainer::Tensor out_tensor = batchsliced_hidden.getSharedDataTensor(
        out_tensor_dim, out_dim * (i - from));

      const int K = weight.width();
      const int N = weight.height();

      if (weight.getDataType() == nntrainer::TensorDim::DataType::Q6_K) {
        ///@note this should be replaced with quantizer operation
        int num_blocks_per_row = (weight.width() + 256 - 1) / 256;
        nntrainer::dequantize_row_q6_K(
          (void *)((char *)weight.getData<uint8_t>() +
                   (210 * num_blocks_per_row) * embed_idx),
          out_tensor.getData(), out_dim);
      } else if (weight.getDataType() ==
                 nntrainer::TensorDim::DataType::QINT4) {
        nntrainer::Int4Utils::dequantizePackedRow(
          weight.getData<uint8_t>(), weight.getScale<uint16_t>(1), N, K, 32,
          embed_idx, out_tensor.getData());
      } else {
        out_tensor.copyData(cur_weight);
      }
    }

#ifdef DEBUG
    std::cout << context.getName() << " : "
              << "\n input:" << input_ << "\n weight: " << weight
              << "\n hidden: " << hidden_ << std::endl;
#endif
  }
}

void CustomTieWordEmbeddingLayer::incremental_forwarding_lmhead(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  nntrainer::Tensor weight =
    context.getWeight(weight_idx[TieWordEmbeddingParams::weight]);

  nntrainer::Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);

  ml::train::TensorDim input_dim = input_.getDim();
  ml::train::TensorDim hidden_dim = hidden_.getDim();

  ml::train::TensorDim input_step_dim = input_dim;
  ml::train::TensorDim hidden_step_dim = hidden_dim;

  unsigned int _from = from;

  if (from) {
    NNTR_THROW_IF(to - from != 1, std::invalid_argument)
      << "incremental step size is not 1";
    from = 0;
    to = 1;
  }

  input_step_dim.batch(1);
  input_step_dim.height(1);
  hidden_step_dim.batch(1);
  hidden_step_dim.height(1);

  bool smart_reply = std::get<props::SmartReply>(custom_embedding_props).get();

  unsigned int b_size = input_dim.batch();
  //  unsigned omp_num = 4;
  if (smart_reply && !_from) {
    b_size = 1;
    //    omp_num = 1;
  }

  // #pragma omp parallel for num_threads(omp_num)
  for (unsigned int b = 0; b < b_size; ++b) {
    nntrainer::Tensor input_step = input_.getSharedDataTensor(
      input_step_dim,
      b * input_dim.getFeatureLen() +
        (to - from == 1 ? 0 : (to - 1) * input_.width()),
      true);
    nntrainer::Tensor hidden_step = hidden_.getSharedDataTensor(
      hidden_step_dim,
      b * hidden_dim.getFeatureLen() +
        (to - from == 1 ? 0 : (to - 1) * hidden_.width()),
      true);

    auto use_vocab_selection =
      std::get<props::UseVocabSelection>(custom_embedding_props).get();

    if (use_vocab_selection) {
#ifdef ENABLE_FP16
      auto lsh_choices =
        std::get<props::LshChoices>(custom_embedding_props).get();
      auto vocab = VocabSelection->getVocabs(input_step);

      hidden_step.setValue(0);

      ml::train::TensorDim weight_T_ith_choice_dim = weight->getDim();
      weight_T_ith_choice_dim.width(1);
      ml::train::TensorDim hidden_step_ith_choice_dim = hidden_step_dim;
      hidden_step_ith_choice_dim.width(1);
      nntrainer::Tensor weight_T_ith_choice;

      for (unsigned int i = 0; i < lsh_choices; ++i) {
        weight_T_ith_choice = weight_T->getSharedDataTensor(
          weight_T_ith_choice_dim, vocab[0][i] * input_step.width(), true);
        nntrainer::Tensor hidden_step_ith_choice =
          hidden_step.getSharedDataTensor(hidden_step_ith_choice_dim,
                                          vocab[0][i], true);
        input_step.dot(weight_T_ith_choice, hidden_step_ith_choice, false,
                       false);
      }
#else
      throw std::invalid_argument("FP16 is not enabled");
#endif
    } else {
      ///@note Since tieword embedding shares the weight with embedding,
      /// the weight is transposed. Thus, the dot product should be consider
      /// this. current BCQ weight custom_dot is not yet supported for this
      /// transposed case.
      NNTR_THROW_IF(weight.getDataType() == nntrainer::TensorDim::DataType::BCQ,
                    std::invalid_argument)
        << "weight type is not supported for custom tie word embedding layer";

      input_step.dot(weight, hidden_step, false, true);
    }

    if (auto &disable_bias =
          std::get<nntrainer::props::DisableBias>(*layer_impl_props);
        disable_bias.empty() || disable_bias.get() == false) {
      nntrainer::Tensor &bias =
        context.getWeight(weight_idx[TieWordEmbeddingParams::bias]);
      hidden_step.add_i(bias);
    }
  }
}

void CustomTieWordEmbeddingLayer::calcDerivative(
  nntrainer::RunLayerContext &context) {
  throw nntrainer::exception::not_supported(
    "calcDerivative for Embedding layer is not supported");
}

void CustomTieWordEmbeddingLayer::calcGradient(
  nntrainer::RunLayerContext &context) {}

void CustomTieWordEmbeddingLayer::exportTo(
  nntrainer::Exporter &exporter, const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(custom_embedding_props, method, this);
}

void CustomTieWordEmbeddingLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  nntrainer::TensorDim in_dim = context.getInput(SINGLE_INOUT_IDX).getDim();
  nntrainer::TensorDim out_dim = context.getOutput(SINGLE_INOUT_IDX).getDim();

  unsigned int height = input_dimensions[0].height();

  if (mode_ == mode::embedding) {
    in_dim.width(height);
  } else {
    in_dim.height(height);
  }
  out_dim.height(height);

  context.updateInput(SINGLE_INOUT_IDX, in_dim);
  context.updateOutput(SINGLE_INOUT_IDX, out_dim);
}

void CustomTieWordEmbeddingLayer::read(
  std::ifstream &file, nntrainer::RunLayerContext &context, bool opt_var,
  ml::train::ExecutionMode mode, bool trainable,
  nntrainer::TensorDim::DataType definedWeightDataType, bool fsu,
  size_t start_offset, bool read_from_offset, int file_fd) {

  // Only read when mode is embedding
  if (mode_ == mode::embedding) {
    if (opt_var) {
      for (unsigned int i = 0; i < context.getNumWeights(); ++i) {
        if (context.isGradientLastAccess(i) && trainable) {
          /// @note read optimizer variables
          for (unsigned int j = 0; j < context.getNumWeightOptVar(i); ++j) {
            context.getWeightOptVar(i, j).read(file);
          }
        }
      }
    } else {
      for (unsigned int i = 0; i < context.getNumWeights(); ++i) {
        /// @note shared weights are only be read at the first acecss
        if (context.isGradientFirstAccess(i)) {
          context.getWeight(i).read(file);
          if (context.isMixedPrecision(i) && trainable &&
              !context.getWeightFP32(i).empty()) {
            context.getWeightFP32(i).copyData(context.getWeight(i));
          }
        }
      }
    }
  }
}

void CustomTieWordEmbeddingLayer::read(
  nntrainer::ReadSource src, nntrainer::RunLayerContext &context, bool opt_var,
  ml::train::ExecutionMode mode, bool trainable,
  nntrainer::TensorDim::DataType definedWeightDataType, bool fsu,
  size_t start_offset, bool read_from_offset) {

  // Only read when mode is embedding
  if (mode_ == mode::embedding) {
    if (opt_var) {
      for (unsigned int i = 0; i < context.getNumWeights(); ++i) {
        if (context.isGradientLastAccess(i) && trainable) {
          /// @note read optimizer variables
          for (unsigned int j = 0; j < context.getNumWeightOptVar(i); ++j) {
            context.getWeightOptVar(i, j).read(src, 0, read_from_offset);
          }
        }
      }
    } else {
      for (unsigned int i = 0; i < context.getNumWeights(); ++i) {
        /// @note shared weights are only be read at the first acecss
        if (context.isGradientFirstAccess(i)) {
          context.getWeight(i).read(src, 0, read_from_offset);
          if (context.isMixedPrecision(i) && trainable &&
              !context.getWeightFP32(i).empty()) {
            context.getWeightFP32(i).copyData(context.getWeight(i));
          }
        }
      }
    }
  }
}

void CustomTieWordEmbeddingLayer::save(
  std::ofstream &file, nntrainer::RunLayerContext &run_context, bool opt_var,
  ml::train::ExecutionMode mode, bool trainable,
  nntrainer::TensorDim::DataType definedWeightDataType) const {
  // Only read when mode is embedding
  if (mode_ == mode::embedding) {
    if (opt_var) {
      for (unsigned int i = 0; i < run_context.getNumWeights(); ++i) {
        if (run_context.isGradientFirstAccess(i) && trainable) {
          // @note save optimizer variables
          if (run_context.weightHasGradient(i)) {
            for (unsigned int j = 0; j < run_context.getNumWeightOptVar(i);
                 ++j) {
              run_context.getWeightOptVar(i, j).save(file);
            }
          }
        }
      }
    } else {
      // @note shared weights are only be saved at the first access
      for (unsigned int i = 0; i < run_context.getNumWeights(); ++i) {
        if (run_context.isGradientFirstAccess(i)) {
          run_context.getWeight(i).save(file);
        }
      }
    }
  }
}

#ifdef PLUGGABLE

nntrainer::Layer *create_custom_tie_word_embedding_layer() {
  auto layer = new CustomTieWordEmbeddingLayer();
  std::cout << "embedding layer created\n";
  return layer;
}

void destroy_custom_tie_word_embedding_layer(nntrainer::Layer *layer) {
  std::cout << "embeddinglayer is deleted\n";
  delete layer;
}

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{
  create_custom_tie_word_embedding_layer,
  destroy_custom_tie_word_embedding_layer};
}

#endif

} // namespace custom

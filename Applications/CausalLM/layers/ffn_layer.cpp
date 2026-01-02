/**
 * Copyright (C) 2020 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *
 * @file	fc_layer.cpp
 * @date	14 May 2020
 * @brief	This is Fully Connected Layer Class for Neural Network
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Eunju Yang <ej.yang@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <causallm_common_properties.h>
#include <common_properties.h>
#include <dot_wrapper.h>
#include <ffn_layer.h>

#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <q4_0_utils.h>
#include <util_func.h>

#include <iostream>

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum FCParams { FFN1, FFN2 };
enum LORAParams { loraA, loraB, loraTmp, loraOut };

FFNLayer::FFNLayer() :
  LayerImpl(),
  lora_scaling(1.0f),
  fc_props(nntrainer::props::Unit(), nntrainer::props::LoraRank(),
           nntrainer::props::LoraAlpha(), {}, props::LoraEnable(),
           props::SkipPrefill()) {
  weight_idx.fill(std::numeric_limits<unsigned>::max());
  lora_idx.fill(std::numeric_limits<unsigned>::max());
}

void FFNLayer::finalize(nntrainer::InitLayerContext &context) {
  auto &weight_regularizer =
    std::get<nntrainer::props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<nntrainer::props::WeightRegularizerConstant>(*layer_impl_props);
  /** Initialize with NONE for BCQ weight */
  // auto &weight_initializer =
  // std::get<nntrainer::props::WeightInitializer>(*layer_impl_props);
  auto weight_initializer = nntrainer::props::InitializerInfo::Enum::NONE;
  /****************************************/
  auto &weight_decay =
    std::get<nntrainer::props::WeightDecay>(*layer_impl_props);

  const auto &unit = std::get<nntrainer::props::Unit>(fc_props).get();
  const auto &lora_rank =
    (std::get<nntrainer::props::LoraRank>(fc_props).empty())
      ? 0
      : std::get<nntrainer::props::LoraRank>(fc_props).get();
  lora_scaling =
    (lora_rank && !std::get<nntrainer::props::LoraAlpha>(fc_props).empty())
      ? (float)std::get<nntrainer::props::LoraAlpha>(fc_props) / lora_rank
      : 1;
  std::vector<nntrainer::props::TensorDataType> t_dtype;
  if (!std::get<std::vector<nntrainer::props::TensorDataType>>(fc_props)
         .empty()) {
    t_dtype = std::get<std::vector<nntrainer::props::TensorDataType>>(fc_props);
  }

  const auto &lora_enable =
    (std::get<causallm::props::LoraEnable>(fc_props).empty())
      ? false
      : std::get<causallm::props::LoraEnable>(fc_props).get();

  if (!std::get<props::SkipPrefill>(fc_props).empty()) {
    skip_prefill = std::get<props::SkipPrefill>(fc_props).get();
  }

  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "Fully connected layer takes only one input";

  std::vector<nntrainer::TensorDim> output_dims(2);

  /// @todo fc actaully supports multidimensions. EffDimFlag shouldn't be fixed
  /// like this.
  context.setEffDimFlagInputDimension(0, 0b1001);
  context.setDynDimFlagInputDimension(0, 0b1000);

  bool is_nchw = (context.getFormat() == nntrainer::Tformat::NCHW);
  /** set output dimensions */
  auto const &in_dim = context.getInputDimensions()[0];
  output_dims[0] = in_dim;
  is_nchw ? output_dims[0].width(unit) : output_dims[0].channel(unit);

  output_dims[1] = in_dim;
  is_nchw ? output_dims[1].width(unit) : output_dims[1].channel(unit);

  output_dims[0].setTensorType(
    {context.getFormat(), context.getActivationDataType()});

  output_dims[1].setTensorType(
    {context.getFormat(), context.getActivationDataType()});

  context.setOutputDimensions(output_dims);

  /** set weight specifications */
  // @todo : This NCHW format setting is just temporal, it needs to be set by
  // global configuration

  /** Bias Dimension : (1, 1, 1, unit) */
  // nntrainer::TensorDim bias_dim(
  //   1, is_nchw ? 1 : unit, 1, is_nchw ? unit : 1,
  //   nntrainer::TensorDim::TensorType(context.getFormat(),
  //                                    context.getWeightDataType()),
  //   is_nchw ? 0b0001 : 0b0100);

  /** Weight Dimension : (1, 1, in_dim.width(), unit)*/
  nntrainer::TensorDim weight_dim(
    1, is_nchw ? 1 : unit, is_nchw ? in_dim.width() : 1,
    is_nchw ? unit : in_dim.channel(),
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()),
    is_nchw ? 0b0011 : 0b0101);

  weight_idx[FCParams::FFN1] = context.requestWeight(
    weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "ffn1", true);

  weight_idx[FCParams::FFN2] = context.requestWeight(
    weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "ffn2", true);

  /** create weights for LoRA */
  if (lora_rank) {

    /** loraA Dimension : (1, 1, in_dim.width, lora_rank) */
    nntrainer::TensorDim loraA_dim(
      1, is_nchw ? 1 : lora_rank, is_nchw ? in_dim.width() : 1,
      is_nchw ? lora_rank : in_dim.channel(),
      nntrainer::TensorDim::TensorType(context.getFormat(),
                                       context.getWeightDataType()),
      is_nchw ? 0b0011 : 0b0101);

    /** loraB Dimension : (1, 1, lora_rank, unit) */
    nntrainer::TensorDim loraB_dim(
      1, is_nchw ? 1 : unit, is_nchw ? lora_rank : 1,
      is_nchw ? unit : lora_rank,
      nntrainer::TensorDim::TensorType(context.getFormat(),
                                       context.getWeightDataType()),
      is_nchw ? 0b0011 : 0b0101);

    /** loraTmp Dimension : (B, 1, in_dim.height(), lora_rank) */
    nntrainer::TensorDim loraTmp_dim(
      in_dim.batch(), is_nchw ? 1 : lora_rank, is_nchw ? in_dim.height() : 1,
      is_nchw ? lora_rank : in_dim.width(),
      nntrainer::TensorDim::TensorType(context.getFormat(),
                                       context.getWeightDataType()),
      is_nchw ? 0b1011 : 0b1101);

    /** loraTmp Dimension : (B, 1, in_dim.height(), unit) */
    nntrainer::TensorDim loraOut_dim(
      in_dim.batch(), is_nchw ? 1 : unit, is_nchw ? in_dim.height() : 1,
      is_nchw ? unit : in_dim.width(),
      nntrainer::TensorDim::TensorType(context.getFormat(),
                                       context.getWeightDataType()),
      is_nchw ? 0b1011 : 0b1101);

    lora_idx[LORAParams::loraA] = context.requestWeight(
      loraA_dim, nntrainer::Initializer::ZEROS, weight_regularizer,
      weight_regularizer_constant, weight_decay, "loraA", true);

    lora_idx[LORAParams::loraB] = context.requestWeight(
      loraB_dim, nntrainer::Initializer::LECUN_NORMAL, weight_regularizer,
      weight_regularizer_constant, weight_decay, "loraB", true);

    lora_idx[LORAParams::loraTmp] = context.requestTensor(
      loraTmp_dim, "hidden_tmp_lora", nntrainer::Initializer::NONE, true,
      nntrainer::TensorLifespan::FORWARD_DERIV_LIFESPAN);

    lora_idx[LORAParams::loraOut] = context.requestTensor(
      loraOut_dim, "hidden_lora", nntrainer::Initializer::NONE, true,
      nntrainer::TensorLifespan::FORWARD_FUNC_LIFESPAN);
  }
}

void FFNLayer::exportTo(nntrainer::Exporter &exporter,
                        const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(fc_props, method, this);
}

void FFNLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, fc_props);
  LayerImpl::setProperty(remain_props);
}

void FFNLayer::setBatch(nntrainer::RunLayerContext &context,
                        unsigned int batch) {
  if (!std::get<nntrainer::props::LoraRank>(fc_props).empty()) {
    // update Lora Tensor's batch info.
    context.updateTensor(lora_idx[LORAParams::loraTmp], batch);
    context.updateTensor(lora_idx[LORAParams::loraOut], batch);
  }
}

void FFNLayer::forwarding(nntrainer::RunLayerContext &context, bool training) {
  return;
}

void FFNLayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                      unsigned int from, unsigned int to,
                                      bool training) {

  if (skip_prefill && !from)
    return;

  nntrainer::Tensor &weight1 = context.getWeight(weight_idx[FCParams::FFN1]);
  nntrainer::Tensor &weight2 = context.getWeight(weight_idx[FCParams::FFN2]);

  nntrainer::Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);

  nntrainer::Tensor &hidden1 = context.getOutput(FCParams::FFN1);
  nntrainer::Tensor &hidden2 = context.getOutput(FCParams::FFN2);

  nntrainer::TensorDim input_dim = input_.getDim();
  nntrainer::TensorDim input_step_dim = input_dim;
  input_step_dim.batch(1);
  input_step_dim.height(to - from);
  // std::cout <<to-from <<" " << input_step_dim.height()<<" "<<
  // input_step_dim.width()<<std::endl; std::cout <<to-from <<" " <<
  // input_dim.height()<<" "<< input_dim.width()<<std::endl; std::cout <<to-from
  // <<" " << Qhidden_.getDim().height()<<" "<<
  // Qhidden_.getDim().width()<<std::endl; std::cout <<to-from <<" " <<
  // Khidden_.getDim().height()<<" "<< Khidden_.getDim().width()<<std::endl;
  // std::cout <<to-from <<" " << Vhidden_.getDim().height()<<" "<<
  // Vhidden_.getDim().width()<<std::endl;

  // auto &pool = nntrainer::ThreadPoolManager::getInstance();
  // auto &pool = nntrainer::ThreadPoolManager::Global().getThreadPool();

  //  std::vector<std::future<void>> futures;

  nntrainer::Tensor input_step =
    input_.getSharedDataTensor(input_step_dim, 0, true);

  // std::cout << "input_step"<<std::endl;

  nntrainer::TensorDim hidden1_dim = hidden1.getDim();
  nntrainer::TensorDim hidden1_step_dim = hidden1.getDim();
  hidden1_step_dim.batch(1);
  hidden1_step_dim.height(to - from);

  nntrainer::Tensor hidden1_step =
    hidden1.getSharedDataTensor(hidden1_step_dim, 0, true);

  nntrainer::TensorDim hidden2_dim = hidden2.getDim();
  nntrainer::TensorDim hidden2_step_dim = hidden2.getDim();
  hidden2_step_dim.batch(1);
  hidden2_step_dim.height(to - from);
  nntrainer::Tensor hidden2_step =
    hidden2.getSharedDataTensor(hidden2_step_dim, 0, true);

  std::vector<nntrainer::Tensor *> Weights({&weight1, &weight2});
  std::vector<nntrainer::Tensor *> Outputs({&hidden1_step, &hidden2_step});

  custom::custom_dot(Outputs, Weights, input_step, from, to);
}

void FFNLayer::calcDerivative(nntrainer::RunLayerContext &context) { return; }

void FFNLayer::calcGradient(nntrainer::RunLayerContext &context) { return; }

void FFNLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  ml::train::TensorDim input_dim = context.getInput(SINGLE_INOUT_IDX).getDim();
  ml::train::TensorDim output1_dim = context.getOutput(FCParams::FFN1).getDim();
  ml::train::TensorDim output2_dim = context.getOutput(FCParams::FFN2).getDim();

  input_dim.height(input_dimensions[0].height());
  output1_dim.height(input_dimensions[0].height());
  output2_dim.height(input_dimensions[0].height());

  context.updateInput(SINGLE_INOUT_IDX, input_dim);
  context.updateOutput(FCParams::FFN1, output1_dim);
  context.updateOutput(FCParams::FFN2, output2_dim);
}

void FFNLayer::read(std::ifstream &file,
                    nntrainer::RunLayerContext &run_context, bool opt_var,
                    ml::train::ExecutionMode mode, bool trainable,
                    nntrainer::TensorDim::DataType definedWeightDataType,
                    bool fsu, size_t start_offset, bool read_from_offset,
                    int file_fd) {

  if (fsu) {
    for (unsigned int i = 0; i < run_context.getNumWeights(); ++i) {
      if (run_context.getWeight(i).getDataType() ==
          nntrainer::TensorDim::DataType::BCQ) {
        run_context.getWeight(i).readFSU();
      }
    }
  } else {
    if (opt_var) {
      for (unsigned int i = 0; i < run_context.getNumWeights(); ++i) {
        if (run_context.isGradientLastAccess(i) && trainable) {
          /// @note read optimizer variables
          for (unsigned int j = 0; j < run_context.getNumWeightOptVar(i); ++j) {
            run_context.getWeightOptVar(i, j).read(file, start_offset);
          }
        }
      }
    } else {
      for (unsigned int i = 0; i < run_context.getNumWeights(); ++i) {
        /// @note shared weights are only be read at the first acecss
        if (run_context.isGradientFirstAccess(i)) {
          run_context.getWeight(i).read(file, start_offset, read_from_offset,
                                        file_fd);
          /// @note This code converts QINT4 data to Q4_0 data when the GPU is
          /// unavailable. This modification is needed until QINT4 CPU
          /// computation is supported.
          if (!run_context.getWeight(i).getMemoryData()->isSVM() &&
              run_context.getWeight(i).getDataType() ==
                nntrainer::TensorDim::DataType::QINT4) {
            int N = run_context.getWeight(i).width();
            int K = run_context.getWeight(i).height();
            int padded_N = ((N + 32 - 1) / 32) * 32;
            std::vector<uint8_t> q4_0x8_data(padded_N * K / 32 *
                                             sizeof(block_q4_0));
            nntrainer::Q4_0Utils::transformQ4_0x_FromInt4(
              padded_N, K, run_context.getWeight(i).getData<uint8_t>(),
              run_context.getWeight(i).getScale<uint16_t>(), 32, 8,
              q4_0x8_data.data());

            memcpy(run_context.getWeight(i).getData<uint8_t>(),
                   q4_0x8_data.data(),
                   run_context.getWeight(i).getMemoryBytes());
          }

          if (run_context.isMixedPrecision(i) && trainable &&
              !run_context.getWeightFP32(i).empty()) {
            run_context.getWeightFP32(i).copyData(run_context.getWeight(i));
          }
        }
      }
    }
  }
}

void FFNLayer::read(nntrainer::ReadSource src,
                    nntrainer::RunLayerContext &run_context, bool opt_var,
                    ml::train::ExecutionMode mode, bool trainable,
                    nntrainer::TensorDim::DataType definedWeightDataType,
                    bool fsu, size_t start_offset, bool read_from_offset) {

  // Only read when mode is embedding
  if (fsu) {
    for (unsigned int i = 0; i < run_context.getNumWeights(); ++i) {
      if (run_context.getWeight(i).getDataType() ==
          nntrainer::TensorDim::DataType::BCQ) {
        run_context.getWeight(i).readFSU();
      }
    }
  } else {
    if (opt_var) {
      for (unsigned int i = 0; i < run_context.getNumWeights(); ++i) {
        if (run_context.isGradientLastAccess(i) && trainable) {
          /// @note read optimizer variables
          for (unsigned int j = 0; j < run_context.getNumWeightOptVar(i); ++j) {
            run_context.getWeightOptVar(i, j).read(src, start_offset);
          }
        }
      }
    } else {
      for (unsigned int i = 0; i < run_context.getNumWeights(); ++i) {
        /// @note shared weights are only be read at the first acecss
        if (run_context.isGradientFirstAccess(i)) {
          run_context.getWeight(i).read(src, start_offset, read_from_offset);

          /// @note This code converts QINT4 data to Q4_0 data when the GPU is
          /// unavailable. This modification is needed until QINT4 CPU
          /// computation is supported.
          if (!run_context.getWeight(i).getMemoryData()->isSVM() &&
              run_context.getWeight(i).getDataType() ==
                nntrainer::TensorDim::DataType::QINT4) {
            int N = run_context.getWeight(i).width();
            int K = run_context.getWeight(i).height();
            int padded_N = ((N + 32 - 1) / 32) * 32;
            std::vector<uint8_t> q4_0x8_data(padded_N * K / 32 *
                                             sizeof(block_q4_0));

            nntrainer::Q4_0Utils::transformQ4_0x_FromInt4(
              padded_N, K, run_context.getWeight(i).getData<uint8_t>(),
              run_context.getWeight(i).getScale<uint16_t>(), 32, 8,
              q4_0x8_data.data());

            memcpy(run_context.getWeight(i).getData<uint8_t>(),
                   q4_0x8_data.data(),
                   run_context.getWeight(i).getMemoryBytes());
          }

          if (run_context.isMixedPrecision(i) && trainable &&
              !run_context.getWeightFP32(i).empty()) {
            run_context.getWeightFP32(i).copyData(run_context.getWeight(i));
          }
        }
      }
    }
  }
}

} // namespace causallm

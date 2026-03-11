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

#include <common_properties.h>
#include <custom_common_properties.h>
#include <custom_dot_wrapper.h>
#include <custom_qkv_layer.h>

#include <bs_thread_pool_manager.hpp>
#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

#include <iostream>

// for 1b GQA 6. 3b GQA 12
#define GQA 12

namespace custom {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum FCParams { Qweight, Kweight, Vweight };
enum LORAParams { loraA, loraB, loraTmp, loraOut };

QKVLayer::QKVLayer() :
  LayerImpl(),
  lora_scaling(1.0f),
  fc_props(nntrainer::props::Unit(), nntrainer::props::LoraRank(),
           props::LoraAlpha(), {}, props::LoraEnable()) {
  weight_idx.fill(std::numeric_limits<unsigned>::max());
  lora_idx.fill(std::numeric_limits<unsigned>::max());
}

void QKVLayer::finalize(nntrainer::InitLayerContext &context) {
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
    (lora_rank && !std::get<custom::props::LoraAlpha>(fc_props).empty())
      ? (float)std::get<custom::props::LoraAlpha>(fc_props) / lora_rank
      : 1;
  std::vector<nntrainer::props::TensorDataType> t_dtype;
  if (!std::get<std::vector<nntrainer::props::TensorDataType>>(fc_props)
         .empty()) {
    t_dtype = std::get<std::vector<nntrainer::props::TensorDataType>>(fc_props);
  }

  const auto &lora_enable =
    (std::get<custom::props::LoraEnable>(fc_props).empty())
      ? false
      : std::get<custom::props::LoraEnable>(fc_props).get();

  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "Fully connected layer takes only one input";

  std::vector<nntrainer::TensorDim> output_dims(3);

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
  is_nchw ? output_dims[0].width(unit) : output_dims[0].channel(unit);

  output_dims[2] = in_dim;
  is_nchw ? output_dims[0].width(unit) : output_dims[0].channel(unit);

  output_dims[0].setTensorType(
    {context.getFormat(), context.getActivationDataType()});

  output_dims[1].setTensorType(
    {context.getFormat(), context.getActivationDataType()});

  output_dims[2].setTensorType(
    {context.getFormat(), context.getActivationDataType()});

  output_dims[1].width(unit / GQA);
  output_dims[2].width(unit / GQA);

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

  weight_idx[FCParams::Qweight] = context.requestWeight(
    weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "qweight", true);

  weight_dim.width(unit / GQA);

  weight_idx[FCParams::Kweight] = context.requestWeight(
    weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "kweight", true);

  weight_idx[FCParams::Vweight] = context.requestWeight(
    weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "vweight", true);

  // if (disable_bias.empty() || disable_bias.get() == false) {
  //   weight_idx[FCParams::bias] = context.requestWeight(
  //     bias_dim, bias_initializer, nntrainer::WeightRegularizer::NONE, 1.0f,
  //     bias_decay, "bias", true);
  // }

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

void QKVLayer::exportTo(nntrainer::Exporter &exporter,
                        const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(fc_props, method, this);
}

void QKVLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, fc_props);
  LayerImpl::setProperty(remain_props);
}

void QKVLayer::setBatch(nntrainer::RunLayerContext &context,
                        unsigned int batch) {
  if (!std::get<nntrainer::props::LoraRank>(fc_props).empty()) {
    // update Lora Tensor's batch info.
    context.updateTensor(lora_idx[LORAParams::loraTmp], batch);
    context.updateTensor(lora_idx[LORAParams::loraOut], batch);
  }
}

void QKVLayer::forwarding(nntrainer::RunLayerContext &context, bool training) {
  return;
}

void QKVLayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                      unsigned int from, unsigned int to,
                                      bool training) {
  nntrainer::Tensor &Qweight = context.getWeight(weight_idx[FCParams::Qweight]);
  nntrainer::Tensor &Kweight = context.getWeight(weight_idx[FCParams::Kweight]);
  nntrainer::Tensor &Vweight = context.getWeight(weight_idx[FCParams::Vweight]);
  nntrainer::Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &Qhidden_ = context.getOutput(FCParams::Qweight);
  nntrainer::Tensor &Khidden_ = context.getOutput(FCParams::Kweight);
  nntrainer::Tensor &Vhidden_ = context.getOutput(FCParams::Vweight);

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

  auto &pool = nntrainer::ThreadPoolManager::Global().getThreadPool();

  //  std::vector<std::future<void>> futures;

  nntrainer::Tensor input_step =
    input_.getSharedDataTensor(input_step_dim, 0, true);

  // std::cout << "input_step"<<std::endl;

  nntrainer::TensorDim Qhidden_dim = Qhidden_.getDim();
  nntrainer::TensorDim Qhidden_step_dim = Qhidden_.getDim();
  Qhidden_step_dim.batch(1);
  Qhidden_step_dim.height(to - from);

  nntrainer::Tensor Qhidden_step =
    Qhidden_.getSharedDataTensor(Qhidden_step_dim, 0, true);

  nntrainer::TensorDim Khidden_dim = Khidden_.getDim();
  nntrainer::TensorDim Khidden_step_dim = Khidden_.getDim();
  Khidden_step_dim.batch(1);
  Khidden_step_dim.height(to - from);
  nntrainer::Tensor Khidden_step =
    Khidden_.getSharedDataTensor(Khidden_step_dim, 0, true);

  nntrainer::TensorDim Vhidden_dim = Vhidden_.getDim();
  nntrainer::TensorDim Vhidden_step_dim = Vhidden_.getDim();
  Vhidden_step_dim.batch(1);
  Vhidden_step_dim.height(to - from);
  nntrainer::Tensor Vhidden_step =
    Vhidden_.getSharedDataTensor(Vhidden_step_dim, 0, true);

  std::vector<nntrainer::Tensor *> Weights({&Qweight, &Kweight, &Vweight});
  std::vector<nntrainer::Tensor *> Outputs(
    {&Qhidden_step, &Khidden_step, &Vhidden_step});

  custom::custom_dot(Outputs, Weights, input_step, from, to);

  // futures.emplace_back(pool.submit_task([=](){
  //   nntrainer::TensorDim Qhidden_dim = Qhidden_.getDim();
  //   nntrainer::TensorDim Qhidden_step_dim = Qhidden_.getDim();
  //   Qhidden_step_dim.batch(1);
  //   Qhidden_step_dim.height(to-from);

  //   nntrainer::Tensor Qhidden_step = Qhidden_.getSharedDataTensor(
  //     Qhidden_step_dim, 0, true);
  //     // std::cout << "Qhidden_step"<<std::endl;
  //     custom::custom_dot(Qhidden_step, Qweight, input_step, from, to);
  // }));

  //  futures.emplace_back(pool.submit_task([=](){
  //   nntrainer::TensorDim Khidden_dim = Khidden_.getDim();
  //   nntrainer::TensorDim Khidden_step_dim = Khidden_.getDim();
  //   Khidden_step_dim.batch(1);
  //   Khidden_step_dim.height(to - from);
  //   nntrainer::Tensor Khidden_step = Khidden_.getSharedDataTensor(
  //     Khidden_step_dim, 0, true);
  //     // std::cout << "Khidden_step"<<std::endl;
  //   custom::custom_dot(Khidden_step, Kweight, input_step, from, to);
  //   //input_step.dot(Kweight, Khidden_step, false, false);
  //  }));

  //  futures.emplace_back(pool.submit_task([=]() {
  //   nntrainer::TensorDim Vhidden_dim = Vhidden_.getDim();
  //   nntrainer::TensorDim Vhidden_step_dim = Vhidden_.getDim();
  //   Vhidden_step_dim.batch(1);
  //   Vhidden_step_dim.height(to - from);
  //   nntrainer::Tensor Vhidden_step = Vhidden_.getSharedDataTensor(
  //     Vhidden_step_dim, 0, true);
  //     // std::cout << "Vhidden_step"<<std::endl;
  //     custom::custom_dot(Vhidden_step, Vweight, input_step,from, to);
  //   //input_step.dot(Vweight, Vhidden_step, false, false);
  // }));

  //  for(auto &fut : futures)
  //   fut.get();
  // std::cout << Qhidden_.getData<float>() << " " <<Khidden_.getData<float>()
  // << " "<<Vhidden_.getData<float>() << std::endl;
}

void QKVLayer::calcDerivative(nntrainer::RunLayerContext &context) { return; }

void QKVLayer::calcGradient(nntrainer::RunLayerContext &context) { return; }

void QKVLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  ml::train::TensorDim input_dim = context.getInput(SINGLE_INOUT_IDX).getDim();
  ml::train::TensorDim Qoutput_dim =
    context.getOutput(FCParams::Qweight).getDim();
  ml::train::TensorDim Koutput_dim =
    context.getOutput(FCParams::Kweight).getDim();
  ml::train::TensorDim Voutput_dim =
    context.getOutput(FCParams::Vweight).getDim();

  input_dim.height(input_dimensions[0].height());
  Qoutput_dim.height(input_dimensions[0].height());
  Koutput_dim.height(input_dimensions[0].height());
  Voutput_dim.height(input_dimensions[0].height());

  context.updateInput(SINGLE_INOUT_IDX, input_dim);
  context.updateOutput(FCParams::Qweight, Qoutput_dim);
  context.updateOutput(FCParams::Kweight, Koutput_dim);
  context.updateOutput(FCParams::Vweight, Voutput_dim);
}
} // namespace custom

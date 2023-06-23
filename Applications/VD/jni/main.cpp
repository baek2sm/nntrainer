// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   main.cpp
 * @date   24 Jun 2021
 * @todo   move resnet model creating to separate sourcefile
 * @brief  task runner for the resnet
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */
#include <array>
#include <chrono>
#include <ctime>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include <layer.h>
#include <model.h>
#include <optimizer.h>

#include <multi_data.h>

using LayerHandle = std::shared_ptr<ml::train::Layer>;
using ModelHandle = std::unique_ptr<ml::train::Model>;

using UserDataType = std::unique_ptr<nntrainer::util::DataLoader>;

/** cache loss values post training for test */
float training_loss = 0.0;
float validation_loss = 0.0;

/**
 * @brief make "key=value" from key and value
 *
 * @tparam T type of a value
 * @param key key
 * @param value value
 * @return std::string with "key=value"
 */
template <typename T>
static std::string withKey(const std::string &key, const T &value) {
  std::stringstream ss;
  ss << key << "=" << value;
  return ss.str();
}

template <typename T>
static std::string withKey(const std::string &key,
                           std::initializer_list<T> value) {
  if (std::empty(value)) {
    throw std::invalid_argument("empty data cannot be converted");
  }

  std::stringstream ss;
  ss << key << "=";

  auto iter = value.begin();
  for (; iter != value.end() - 1; ++iter) {
    ss << *iter << ',';
  }
  ss << *iter;

  return ss.str();
}


ModelHandle createMultiInputModel() {
  using ml::train::createLayer;  
  ModelHandle model = ml::train::createModel(ml::train::ModelType::NEURAL_NET,
                                             {withKey("loss", "mse")});

  std::vector<LayerHandle> layers;

  layers.push_back(createLayer(
    "input", {withKey("name", "input_prev_embeddings"), withKey("input_shape", "5:1:10")}));
  layers.push_back(createLayer(
    "input", {withKey("name", "input_time_sin"), withKey("input_shape", "1:1:1")}));
  layers.push_back(createLayer(
    "input", {withKey("name", "input_time_cos"), withKey("input_shape", "1:1:1")}));
  layers.push_back(createLayer(
    "input", {withKey("name", "input_day_sin"), withKey("input_shape", "1:1:1")}));
  layers.push_back(createLayer(
    "input", {withKey("name", "input_day_cos"), withKey("input_shape", "1:1:1")}));

  layers.push_back(createLayer(
    "conv1d", {
      withKey("name", "conv_embedding1"),
      withKey("filters", 10),
      withKey("kernel_size", 3),
      withKey("activation", "relu"),
      withKey("input_layers", "input_prev_embeddings")}));

  layers.push_back(createLayer(
    "conv1d", {
      withKey("name", "conv_embedding2"),
      withKey("filters", 10),
      withKey("kernel_size", 3),
      withKey("activation", "relu"),
      withKey("input_layers", "conv_embedding1")}));

    layers.push_back(createLayer(
    "conv1d", {
      withKey("name", "conv_embedding3"),
      withKey("filters", 10),
      withKey("kernel_size", 3),
      withKey("activation", "relu"),
      withKey("input_layers", "conv_embedding2")}));

  layers.push_back(createLayer("flatten", {
      withKey("name", "flatten"),
      withKey("input_layers", "conv_embedding3"),
    }));

  layers.push_back(createLayer(
    "concat",
    {withKey("name", "concat0"), withKey("axis", "3"),
     withKey("input_layers", "flatten, input_time_sin, input_time_cos, input_day_sin, input_day_cos")}));

  layers.push_back(
    createLayer("fully_connected", {
        withKey("name", "fully_connected1"),
        withKey("unit", 100),
        withKey("activation", "relu"),
        withKey("input_layers", "concat0")
      }));

  layers.push_back(
    createLayer("dropout", {
      withKey("name", "dropout"),
      withKey("dropout_rate", 0.5),
      withKey("input_layers", "fully_connected1")
    }));

  layers.push_back(
    createLayer("fully_connected", {
        withKey("name", "fully_connected2"),
        withKey("unit", 50),
        withKey("activation", "relu"),
        withKey("input_layers", "dropout")
      }));

  layers.push_back(
    createLayer("fully_connected", {
        withKey("name", "fully_connected3"),
        withKey("unit", 6),
        withKey("activation", "softmax"),
        withKey("input_layers", "fully_connected2")
      }));

  for (auto &layer : layers) {
    model->addLayer(layer);
  }

  return model;
}

int trainData_cb(float **input, float **label, bool *last, void *user_data) {
  auto data = reinterpret_cast<nntrainer::util::DataLoader *>(user_data);

  data->next(input, label, last);
  return 0;
}

/// @todo maybe make num_class also a parameter
void createAndRun(unsigned int epochs, unsigned int batch_size,
                  UserDataType &train_user_data) {

  ModelHandle model = createMultiInputModel();

  model->setProperty({withKey("batch_size", batch_size),
                      withKey("epochs", epochs),
                      withKey("save_path", "vd_model.bin")});

  auto optimizer = ml::train::createOptimizer("adam", {"learning_rate=0.001"});
  model->setOptimizer(std::move(optimizer));

  int status = model->compile();
  if (status) {
    throw std::invalid_argument("model compilation failed!");
  }

  status = model->initialize();
  if (status) {
    throw std::invalid_argument("model initialization failed!");
  }

  // model->load("weights.bin");

  auto dataset_train = ml::train::createDataset(
    ml::train::DatasetType::GENERATOR, trainData_cb, train_user_data.get());

  model->setDataset(ml::train::DatasetModeType::MODE_TRAIN,
                    std::move(dataset_train));

  model->summarize(std::cout, ml_train_summary_type_e::ML_TRAIN_SUMMARY_MODEL);
  
  // std::vector<float *> inputs;
  // std::vector<float *> outputs;
  // float input_sample[55];
  // float output_sample[6];
  // for (int i=0; i<55; i++) {
  //   input_sample[i] = 1;
  // }
  // for (int i=0; i<6; i++) {
  //   output_sample[i] = 1;
  // }
  // inputs.push_back(input_sample);
  // outputs.push_back(output_sample);

  // auto result = model->inference(1, inputs, outputs);
  // for (int i=0; i<6; i++) {
  //   std::cout << result[i] << " ";
  // }

  // weights은 다 옮겨진듯(1 iteration 때 같음.)
  // transpose를 오히려 안해야 제대로 옮겨짐(왜인지 모름)
  // inference 결과는 확인을 못해봄 (inference의 multi-input을 어떻게 넣어줘야할지 모름)
  // training 시에는 dropout 때문에 같은 weights여도 output이 달라서 비교 불가
  model->train();
}

std::array<UserDataType,1>
createFakeMultiDataGenerator(unsigned int batch_size,
                                          unsigned int simulated_data_size) {
  UserDataType train_data(new nntrainer::util::MultiDataLoader(
    {{batch_size, 1, 10, 5},
     {batch_size, 1, 1, 1},
     {batch_size, 1, 1, 1},
     {batch_size, 1, 1, 1},     
     {batch_size, 1, 1, 1}     
    },
    {{batch_size, 1, 1, 6}}, simulated_data_size));

  return {std::move(train_data)};
}

int main(int argc, char *argv[]) {

  unsigned int batch_size = 8;
  unsigned int epoch = 1;
  int status = 0;

  /// warning: the data loader will be destroyed at the end of this function,
  /// and passed as a pointer to the databuffer
  std::array<UserDataType,1> user_datas;
  user_datas = createFakeMultiDataGenerator(batch_size, 512);
  auto &[train_user_data] = user_datas;

  try {
    createAndRun(epoch, batch_size, train_user_data);
  } catch (const std::exception &e) {
    std::cerr << "uncaught error while running! details: " << e.what()
              << std::endl;
    return EXIT_FAILURE;
  }

  return status;
}

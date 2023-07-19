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
#include <cifar_dataloader.h>

#include <app_context.h>
#include <rms_norm.h>
#include <swiglu.h>

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

ModelHandle create_model() {
  using ml::train::createLayer;

  ModelHandle model = ml::train::createModel(ml::train::ModelType::NEURAL_NET,
                                            {withKey("loss", "cross")});

  std::vector<LayerHandle> layers;

  layers.push_back(createLayer(
    "input", {withKey("name", "input0"), withKey("input_shape", "1:1:10")}));

  layers.push_back(createLayer(
    "swiglu", {withKey("name", "swiglu0")}));

  layers.push_back(createLayer(
    "rms_norm", {withKey("name", "rmsnorm0")}));
  
  layers.push_back(
    createLayer("fully_connected",
                {withKey("unit", 10), withKey("activation", "softmax")}));

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

void createAndRun(unsigned int epochs, unsigned int batch_size,
                  UserDataType &train_user_data) {

  // setup model
  ModelHandle model = create_model();
  model->setProperty({withKey("batch_size", batch_size),
                      withKey("epochs", epochs),
                      withKey("save_path", "test_model.bin")});

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

  auto dataset_train = ml::train::createDataset(
    ml::train::DatasetType::GENERATOR, trainData_cb, train_user_data.get());

  model->setDataset(ml::train::DatasetModeType::MODE_TRAIN,
                    std::move(dataset_train));
  
  model->summarize(std::cout, ML_TRAIN_SUMMARY_MODEL);

  model->train();

}

std::array<UserDataType, 1>
createFakeDataGenerator(unsigned int batch_size,
                        unsigned int total_size) {

  UserDataType train_data(new nntrainer::util::RandomDataLoader(
    {{batch_size, 1, 1, 10}}, {{batch_size, 1, 1, 10}}, total_size));

  return {std::move(train_data)};
}

int main(int argc, char *argv[]) {  
  unsigned int total_size = 4;
  unsigned int batch_size = 4;
  unsigned int epoch = 10;

  try {
    auto &app_context = nntrainer::AppContext::Global();
    app_context.registerFactory(nntrainer::createLayer<custom::SwiGLULayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
    return 1;
  }

  try {
    auto &app_context = nntrainer::AppContext::Global();
    app_context.registerFactory(nntrainer::createLayer<custom::RMSNormLayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
    return 1;
  }


  std::array<UserDataType, 1> user_datas;

  try {
    user_datas = createFakeDataGenerator(batch_size, total_size);
  } catch (const std::exception &e) {
    std::cerr << "uncaught error while creating data generator! details: "
              << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  auto &[train_user_data] = user_datas;

  try {
    createAndRun(epoch, batch_size, train_user_data);
  } catch (const std::exception &e) {
    std::cerr << "uncaught error while running! details: " << e.what()
              << std::endl;
    return EXIT_FAILURE;
  }

  int status = EXIT_SUCCESS;
  return status;
}

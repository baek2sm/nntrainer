// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   main.cpp
 * @date   20 May 2025
 * @brief  Different tensor data types per layer test
 * @see    https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 */
#include "network_graph.h"
#include <array>
#include <chrono>
#include <ctime>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include <cstring>
#include <layer.h>
#include <model.h>
#include <optimizer.h>

#include <util_func.h>

using LayerHandle = std::shared_ptr<ml::train::Layer>;
using ModelHandle = std::unique_ptr<ml::train::Model>;

int main(int argc, char **argv) {
  ModelHandle model = ml::train::createModel();
  std::vector<LayerHandle> layers;

  layers.push_back(ml::train::createLayer(
    "input", {nntrainer::withKey("name", "in"),
              nntrainer::withKey("input_shape", "1:1:2")}));

  layers.push_back(ml::train::createLayer(
    "fully_connected",
    {nntrainer::withKey("name", "fc1"), nntrainer::withKey("unit", 3),
     nntrainer::withKey("disable_bias", "true"),
     nntrainer::withKey("tensor_dtype", "FP16"),
     nntrainer::withKey("input_layers", "in")}));

  layers.push_back(ml::train::createLayer(
    "fully_connected", {
                         nntrainer::withKey("name", "fc2"),
                         nntrainer::withKey("unit", 2),
                         nntrainer::withKey("disable_bias", "true"),
                         nntrainer::withKey("input_layers", "fc1"),
                       }));

  for (auto &layer : layers) {
    model->addLayer(layer);
  }

  model->setProperty(
    {nntrainer::withKey("batch_size", 1), nntrainer::withKey("epochs", 1)});
  model->setProperty({nntrainer::withKey("tensor_format", "NCHW")});
  model->compile(nntrainer::ExecutionMode::INFERENCE);
  model->initialize(nntrainer::ExecutionMode::INFERENCE);
  model->load("./model_weights_nntr.bin");

  std::vector<float *> input;
  std::vector<float *> label;

  float init_input[] = {1, 2};
  float *input_sample = (float *)malloc(sizeof(float) * 1);
  memcpy(input_sample, init_input, sizeof(float) * 1);

  input.push_back(input_sample);

  auto output = model->inference(1, input, label);
  std::cout << output[0][0] << ", " << output[0][1] << std::endl;
}
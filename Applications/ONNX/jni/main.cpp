// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   main.cpp
 * @date   26 Feb 2025
 * @brief  onnx example using nntrainer-onnx-api
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.honge@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <iostream>
#include <layer.h>
#include <model.h>
#include <nntrainer-api-common.h>
#include <optimizer.h>
#include <util_func.h>

int main() {
  auto model = ml::train::createModel();

  try {
    std::string path = "/home/seungbaek/pj/nntrainer/Applications/ONNX/jni/"
                       "model_example.onnx";
    model->load(path, ml::train::ModelFormat::MODEL_FORMAT_ONNX);
  } catch (const std::exception &e) {
    std::cerr << "Error during load: " << e.what() << "\n";
    return 1;
  }

  try {
    model->compile();
  } catch (const std::exception &e) {
    std::cerr << "Error during compile: " << e.what() << "\n";
    return 1;
  }

  try {
    model->initialize();
  } catch (const std::exception &e) {
    std::cerr << "Error during initialize: " << e.what() << "\n";
    return 1;
  }

  model->summarize(std::cout, ML_TRAIN_SUMMARY_MODEL);

  std::vector<float *> input;
  std::vector<float *> label;

  float init_input[] = {1, 2, 3, 4, 5, 6, 7, 8};
  float *input_sample1 = (float *)malloc(sizeof(float) * 8);
  // float *input_sample2 = (float *)malloc(sizeof(float) * 6);
  memcpy(input_sample1, init_input, sizeof(float) * 8);
  // memcpy(input_sample2, init_input, sizeof(float) * 6);

  input.push_back(input_sample1);
  // input.push_back(input_sample2);

  // std::cout << "input0 : " << input[0][0] << std::endl;
  // std::cout << "input1 : " << input[1][0] << std::endl;

  // auto output = model->inference(1, input, label);

  // std::cout << output[0][0] << std::endl;
  // std::cout << output[0][1] << std::endl;
  // std::cout << output[0][2] << std::endl;
  // std::cout << output[0][3] << std::endl;

  return 0;
}

// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
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
 * @file  mha_core_test_app.cpp
 * @date  13 March 2026
 * @brief Standalone validation app for self/cross attention in mha_core
 */

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <app_context.h>
#include <engine.h>
#include <layer.h>
#include <model.h>
#include <mha_core.h>
#include <util_func.h>

using LayerHandle = std::shared_ptr<ml::train::Layer>;
using ModelHandle = std::unique_ptr<ml::train::Model>;

namespace {

constexpr unsigned int BATCH_SIZE = 1;
constexpr unsigned int NUM_HEADS = 2;
constexpr unsigned int HEAD_DIM = 2;
constexpr unsigned int HIDDEN_SIZE = NUM_HEADS * HEAD_DIM;
constexpr unsigned int MAX_TIMESTEP = 8;
constexpr float EXPECT_TOLERANCE = 1.0e-4f;

struct AttentionCase {
  const char *name;
  bool is_cross_attention;
  bool is_causal;
  unsigned int query_len;
  unsigned int key_len;
  std::vector<float> query;
  std::vector<float> key;
  std::vector<float> value;
  std::vector<float> expected;
};

const std::vector<AttentionCase> &getAttentionCases() {
  static const std::vector<AttentionCase> cases = {
    {"self_attention", false, true, 2, 2,
     {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f},
     {0.2f, 0.1f, 0.4f, 0.3f, 0.6f, 0.5f, 0.8f, 0.7f},
     {0.9f, 1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f},
     {0.89990234f, 1.0f,        1.09960938f, 1.20019531f,
      1.13977249f, 1.24016302f, 1.36347616f, 1.46341848f}},
    {"cross_attention", true, false, 2, 3,
     {0.15f, 0.05f, 0.25f, 0.35f, 0.45f, 0.55f, 0.65f, 0.75f},
     {0.12f, 0.22f, 0.32f, 0.42f, 0.52f, 0.62f,
      0.72f, 0.82f, 0.92f, 1.02f, 1.12f, 1.22f},
     {1.05f, 1.15f, 1.25f, 1.35f, 1.45f, 1.55f,
      1.65f, 1.75f, 1.85f, 1.95f, 2.05f, 2.15f},
     {1.46507692f, 1.56507695f, 1.69503891f, 1.79503894f,
      1.52443612f, 1.62443614f, 1.75292563f, 1.85292566f}},
  };

  return cases;
}

std::string boolToString(bool value) { return value ? "true" : "false"; }

std::string formatValues(const std::vector<float> &values) {
  std::ostringstream ss;
  ss << std::fixed << std::setprecision(8) << "[";
  for (size_t idx = 0; idx < values.size(); ++idx) {
    if (idx != 0) {
      ss << ", ";
    }
    ss << values[idx];
  }
  ss << "]";
  return ss.str();
}

void registerMhaCoreLayer() {
  static bool registered = false;

  if (registered) {
    return;
  }

  auto *app_context = static_cast<nntrainer::AppContext *>(
    nntrainer::Engine::Global().getRegisteredContext("cpu"));

  if (app_context == nullptr) {
    throw std::runtime_error("failed to get cpu app context");
  }

  try {
    app_context->registerFactory(nntrainer::createLayer<causallm::MHACoreLayer>);
  } catch (const std::invalid_argument &) {
    // registerFactory throws if the layer type is already present.
  }

  registered = true;
}

ModelHandle createAttentionModel(const AttentionCase &test_case) {
  using ml::train::createLayer;
  using nntrainer::withKey;

  registerMhaCoreLayer();

  auto model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);
  std::vector<LayerHandle> layers;

  layers.push_back(createLayer(
    "input", {withKey("name", "query_input"),
              withKey("input_shape",
                      "1:" + std::to_string(test_case.query_len) + ":" +
                        std::to_string(HIDDEN_SIZE))}));
  layers.push_back(createLayer(
    "input", {withKey("name", "key_input"),
              withKey("input_shape",
                      "1:" + std::to_string(test_case.key_len) + ":" +
                        std::to_string(HIDDEN_SIZE))}));
  layers.push_back(createLayer(
    "input", {withKey("name", "value_input"),
              withKey("input_shape",
                      "1:" + std::to_string(test_case.key_len) + ":" +
                        std::to_string(HIDDEN_SIZE))}));

  layers.push_back(createLayer(
    "mha_core",
    {withKey("name", test_case.name), withKey("num_heads", NUM_HEADS),
     withKey("num_heads_kv", NUM_HEADS),
     withKey("max_timestep", MAX_TIMESTEP), withKey("rope_theta", 10000),
     withKey("max_position_embeddings", MAX_TIMESTEP),
     withKey("is_causal", boolToString(test_case.is_causal)),
     withKey("is_cross_attention",
             boolToString(test_case.is_cross_attention)),
     withKey("input_layers", {"query_input", "key_input", "value_input"})}));

  for (auto &layer : layers) {
    if (model->addLayer(layer) != ML_ERROR_NONE) {
      throw std::runtime_error("failed to add layer to the test model");
    }
  }

  model->setProperty({withKey("batch_size", BATCH_SIZE),
                      withKey("epochs", 1),
                      withKey("model_tensor_type", "FP32-FP32")});

  if (model->compile(ml::train::ExecutionMode::INFERENCE) != ML_ERROR_NONE) {
    throw std::runtime_error("failed to compile test model");
  }

  if (model->initialize(ml::train::ExecutionMode::INFERENCE) != ML_ERROR_NONE) {
    throw std::runtime_error("failed to initialize test model");
  }

  return model;
}

std::vector<float> runAttentionCase(const AttentionCase &test_case) {
  auto model = createAttentionModel(test_case);

  std::vector<float> query = test_case.query;
  std::vector<float> key = test_case.key;
  std::vector<float> value = test_case.value;
  std::vector<float *> input = {query.data(), key.data(), value.data()};

  auto output =
    model->incremental_inference(BATCH_SIZE, input, {}, test_case.query_len, 0,
                                 test_case.query_len, true);

  auto output_dim = model->getOutputDimension();
  output_dim[0].batch(BATCH_SIZE);
  size_t output_len = output_dim[0].getDataLen();

  return std::vector<float>(output[0], output[0] + output_len);
}

float maxAbsDiff(const std::vector<float> &lhs, const std::vector<float> &rhs) {
  if (lhs.size() != rhs.size()) {
    throw std::invalid_argument("cannot compare vectors with different sizes");
  }

  float diff = 0.0f;
  for (size_t idx = 0; idx < lhs.size(); ++idx) {
    diff = std::max(diff, std::fabs(lhs[idx] - rhs[idx]));
  }

  return diff;
}

bool validateCase(const AttentionCase &test_case) {
  const auto actual = runAttentionCase(test_case);
  const auto diff = maxAbsDiff(actual, test_case.expected);

  std::cout << "[" << test_case.name << "]" << std::endl;
  std::cout << "  query    : " << formatValues(test_case.query) << std::endl;
  std::cout << "  key      : " << formatValues(test_case.key) << std::endl;
  std::cout << "  value    : " << formatValues(test_case.value) << std::endl;
  std::cout << "  expected : " << formatValues(test_case.expected)
            << std::endl;
  std::cout << "  actual   : " << formatValues(actual) << std::endl;
  std::cout << "  max_diff : " << std::fixed << std::setprecision(8) << diff
            << std::endl;

  if (diff > EXPECT_TOLERANCE) {
    std::cerr << "  validation failed" << std::endl;
    return false;
  }

  std::cout << "  validation passed" << std::endl;
  return true;
}

} // namespace

int main() {
  try {
    bool success = true;

    for (const auto &test_case : getAttentionCases()) {
      success &= validateCase(test_case);
    }

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
  } catch (const std::exception &e) {
    std::cerr << "uncaught error while running mha_core validation: "
              << e.what() << std::endl;
    return EXIT_FAILURE;
  }
}

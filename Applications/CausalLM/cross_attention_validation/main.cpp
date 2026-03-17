// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   main.cpp
 * @date   17 March 2026
 * @brief  Cross-attention validation app for mha_core in CausalLM
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
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
constexpr unsigned int QUERY_LEN = 2;
constexpr unsigned int KEY_LEN = 3;
constexpr unsigned int INPUT_DIM = 4;
constexpr unsigned int NUM_HEADS = 2;
constexpr unsigned int HEAD_DIM = 2;
constexpr unsigned int HIDDEN_DIM = NUM_HEADS * HEAD_DIM;
constexpr unsigned int OUTPUT_DIM = 4;
constexpr unsigned int MAX_TIMESTEP = 8;
constexpr float DEFAULT_TOLERANCE = 1.0e-4f;

struct AppConfig {
  std::string decoder_input_path = "decoder_input.bin";
  std::string encoder_input_path = "encoder_input.bin";
  std::string expected_output_path = "expected_output.bin";
  std::string weight_path = "nntr_weights.bin";
  std::string output_path;
  float tolerance = DEFAULT_TOLERANCE;
  bool use_cross_attention_property = true;
};

void printUsage(const char *argv0) {
  std::cout << "Usage: " << argv0 << " [options]\n"
            << "  --decoder_input <path>   Decoder input binary\n"
            << "  --encoder_input <path>   Encoder input binary\n"
            << "  --expected_output <path> Expected output binary\n"
            << "  --weight <path>          nntrainer weight binary\n"
            << "  --output <path>          Optional output dump path\n"
            << "  --tolerance <float>      Max absolute diff tolerance\n"
            << "  --disable_cross_property Disable is_cross_attention\n"
            << "  --help                   Show this message\n";
}

std::string getArgumentValue(int argc, char **argv, int &index,
                             const std::string &option) {
  if (index + 1 >= argc) {
    throw std::invalid_argument("missing value for option: " + option);
  }
  return argv[++index];
}

AppConfig parseArguments(int argc, char **argv) {
  AppConfig config;

  for (int i = 1; i < argc; ++i) {
    const std::string option = argv[i];

    if (option == "--help") {
      printUsage(argv[0]);
      std::exit(EXIT_SUCCESS);
    }

    if (option == "--decoder_input") {
      config.decoder_input_path = getArgumentValue(argc, argv, i, option);
    } else if (option == "--encoder_input") {
      config.encoder_input_path = getArgumentValue(argc, argv, i, option);
    } else if (option == "--expected_output") {
      config.expected_output_path = getArgumentValue(argc, argv, i, option);
    } else if (option == "--weight") {
      config.weight_path = getArgumentValue(argc, argv, i, option);
    } else if (option == "--output") {
      config.output_path = getArgumentValue(argc, argv, i, option);
    } else if (option == "--tolerance") {
      config.tolerance =
        std::stof(getArgumentValue(argc, argv, i, option));
    } else if (option == "--disable_cross_property") {
      config.use_cross_attention_property = false;
    } else {
      throw std::invalid_argument("unknown option: " + option);
    }
  }

  return config;
}

std::vector<float> readBinary(const std::string &path, size_t expected_size) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    throw std::invalid_argument("failed to open file: " + path);
  }

  const std::streamsize bytes = file.tellg();
  const std::streamsize expected_bytes =
    static_cast<std::streamsize>(expected_size * sizeof(float));
  if (bytes != expected_bytes) {
    throw std::invalid_argument("unexpected file size for " + path +
                                ", expected bytes: " +
                                std::to_string(expected_bytes) +
                                ", got: " + std::to_string(bytes));
  }

  std::vector<float> data(expected_size);
  file.seekg(0, std::ios::beg);
  file.read(reinterpret_cast<char *>(data.data()), expected_bytes);
  if (!file) {
    throw std::invalid_argument("failed to read file: " + path);
  }

  return data;
}

void writeBinary(const std::string &path, const std::vector<float> &values) {
  std::ofstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::invalid_argument("failed to open output file: " + path);
  }

  file.write(reinterpret_cast<const char *>(values.data()),
             static_cast<std::streamsize>(values.size() * sizeof(float)));
  if (!file) {
    throw std::invalid_argument("failed to write output file: " + path);
  }
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
    // ignore if this layer is already registered
  }

  registered = true;
}

ModelHandle createValidationModel(bool use_cross_attention_property) {
  using ml::train::createLayer;
  using nntrainer::withKey;

  registerMhaCoreLayer();

  auto model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);
  std::vector<LayerHandle> layers;

  layers.push_back(createLayer(
    "input", {withKey("name", "decoder_input"),
              withKey("input_shape",
                      "1:" + std::to_string(QUERY_LEN) + ":" +
                        std::to_string(INPUT_DIM))}));
  layers.push_back(createLayer(
    "input", {withKey("name", "encoder_input"),
              withKey("input_shape",
                      "1:" + std::to_string(KEY_LEN) + ":" +
                        std::to_string(INPUT_DIM))}));

  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", "q_proj"), withKey("unit", HIDDEN_DIM),
     withKey("disable_bias", "true"),
     withKey("input_layers", "decoder_input")}));
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", "k_proj"), withKey("unit", HIDDEN_DIM),
     withKey("disable_bias", "true"),
     withKey("input_layers", "encoder_input")}));
  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", "v_proj"), withKey("unit", HIDDEN_DIM),
     withKey("disable_bias", "true"),
     withKey("input_layers", "encoder_input")}));

  std::vector<std::string> mha_props = {
    withKey("name", "cross_attention_core"), withKey("num_heads", NUM_HEADS),
    withKey("num_heads_kv", NUM_HEADS), withKey("max_timestep", MAX_TIMESTEP),
    withKey("max_position_embeddings", MAX_TIMESTEP),
    withKey("rope_theta", 10000), withKey("is_causal", "false"),
    withKey("input_layers", {"q_proj", "k_proj", "v_proj"})};

  if (use_cross_attention_property) {
    mha_props.push_back(withKey("is_cross_attention", "true"));
  }

  layers.push_back(createLayer("mha_core", mha_props));

  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", "out_proj"), withKey("unit", OUTPUT_DIM),
     withKey("disable_bias", "true"),
     withKey("input_layers", "cross_attention_core")}));

  for (auto &layer : layers) {
    if (model->addLayer(layer) != ML_ERROR_NONE) {
      throw std::runtime_error("failed to add a layer in validation model");
    }
  }

  model->setProperty({withKey("batch_size", BATCH_SIZE), withKey("epochs", 1),
                      withKey("model_tensor_type", "FP32-FP32")});

  if (model->compile(ml::train::ExecutionMode::INFERENCE) != ML_ERROR_NONE) {
    throw std::runtime_error("failed to compile validation model");
  }

  if (model->initialize(ml::train::ExecutionMode::INFERENCE) != ML_ERROR_NONE) {
    throw std::runtime_error("failed to initialize validation model");
  }

  return model;
}

std::vector<float> runValidationModel(ModelHandle &model,
                                      const std::vector<float> &decoder_input,
                                      const std::vector<float> &encoder_input) {
  std::vector<float> decoder_copy = decoder_input;
  std::vector<float> encoder_copy = encoder_input;
  std::vector<float *> input = {decoder_copy.data(), encoder_copy.data()};

  auto output =
    model->incremental_inference(BATCH_SIZE, input, {}, QUERY_LEN, 0,
                                 QUERY_LEN, true);

  auto output_dim = model->getOutputDimension();
  output_dim[0].batch(BATCH_SIZE);
  const size_t output_len = output_dim[0].getDataLen();

  return std::vector<float>(output[0], output[0] + output_len);
}

float maxAbsDiff(const std::vector<float> &lhs, const std::vector<float> &rhs) {
  if (lhs.size() != rhs.size()) {
    throw std::invalid_argument("output size mismatch");
  }

  float diff = 0.0f;
  for (size_t i = 0; i < lhs.size(); ++i) {
    diff = std::max(diff, std::fabs(lhs[i] - rhs[i]));
  }

  return diff;
}

void printValues(const std::string &name, const std::vector<float> &values) {
  std::cout << name << ": [";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      std::cout << ", ";
    }
    std::cout << std::fixed << std::setprecision(8) << values[i];
  }
  std::cout << "]\n";
}

} // namespace

int main(int argc, char **argv) {
  try {
    const AppConfig config = parseArguments(argc, argv);

    const auto decoder_input =
      readBinary(config.decoder_input_path, BATCH_SIZE * QUERY_LEN * INPUT_DIM);
    const auto encoder_input =
      readBinary(config.encoder_input_path, BATCH_SIZE * KEY_LEN * INPUT_DIM);
    const auto expected_output =
      readBinary(config.expected_output_path,
                 BATCH_SIZE * QUERY_LEN * OUTPUT_DIM);

    auto model = createValidationModel(config.use_cross_attention_property);
    model->load(config.weight_path, ml::train::ModelFormat::MODEL_FORMAT_BIN);

    const auto actual_output =
      runValidationModel(model, decoder_input, encoder_input);
    const float diff = maxAbsDiff(actual_output, expected_output);

    printValues("expected", expected_output);
    printValues("actual  ", actual_output);
    std::cout << "max_abs_diff: " << std::fixed << std::setprecision(8)
              << diff << "\n";

    if (!config.output_path.empty()) {
      writeBinary(config.output_path, actual_output);
      std::cout << "saved model output to " << config.output_path << "\n";
    }

    if (diff > config.tolerance) {
      std::cerr << "validation failed: diff(" << diff << ") > tolerance("
                << config.tolerance << ")\n";
      return EXIT_FAILURE;
    }

    std::cout << "validation passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &e) {
    std::cerr << "cross_attention_validation error: " << e.what() << "\n";
    std::cerr << "hint: cross-attention mode requires mha_core with "
              << "is_cross_attention support.\n";
    return EXIT_FAILURE;
  }
}

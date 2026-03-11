// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file embedding.cpp
 * @date 02 Jan 2026
 * @see https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug No known bugs except for NYI items
 * @brief This file defines Embedding's basic actions
 */

#include <app_context.h>
#include <embedding.h>
#include <embedding_normalize_layer.h>
#include <embedding_pooling_layer.h>
#include <engine.h>

#include <algorithm>
#include <filesystem>
#include <iostream>

namespace causallm {

Embedding::Embedding(json &cfg, json &generation_cfg, json &nntr_cfg) :
  Transformer(cfg, generation_cfg, nntr_cfg, ModelType::EMBEDDING) {
  setupParameters(cfg, generation_cfg, nntr_cfg);
}

std::map<std::string, std::string> Embedding::layer_map = {
  {"Pooling", "embedding_pooling"},
  {"Normalize", "embedding_normalize"},
  {"Dense", "fully_connected"}};

void Embedding::setupParameters(json &cfg, json &generation_cfg,
                                json &nntr_cfg) {
  Transformer::setupParameters(cfg, generation_cfg, nntr_cfg);
  modules.clear();
  module_configs.clear();

  std::filesystem::path modules_config_path = "modules.json";
  if (nntr_cfg.contains("module_config_path")) {
    modules_config_path =
      std::filesystem::path(nntr_cfg["module_config_path"].get<std::string>());
  } else {
    std::cout << "module_config_path is not set. Using default: "
              << modules_config_path << std::endl;
  }

  // Resolve modules.json from tokenizer_file directory if relative path is used
  // and cwd does not contain the file (common in Android packaging layouts).
  std::filesystem::path modules_json_path = modules_config_path;
  if (nntr_cfg.contains("tokenizer_file")) {
    std::filesystem::path tokenizer_path =
      std::filesystem::path(nntr_cfg["tokenizer_file"].get<std::string>());
    std::filesystem::path tokenizer_dir = tokenizer_path.parent_path();

    if (!modules_json_path.is_absolute()) {
      std::filesystem::path candidate_from_tokenizer =
        tokenizer_dir / modules_json_path;
      if (std::filesystem::exists(candidate_from_tokenizer)) {
        modules_json_path = candidate_from_tokenizer;
      } else if (std::filesystem::exists(tokenizer_dir / "modules.json")) {
        modules_json_path = tokenizer_dir / "modules.json";
      }
    } else if (!std::filesystem::exists(modules_json_path) &&
               std::filesystem::exists(tokenizer_dir / "modules.json")) {
      modules_json_path = tokenizer_dir / "modules.json";
    }
  }

  // Get the directory containing modules.json to resolve relative module paths.
  std::filesystem::path base_dir = modules_json_path.parent_path();

  try {
    // 1. Load modules.json to get the structure and order of layers
    json modules_json = LoadJsonFile(modules_json_path.string());
    modules = modules_json.get<std::vector<json>>();

    for (auto &module : modules) {
      if (module.contains("path")) {
        std::string module_path_str = module["path"].get<std::string>();
        if (module_path_str.empty()) {
          // For the first module (Transformer), the path might be empty or ".""
          // We generally skip it or handle it if it points to a separate
          // config.
          continue;
        }

        // 2. Resolve config.json path for each module
        std::filesystem::path module_dir = base_dir / module_path_str;

        if (std::filesystem::exists(module_dir) &&
            std::filesystem::is_directory(module_dir)) {
          std::filesystem::path config_path = module_dir / "config.json";
          if (std::filesystem::exists(config_path)) {
            try {
              // 3. Load config.json and store it in module_configs map using
              // idx as key
              json module_config = LoadJsonFile(config_path.string());
              if (module.contains("idx")) {
                int idx = module["idx"].get<int>();
                module_configs[idx] = module_config;
              } else {
                std::cerr << "Warning: Module does not have idx field"
                          << std::endl;
              }
            } catch (const std::exception &e) {
              std::cerr << "Failed to load config for module: "
                        << module_path_str << " Reason: " << e.what()
                        << std::endl;
            }
          } else {
            // It's possible some modules don't have a config.json
          }
        }
      }
    }
  } catch (const std::exception &e) {
    std::cerr << "Failed to load modules config from: "
              << modules_json_path.string()
              << " Reason: " << e.what() << std::endl;
    modules.clear();
  }
}

void Embedding::constructModel() {
  bool transformer_constructed = false;

  if (modules.empty()) {
    std::cerr << "Warning: modules metadata is empty. Falling back to "
                 "Transformer-only embedding graph."
              << std::endl;
    Transformer::constructModel();
    return;
  }

  for (auto &module : modules) {
    if (!module.contains("type")) {
      continue;
    }
    std::string type = module["type"].get<std::string>();
    std::string component = getLastComponent(type);

    if (component == "Transformer") {
      if (!transformer_constructed) {
        Transformer::constructModel();
        transformer_constructed = true;
      } else {
        std::cerr << "Warning: duplicate Transformer module detected. "
                     "Ignoring duplicate entry."
                  << std::endl;
      }
    } else {
      if (!transformer_constructed) {
        std::cerr << "Warning: module order is missing a leading Transformer. "
                     "Constructing Transformer graph before custom modules."
                  << std::endl;
        Transformer::constructModel();
        transformer_constructed = true;
      }
      if (module.contains("idx")) {
        int idx = module["idx"].get<int>();
        // Add module layer using properties from loaded config
        addModule(type, idx);
      } else {
        std::cerr << "Warning: Module does not have idx field, skipping: "
                  << type << std::endl;
      }
    }
  }

  if (!transformer_constructed) {
    std::cerr << "Warning: no Transformer module found in modules.json. "
                 "Falling back to Transformer-only embedding graph."
              << std::endl;
    Transformer::constructModel();
  }
}

void Embedding::addModule(const std::string &type, int idx) {
  json config;
  if (module_configs.find(idx) != module_configs.end()) {
    config = module_configs[idx];
  } else {
    // Config might be empty if no config.json was found.
    // This is valid for layers that don't satisfy specific configurations
    // (e.g., default behavior)
  }

  // Determine the layer type component (e.g., "Pooling" from
  // "sentence_transformers.models.Pooling")
  std::string component = getLastComponent(type);
  std::string layer_name;
  auto it = layer_map.find(component);
  if (it != layer_map.end()) {
    layer_name = it->second;
  }

  if (layer_name.empty()) {
    std::cerr << "Warning: No layer mapping found for module type: " << type
              << " (component: " << component << "). Skipping." << std::endl;
    return;
  }

  // Convert JSON config to nntrainer property format (key=value strings)
  std::vector<std::string> props;
  for (auto &el : config.items()) {
    std::string val_str;
    if (el.value().is_string())
      val_str = el.value().get<std::string>();
    else
      val_str = el.value().dump(); // convert to string

    if (el.key() == "out_features") {
      props.push_back("unit=" + val_str);
    } else if (el.key() == "bias") {
      if (val_str == "false") {
        props.push_back("disable_bias=true");
      }
    } else if (el.key() == "activation_function") {
      if (val_str.find("Identity") == std::string::npos) {
        props.push_back("activation=" + val_str);
      } else {
        // need to support other activations later on
      }
    } else if (el.key() == "in_features") {
      // Ignore in_features as nntrainer infers it
    } else {
      props.push_back(el.key() + "=" + val_str);
    }
  }

  LayerHandle layer = ml::train::createLayer(layer_name, props);
  model->addLayer(layer);
}

void Embedding::run(const WSTR prompt, bool do_sample, const WSTR system_prompt,
                    const WSTR tail_prompt) {

  try {
    std::vector<float *> results = encode(prompt, system_prompt, tail_prompt);
    if (results.empty() || results[0] == nullptr) {
      throw std::runtime_error("Embedding inference returned no output.");
    }

    unsigned int output_dim = DIM;
    auto out_dims = model->getOutputDimension();
    if (!out_dims.empty() && out_dims[0].width() > 0) {
      output_dim = out_dims[0].width();
    }

    std::cout << "Embedding Result (" << BATCH_SIZE
              << " batch(es)):" << std::endl;
    for (unsigned int b = 0; b < BATCH_SIZE; ++b) {
      std::cout << "Batch " << b << ": [";
      // Print first few elements as sample
      unsigned int print_dim = std::min<unsigned int>(output_dim, 10);
      for (unsigned int i = 0; i < print_dim; ++i) {
        std::cout << results[0][b * output_dim + i]
                  << (i == print_dim - 1 ? "" : ", ");
      }
      if (output_dim > 10)
        std::cout << ", ...";
      std::cout << "] (Total DIM: " << output_dim << ")" << std::endl;
    }
  } catch (const std::exception &e) {
    std::cerr << "Error during embedding run: " << e.what() << std::endl;
  }
}

std::vector<float *> Embedding::encode(const WSTR prompt,
                                       const WSTR system_prompt,
                                       const WSTR tail_prompt) {
  if (!is_initialized) {
    throw std::runtime_error("Embedding model is not initialized. Please call "
                             "initialize() before encode().");
  }

#if defined(_WIN32)
  std::wstring prompt_ = system_prompt + prompt + tail_prompt;
  std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
  auto _input = tokenizer->Encode(converter.to_bytes(prompt_), true);
#else
  std::string prompt_ = system_prompt + prompt + tail_prompt;
  auto _input = tokenizer->Encode(prompt_, true);
#endif

  auto input_dims = model->getInputDimension();
  if (input_dims.empty()) {
    throw std::runtime_error("Embedding model has no input dimensions.");
  }

  const unsigned int expected_input_len = input_dims[0].getFeatureLen();
  if (expected_input_len == 0) {
    throw std::runtime_error("Embedding model input dimension is invalid.");
  }

  std::vector<int64_t> init_input;
  unsigned int input_len =
    std::min((unsigned int)_input.size(), expected_input_len);
  if (input_len == 0) {
    throw std::runtime_error("Tokenizer returned an empty token sequence.");
  }

  // feed only available length
  for (unsigned int i = 0; i < input_len; ++i)
    init_input.push_back(_input[i]);

  std::vector<float> input_sample(BATCH_SIZE * expected_input_len, 0.0f);

  for (unsigned int b = 0; b < BATCH_SIZE; ++b) {
    for (unsigned int i = 0; i < input_len; ++i) {
      input_sample[static_cast<size_t>(b) * expected_input_len + i] =
        static_cast<float>(init_input[i]);
    }
  }

  std::vector<float *> input;
  input.push_back(input_sample.data());

  std::vector<float *> label; // Empty label for inference

  // Run incremental inference for the prefill stage
  // start: 0, end: input_len (process all tokens at once)
  // This performs a single forward pass for the entire prompt sequence to get
  // embeddings.
  std::vector<float *> output = model->incremental_inference(
    BATCH_SIZE, input, label, input_len, 0, input_len, false);

  return output;
}

std::string Embedding::getLastComponent(const std::string &type) {
  std::string last_component = type;
  size_t last_dot_pos = type.find_last_of('.');
  if (last_dot_pos != std::string::npos) {
    last_component = type.substr(last_dot_pos + 1);
  }
  return last_component;
}

void Embedding::registerCustomLayers() {
  Transformer::registerCustomLayers();

  const auto &ct_engine = nntrainer::Engine::Global();
  const auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));

  try {
    app_context->registerFactory(
      nntrainer::createLayer<causallm::EmbeddingPoolingLayer>);
    app_context->registerFactory(
      nntrainer::createLayer<causallm::EmbeddingNormalizeLayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
  }
}

} // namespace causallm

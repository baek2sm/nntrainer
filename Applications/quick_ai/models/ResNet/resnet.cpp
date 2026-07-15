// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   resnet.cpp
 * @date   14 July 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @brief  General ResNet vision backbone (including IR & Mona) model
 * implementation.
 */

#include "resnet.h"
#include "prelu_layer.h"
#include "resnet_graph.h"
#include <app_context.h>
#include <chrono>
#include <cmath>
#include <engine.h>
#include <iostream>
#include <mutex>

namespace quick_ai {

static std::string withKey(const std::string &key, unsigned int val) {
  return key + "=" + std::to_string(val);
}

static std::string withKey(const std::string &key, const std::string &val) {
  return key + "=" + val;
}

ResNet::ResNet(json &cfg, json &generation_cfg, json &nntr_cfg) : Model() {
  (void)cfg;
  (void)generation_cfg;
  setupParameters(cfg, generation_cfg, nntr_cfg);
}

void ResNet::setupParameters(json &cfg, json &generation_cfg, json &nntr_cfg) {
  (void)cfg;
  (void)generation_cfg;

  BATCH_SIZE = nntr_cfg.value("batch_size", 1);
  MODEL_TENSOR_TYPE = nntr_cfg.value("model_tensor_type", "FP32-FP32");
  CONV_DTYPE_STR = nntr_cfg.value("conv_dtype", "FP32");
  IMGSZ = nntr_cfg.value("imgsz", 112);

  // Read block depths and widths if dynamically supplied
  if (nntr_cfg.contains("num_blocks")) {
    BLOCK_DEPTHS = nntr_cfg["num_blocks"].get<std::vector<int>>();
  }
  if (nntr_cfg.contains("channels")) {
    BLOCK_WIDTHS = nntr_cfg["channels"].get<std::vector<int>>();
  }

  MONA_VERIFY = nntr_cfg.value("yolo_verify", false);
  MONA_DUMP_RAW = nntr_cfg.value("yolo_dump_raw", false);
  MONA_BENCH_ITERS = nntr_cfg.value("yolo_bench_iters", 1);
  MONA_REF_DIR = nntr_cfg.value("yolo_ref_dir", ".");
  USE_MONA_VAL = nntr_cfg.value("use_mona", 0.0f); // 0.0=Human, 1.0=Pet
}

void ResNet::registerCustomLayers() {
  static std::once_flag registered;
  std::call_once(registered, []() {
    const auto &ct_engine = nntrainer::Engine::Global();
    auto app_context = static_cast<nntrainer::AppContext *>(
      ct_engine.getRegisteredContext("cpu"));
    app_context->registerFactory(nntrainer::createLayer<custom::PReLULayer>);
  });
}

void ResNet::initialize() {
  registerCustomLayers();

  model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);

  std::vector<std::string> model_props = {
    withKey("batch_size", BATCH_SIZE), withKey("epochs", "1"),
    withKey("model_tensor_type", MODEL_TENSOR_TYPE)};
  model->setProperty(model_props);

  resnet::ResNetConfig graph_cfg;
  graph_cfg.d = BLOCK_DEPTHS;
  graph_cfg.w = BLOCK_WIDTHS;

  bool conv_q40 = (CONV_DTYPE_STR == "Q4_0");
  bool is_mixed = (MODEL_TENSOR_TYPE == "FP32-FP16");

  auto [inputs, out_feat] =
    resnet::constructResNetGraph(graph_cfg, IMGSZ, conv_q40, is_mixed);

  std::vector<ml::train::Tensor> outputs = {out_feat};
  if (model->compile(inputs, outputs, ml::train::ExecutionMode::INFERENCE)) {
    throw std::runtime_error("ResNet model compilation failed.");
  }

  is_initialized = true;
}

void ResNet::run(const WSTR prompt, bool do_sample, const WSTR system_prompt,
                 const WSTR tail_prompt, bool log_output) {
  (void)do_sample;
  (void)system_prompt;
  (void)tail_prompt;
  (void)log_output;

  if (!is_initialized) {
    throw std::runtime_error(
      "ResNet is not initialized. Please call initialize() first.");
  }

  // 1) Load Input Tensor [1, 3, IMGSZ, IMGSZ]
  std::string input_path(prompt);
  std::ifstream f(input_path, std::ios::binary);
  if (!f.is_open()) {
    throw std::runtime_error("Failed to open ResNet input file: " + input_path);
  }

  const size_t expect_n = 3 * IMGSZ * IMGSZ;
  std::vector<float> input_buf(expect_n);
  f.read(reinterpret_cast<char *>(input_buf.data()), expect_n * sizeof(float));
  size_t n_bytes = f.gcount();
  if (n_bytes != expect_n * sizeof(float)) {
    throw std::runtime_error("ResNet input size mismatch: got " +
                             std::to_string(n_bytes) + " bytes, expected " +
                             std::to_string(expect_n * sizeof(float)));
  }

  // 2) Load use_mona tensor [1, 1, 1, 1]
  std::vector<float *> inputs = {input_buf.data(), &USE_MONA_VAL};
  std::vector<float *> outs;

  // 3) Execute Benchmark Run
  double total_ms = 0.0;
  for (unsigned int iter = 0; iter < MONA_BENCH_ITERS; ++iter) {
    auto t0 = std::chrono::steady_clock::now();
    outs = model->inference(BATCH_SIZE, inputs, {});
    auto t1 = std::chrono::steady_clock::now();
    total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
  }

  std::cout << "Inference done (256-dim feature output)." << std::endl;
  std::cout << "Inference time: " << (total_ms / MONA_BENCH_ITERS)
            << " ms (avg over " << MONA_BENCH_ITERS << " iters)" << std::endl;

  // 4) L2 Normalization and Output Print
  float *out_raw = outs[0]; // 256 floats
  float norm_sum = 0.0f;
  for (int i = 0; i < 256; ++i) {
    norm_sum += out_raw[i] * out_raw[i];
  }
  float norm = std::sqrt(norm_sum);
  if (norm > 1e-8f) {
    for (int i = 0; i < 256; ++i) {
      out_raw[i] /= norm;
    }
  }

  // Print first 10 elements of normalised features
  std::cout << "\nNormalised Feature Vector (First 10 dimensions):"
            << std::endl;
  std::cout << " [";
  for (int i = 0; i < 10; ++i) {
    std::cout << out_raw[i] << (i < 9 ? ", " : "");
  }
  std::cout << "...]" << std::endl;

  // 5) Verify and Dump Raw
  if (MONA_VERIFY) {
    verifyAgainstPyTorch(outs);
  }
}

void ResNet::verifyAgainstPyTorch(const std::vector<float *> &outs) {
  float *out_raw = outs[0]; // 256 elements

  // Normalize
  float norm_sum = 0.0f;
  for (int i = 0; i < 256; ++i) {
    norm_sum += out_raw[i] * out_raw[i];
  }
  float norm = std::sqrt(norm_sum);
  std::vector<float> norm_feat(256);
  if (norm > 1e-8f) {
    for (int i = 0; i < 256; ++i) {
      norm_feat[i] = out_raw[i] / norm;
    }
  } else {
    std::copy(out_raw, out_raw + 256, norm_feat.begin());
  }

  if (MONA_DUMP_RAW) {
    std::string dump_path = MONA_REF_DIR + "/dump_decoded.bin";
    std::ofstream ofs(dump_path, std::ios::binary);
    if (ofs) {
      ofs.write(reinterpret_cast<const char *>(norm_feat.data()),
                256 * sizeof(float));
      std::cout << "  [dump] " << dump_path << " (256 floats)" << std::endl;
    } else {
      std::cout << "  [dump] FAILED to write " << dump_path << std::endl;
    }
  }
}

} // namespace quick_ai

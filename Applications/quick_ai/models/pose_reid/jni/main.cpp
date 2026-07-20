// SPDX-License-Identifier: Apache-2.0
/**
 * @file   main.cpp
 * @brief  YOLOv7ReIDtiny inference and verification driver
 */
#include "yoloreid_graph.h"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <layer.h>
#include <model.h>
#include <numeric>
#include <tensor_api.h>
#include <tensor_dim.h>
#include <vector>

// Helper to load binary files
std::vector<float> loadBin(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f.is_open()) {
    throw std::runtime_error("Cannot open: " + path);
  }
  f.seekg(0, std::ios::end);
  size_t n = f.tellg() / sizeof(float);
  f.seekg(0);
  std::vector<float> v(n);
  f.read(reinterpret_cast<char *>(v.data()), n * sizeof(float));
  return v;
}

int main(int argc, char **argv) {
  std::cout << "[YOLOv7ReIDtiny] imgsz=320 nkpt=87 embed_dim=128" << std::endl;

  std::string RES_DIR = ".";
  if (argc > 1) {
    RES_DIR = argv[1];
  }

  std::string weights_path = RES_DIR + "/yoloreid.safetensors";
  std::string input_path = RES_DIR + "/input.bin";

  try {
    // Create CCAPI Model
    auto model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);

    // Input symbolic tensor
    auto x = ml::train::Tensor(
      ml::train::TensorDim(1, 3, 320, 320, ml::train::TensorDim::Format::NCHW,
                           ml::train::TensorDim::DataType::FP32),
      "input0");

    // Build Backbone FPN and Heads
    auto backbone_feats = yoloreid::buildBackbone(x);

    // Pose neck + head (nkpt = 87)
    auto pose_fpn =
      yoloreid::buildFeatureFPN("model.0.features", backbone_feats);
    auto pose_head = yoloreid::buildRTMCCHead("model.1", pose_fpn, 87);

    // ReID neck + head (embed_dim = 128)
    auto reid_fpn =
      yoloreid::buildFeatureFPN("model.0.features_feat", backbone_feats);
    auto reid_head = yoloreid::buildReIDHead("model.2", reid_fpn, 128);

    std::vector<ml::train::Tensor> outputs = {pose_head, reid_head};

    // Compile Model
    if (int ret = model->compile(x, outputs)) {
      throw std::runtime_error("compile failed: " + std::to_string(ret));
    }

    // Initialize Model
    if (int ret = model->initialize()) {
      throw std::runtime_error("initialize failed: " + std::to_string(ret));
    }

    // Load Weights
    model->load(weights_path, ml::train::ModelFormat::MODEL_FORMAT_SAFETENSORS);
    std::cout << "Model built and weights loaded (" << weights_path << ")."
              << std::endl;

    // Load input
    std::vector<float> input;
    try {
      input = loadBin(input_path);
      std::cout << "Input loaded successfully from: " << input_path
                << std::endl;
    } catch (...) {
      std::cout << "Input bin not found, generating dummy input..."
                << std::endl;
      input.resize(1 * 3 * 320 * 320, 0.5f);
    }
    std::vector<float *> in_ptr = {input.data()};

    // Inference
    auto outs = model->inference(1, in_ptr, std::vector<float *>());
    std::cout << "Inference done (" << outs.size() << " outputs)." << std::endl;

    const float *pose_out =
      outs[0]; // [1, 1280] (which is cls_x and cls_y concatenated)
    const float *reid_out = outs[1]; // [1, 128]

    // Print out the first few values of pose and ReID to confirm execution
    std::cout << "\n=== Pose Head Output (First 10 values) ===" << std::endl;
    for (int i = 0; i < 10; ++i) {
      std::cout << "pose[" << i << "] = " << pose_out[i] << std::endl;
    }

    std::cout << "\n=== ReID Head Output (First 10 values) ===" << std::endl;
    for (int i = 0; i < 10; ++i) {
      std::cout << "reid[" << i << "] = " << reid_out[i] << std::endl;
    }

    std::cout << "\nYOLOv7ReIDtiny executed successfully!" << std::endl;
    return 0;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}

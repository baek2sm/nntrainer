// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   verify_cross_attention.cpp
 * @brief  Standalone test application for cross-attention verification.
 *         Builds a minimal nntrainer model:
 *           query_input -> q_proj FC \
 *           kv_input    -> k_proj FC  -> mha_core (cross-attn) -> o_proj FC
 *                       -> v_proj FC /
 *         Loads weights and reference I/O from Python script, runs inference,
 *         and prints outputs for comparison.
 *
 * Usage:
 *   ./verify_cross_attention <weight_file> <io_file>
 *
 * Prerequisites:
 *   1. Run verify_cross_attention.py to generate weight and I/O files
 *   2. Build with: add this to CausalLM/meson.build as a test executable
 *
 * @note Uses model_tensor_type=FP32-FP32
 */

#include <cstdint>
#include <cstring>
#include <engine.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <app_context.h>
#include <model.h>

// Custom layer headers
#include <factory.h>
#include <layers/mha_core.h>

// ============================================================
// Configuration (must match Python script)
// ============================================================
static constexpr unsigned int BATCH_SIZE = 1;
static constexpr unsigned int D_MODEL = 64;
static constexpr unsigned int NUM_HEADS_Q = 4;
static constexpr unsigned int NUM_HEADS_KV = 2;
static constexpr unsigned int HEAD_DIM = 16;
static constexpr unsigned int Q_SEQ_LEN = 3;
static constexpr unsigned int KV_SEQ_LEN = 5;
static constexpr unsigned int D_Q = NUM_HEADS_Q * HEAD_DIM;   // 64
static constexpr unsigned int D_KV = NUM_HEADS_KV * HEAD_DIM; // 32

// ============================================================
// Helper: create layer with properties
// ============================================================
using LayerHandle = std::shared_ptr<ml::train::Layer>;

template <typename T>
static std::string withKey(const std::string &key, const T &value) {
  return key + "=" + std::to_string(value);
}

static std::string withKey(const std::string &key, const std::string &value) {
  return key + "=" + value;
}

static std::string withKey(const std::string &key, const char *value) {
  return key + "=" + std::string(value);
}

static LayerHandle createLayer(const std::string &type,
                               const std::vector<std::string> &props) {
  auto layer = ml::train::createLayer(type, props);
  return layer;
}

// ============================================================
// Read I/O file header and data
// ============================================================
struct TestIO {
  std::vector<float> query_input;
  std::vector<float> kv_input;
  std::vector<float> ref_output;
};

static TestIO loadTestIO(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f.is_open()) {
    throw std::runtime_error("Cannot open I/O file: " + path);
  }

  // Read header (9 uint32 values)
  uint32_t header[9];
  f.read(reinterpret_cast<char *>(header), sizeof(header));

  uint32_t batch = header[0], q_len = header[1], kv_len = header[2];
  uint32_t d_model = header[3], d_q = header[4], d_kv = header[5];

  std::cout << "I/O header: batch=" << batch << " q_len=" << q_len
            << " kv_len=" << kv_len << " d_model=" << d_model << " d_q=" << d_q
            << " d_kv=" << d_kv << std::endl;

  TestIO io;
  auto readVec = [&](size_t count) {
    std::vector<float> v(count);
    f.read(reinterpret_cast<char *>(v.data()), count * sizeof(float));
    return v;
  };

  io.query_input = readVec(batch * q_len * d_model);
  io.kv_input = readVec(batch * kv_len * d_model);
  io.ref_output = readVec(batch * q_len * d_model);

  return io;
}

// ============================================================
// Print helper
// ============================================================
static void printTensor(const std::string &name, const float *data,
                        unsigned int rows, unsigned int cols) {
  std::cout << "\n--- " << name << " (" << rows << " x " << cols << ") ---"
            << std::endl;
  for (unsigned int r = 0; r < rows; ++r) {
    std::cout << "  Row " << r << ": [";
    for (unsigned int c = 0; c < cols; ++c) {
      if (c > 0)
        std::cout << ", ";
      std::cout << std::setw(12) << std::setprecision(6) << std::scientific
                << data[r * cols + c];
    }
    std::cout << "]" << std::endl;
  }
}

// ============================================================
// Main
// ============================================================
int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <weight_file> <io_file>"
              << std::endl;
    std::cerr << "  Generate files with: python3 verify_cross_attention.py"
              << std::endl;
    return 1;
  }

  const std::string weight_path = argv[1];
  const std::string io_path = argv[2];

  std::cout << "============================================================"
            << std::endl;
  std::cout << "Cross-Attention Verification - nntrainer" << std::endl;
  std::cout << "============================================================"
            << std::endl;

  // ============================================================
  // 1. Register custom layers
  // ============================================================
  const auto &ct_engine = nntrainer::Engine::Global();
  const auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));
  app_context->registerFactory(nntrainer::createLayer<causallm::MHACoreLayer>);
  // ============================================================
  // 2. Build the model
  // ============================================================
  auto model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);

  model->setProperty({
    "batch_size=1",
    "model_tensor_type=FP32-FP32",
  });

  // Input layers
  model->addLayer(createLayer(
    "input", {
               withKey("name", "query_input"),
               withKey("input_shape", "1:1:" + std::to_string(Q_SEQ_LEN) + ":" +
                                        std::to_string(D_MODEL)),
             }));
  model->addLayer(createLayer(
    "input", {
               withKey("name", "kv_input"),
               withKey("input_shape", "1:1:" + std::to_string(KV_SEQ_LEN) +
                                        ":" + std::to_string(D_MODEL)),
             }));

  // Q projection
  model->addLayer(
    createLayer("fully_connected", {
                                     withKey("name", "q_proj"),
                                     withKey("unit", D_Q),
                                     withKey("input_layers", "query_input"),
                                     withKey("disable_bias", "true"),
                                     withKey("weight_initializer", "ones"),
                                   }));

  // K projection
  model->addLayer(
    createLayer("fully_connected", {
                                     withKey("name", "k_proj"),
                                     withKey("unit", D_KV),
                                     withKey("input_layers", "kv_input"),
                                     withKey("disable_bias", "true"),
                                     withKey("weight_initializer", "ones"),
                                   }));

  // V projection
  model->addLayer(
    createLayer("fully_connected", {
                                     withKey("name", "v_proj"),
                                     withKey("unit", D_KV),
                                     withKey("input_layers", "kv_input"),
                                     withKey("disable_bias", "true"),
                                     withKey("weight_initializer", "ones"),
                                   }));

  // MHA Core (cross-attention)
  model->addLayer(createLayer(
    "mha_core", {
                  withKey("name", "cross_attn"),
                  withKey("num_heads", NUM_HEADS_Q),
                  withKey("num_heads_KV", NUM_HEADS_KV),
                  withKey("max_timestep",
                          std::to_string(std::max(Q_SEQ_LEN, KV_SEQ_LEN) + 10)),
                  withKey("is_causal", "false"),
                  withKey("is_cross_attention", "true"),
                  withKey("input_layers", "q_proj,k_proj,v_proj"),
                }));

  // Output projection
  model->addLayer(
    createLayer("fully_connected", {
                                     withKey("name", "o_proj"),
                                     withKey("unit", D_MODEL),
                                     withKey("input_layers", "cross_attn"),
                                     withKey("disable_bias", "true"),
                                     withKey("weight_initializer", "ones"),
                                   }));

  // ============================================================
  // 3. Compile and initialize
  // ============================================================
  std::cout << "\nCompiling model..." << std::endl;
  model->compile(ml::train::ExecutionMode::INFERENCE);

  std::cout << "Initializing model..." << std::endl;
  model->initialize(ml::train::ExecutionMode::INFERENCE);

  model->summarize(std::cout, ML_TRAIN_SUMMARY_MODEL);

  // ============================================================
  // 4. Load weights
  // ============================================================
  std::cout << "Loading weights from: " << weight_path << std::endl;
  model->load(weight_path, ml::train::ModelFormat::MODEL_FORMAT_BIN);

  // ============================================================
  // 5. Load test I/O
  // ============================================================
  std::cout << "Loading test I/O from: " << io_path << std::endl;
  TestIO io = loadTestIO(io_path);

  // ============================================================
  // 6. Run inference
  // ============================================================
  std::cout << "\nRunning incremental inference..." << std::endl;

  // Model has 2 input layers: query_input and kv_input
  std::vector<float *> inputs = {io.query_input.data(), io.kv_input.data()};
  std::vector<float *> labels = {};

  auto output = model->incremental_inference(BATCH_SIZE, inputs, labels,
                                             Q_SEQ_LEN, // init_seq_len
                                             0,         // from
                                             Q_SEQ_LEN, // to
                                             false      // output_hidden_state
  );

  // ============================================================
  // 7. Print results and compare
  // ============================================================
  if (output.empty() || output[0] == nullptr) {
    std::cerr << "ERROR: No output from inference!" << std::endl;
    return 1;
  }

  std::cout << "\n============================================================"
            << std::endl;
  std::cout << "nntrainer OUTPUT:" << std::endl;
  std::cout << "============================================================"
            << std::endl;
  printTensor("nntrainer output", output[0], Q_SEQ_LEN, D_MODEL);

  std::cout << "\n============================================================"
            << std::endl;
  std::cout << "Reference OUTPUT (FP32):" << std::endl;
  std::cout << "============================================================"
            << std::endl;
  printTensor("PyTorch reference", io.ref_output.data(), Q_SEQ_LEN, D_MODEL);

  // Compute max absolute difference
  float max_diff = 0.0f;
  for (unsigned int i = 0; i < Q_SEQ_LEN * D_MODEL; ++i) {
    float diff = std::abs(output[0][i] - io.ref_output[i]);
    if (diff > max_diff)
      max_diff = diff;
  }

  std::cout << "\n============================================================"
            << std::endl;
  std::cout << "Max absolute difference: " << std::scientific << max_diff
            << std::endl;
  std::cout << "============================================================"
            << std::endl;

  if (max_diff < 1e-2f) {
    std::cout << "PASS: Outputs match within tolerance!" << std::endl;
  } else {
    std::cout << "WARNING: Large difference detected. Check implementation."
              << std::endl;
  }

  return 0;
}
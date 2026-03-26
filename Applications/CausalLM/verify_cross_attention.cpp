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
static constexpr unsigned int MAX_Q_SEQ_LEN = 32;
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
  // Note: query_input uses KV_SEQ_LEN shape so incremental_inference can use
  // a single to=KV_SEQ_LEN for all layers (FC clips height to to-from).
  // The last (KV_SEQ_LEN - Q_SEQ_LEN) token slots are zero-padded.
  model->addLayer(createLayer(
    "input", {
               withKey("name", "query_input"),
               withKey("input_shape", "1:1:" + std::to_string(KV_SEQ_LEN) +
                                        ":" + std::to_string(D_MODEL)),
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

  // Print first 3 values of query_input and kv_input for verification
  std::cout << "\nquery_input (first 3 values): [";
  for (unsigned int i = 0; i < 3 && i < io.query_input.size(); ++i) {
    if (i > 0)
      std::cout << ", ";
    std::cout << io.query_input[i];
  }
  std::cout << "]" << std::endl;

  std::cout << "kv_input (first 3 values): [";
  for (unsigned int i = 0; i < 3 && i < io.kv_input.size(); ++i) {
    if (i > 0)
      std::cout << ", ";
    std::cout << io.kv_input[i];
  }
  std::cout << "]" << std::endl;

  // ============================================================
  // 6. Run inference
  // ============================================================
  std::cout << "\nRunning incremental inference..." << std::endl;

  // Zero-pad query_input from Q_SEQ_LEN to KV_SEQ_LEN tokens so that
  // incremental_inference can use to=KV_SEQ_LEN for all layers uniformly.
  std::vector<float> query_input_padded(BATCH_SIZE * MAX_Q_SEQ_LEN * D_MODEL,
                                        0.0f);
  std::copy(io.query_input.begin(), io.query_input.end(),
            query_input_padded.begin());

  // Model graph order (topological sort): kv_input=0, query_input=1
  std::vector<float *> inputs = {io.kv_input.data(), query_input_padded.data()};
  std::vector<float *> labels = {};

  // Use to=KV_SEQ_LEN so k_proj/v_proj process all 5 KV tokens.
  // output_hidden_state=true returns pointer into the full hidden tensor
  // (KV_SEQ_LEN rows), from which we compare only the first Q_SEQ_LEN rows.
  auto output = model->incremental_inference(BATCH_SIZE, inputs, labels,
                                             Q_SEQ_LEN,  // init_seq_len
                                             0,          // from
                                             KV_SEQ_LEN, // to
                                             true        // output_hidden_state
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

  // ============================================================
  // 8. Token generation test (query_seq_len = 1)
  //
  // After the prefill run, k_proj/v_proj output tensors retain all 5 KV
  // positions.  Running with to - from = 1 exercises the q_len == 1 branch in
  // compute_kcaches / compute_fp16vcache_transposed while cross_attn still
  // attends to all 5 KV positions (key_batch dim is not clipped in
  // one_batch_crs_incremental_forwarding).
  //
  // For each decoder position i we run from=0, to=1 with query_input[i] at
  // row 0.  The FC layer always reads from the start of the tensor regardless
  // of 'from', so placing the desired query at row 0 is sufficient.
  // ============================================================
  std::cout
    << "\n============================================================"
    << std::endl;
  std::cout << "TOKEN GENERATION TEST (query_seq_len = 1 per step):"
            << std::endl;
  std::cout << "============================================================"
            << std::endl;

  bool token_gen_pass = true;

  for (unsigned int qi = 0; qi < Q_SEQ_LEN; ++qi) {
    // Build a single-token query: place query row qi at row 0, zero the rest.
    std::vector<float> single_query(BATCH_SIZE * MAX_Q_SEQ_LEN * D_MODEL, 0.0f);
    const float *src = io.query_input.data() + qi * D_MODEL;
    std::copy(src, src + D_MODEL, single_query.begin());

    std::vector<float *> tg_inputs = {io.kv_input.data(), single_query.data()};
    std::vector<float *> tg_labels = {};

    auto tg_output = model->incremental_inference(BATCH_SIZE, tg_inputs,
                                                   tg_labels,
                                                   1,   // init_seq_len = 1
                                                   0,   // from
                                                   1,   // to  (one token)
                                                   true // output_hidden_state
    );

    if (tg_output.empty() || tg_output[0] == nullptr) {
      std::cerr << "ERROR: No output for token gen step " << qi << std::endl;
      token_gen_pass = false;
      continue;
    }

    // Compare tg_output[0][0..D_MODEL-1] against ref_output row qi.
    const float *ref_row = io.ref_output.data() + qi * D_MODEL;
    float step_diff = 0.0f;
    for (unsigned int d = 0; d < D_MODEL; ++d) {
      float diff = std::abs(tg_output[0][d] - ref_row[d]);
      if (diff > step_diff)
        step_diff = diff;
    }

    std::cout << "  Step " << qi << " max_diff = " << std::scientific
              << step_diff;
    if (step_diff < 1e-2f) {
      std::cout << " PASS" << std::endl;
    } else {
      std::cout << " FAIL" << std::endl;
      token_gen_pass = false;
    }
  }

  std::cout << "\n============================================================"
            << std::endl;
  if (token_gen_pass) {
    std::cout << "TOKEN GENERATION TEST: PASS" << std::endl;
  } else {
    std::cout << "TOKEN GENERATION TEST: FAIL" << std::endl;
  }
  std::cout << "============================================================"
            << std::endl;

  return (max_diff < 1e-2f && token_gen_pass) ? 0 : 1;
}
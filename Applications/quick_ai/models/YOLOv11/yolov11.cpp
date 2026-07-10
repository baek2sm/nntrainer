// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   yolov11_transformer.cpp
 * @date   8 Jul 2026
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Factory-registered, config-driven YOLOv11 detection model.
 *
 * Conformance port of the YOLOv11 path to the standard Quick_AI architecture:
 * settings come from nntr_config.json (no YOLO_* env vars), the graph is built
 * by overriding the Transformer base's constructModel flow, and the detection
 * head's three scale outputs are compiled through the multi-output
 * Model::compile overload. See yolov11_transformer.h for the design rationale.
 */

#include "yolov11.h"

#include "c2psa_layer.h"
#include "yolo_postprocess.h"

#include <app_context.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <factory.h>
#include <fstream>
#include <iostream>
#include <llm_util.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace quick_ai {

/**
 * @brief Set YOLOv11 parameters from the model and runtime configs.
 *
 * All knobs are read from nntr_config.json (standard Quick_AI approach), NOT
 * from YOLO_* env vars. Standard fields (model_tensor_type, batch_size) are
 * consumed by the base Transformer; YOLO-specific fields are stored here.
 */
void Yolov11::setupParameters(json &cfg, json &generation_cfg,
                                         json &nntr_cfg) {
  (void)generation_cfg;
  (void)cfg; // YOLOv11 architecture is fixed; widths come from the variant.

  BATCH_SIZE = nntr_cfg.value("batch_size", 1);
  MODEL_TENSOR_TYPE = nntr_cfg.value("model_tensor_type", "FP32-FP32");
  FC_LAYER_DTYPE = nntr_cfg.value("fc_layer_dtype", "FP32");

  // LLM-only fields the base setupParameters reads unconditionally are absent
  // for YOLO; bind defaults so the base does not throw on .get<>() of a
  // missing key. (The base is bypassed here because skip_tokenizer+MODEL keeps
  // the base ctor from calling setupParameters — transformer.cpp — so this
  // override IS the one that runs.)
  INIT_SEQ_LEN = nntr_cfg.value("init_seq_len", 1);
  MAX_SEQ_LEN = nntr_cfg.value("max_seq_len", 1);
  NUM_TO_GENERATE = nntr_cfg.value("num_to_generate", 0);
  EMBEDDING_DTYPE = nntr_cfg.value("embedding_dtype", "FP32");

  // YOLO-specific fields.
  IMGSZ = nntr_cfg.value("imgsz", 832);
  NC = nntr_cfg.value("nc", 1);
  REG_MAX = nntr_cfg.value("yolo_reg_max", 16);
  YOLO_VARIANT = nntr_cfg.value("yolo_variant", "v11m");
  INPUT_NHWC = nntr_cfg.value("input_format", "NCHW") == "NHWC";
  NMS_CONF = nntr_cfg.value("nms_conf", 0.25f);
  NMS_IOU = nntr_cfg.value("nms_iou", 0.70f);
  YOLO_VERIFY = nntr_cfg.value("yolo_verify", false);
  YOLO_DUMP_RAW = nntr_cfg.value("yolo_dump_raw", false);
  YOLO_BENCH_ITERS = nntr_cfg.value("yolo_bench_iters", 1);
  YOLO_REF_DIR = nntr_cfg.value("yolo_ref_dir", std::string());
  // Conv weight dtype (Q4_0/Q8_0 for quantized conv models, FP32 otherwise).
  // Deliberately separate from model_tensor_type: a quantized-conv model keeps
  // model_tensor_type FP32-FP{32,16} (activation dtype) and only flips conv
  // weights via this knob + the graph builder's conv_q40 gate.
  CONV_DTYPE_STR = nntr_cfg.value("conv_dtype", "FP32");

  if (YOLO_VARIANT == "v11s" || YOLO_VARIANT == "V11S") {
    cfg_ = yolov11::ModelConfig::v11s();
  } else {
    cfg_ = yolov11::ModelConfig::v11m();
  }
}

/**
 * @brief Build the symbolic YOLOv11 graph: input → backbone → head → 3 scales.
 *
 * Returns the leaf input and the three detect scale outputs (P3/P4/P5). The
 * input tensor dtype/format is derived from MODEL_TENSOR_TYPE so an FP16-
 * activation model declares an FP16 input.
 *
 * @return {input, {P3, P4, P5}}.
 */
std::pair<Tensor, std::vector<Tensor>>
Yolov11::constructYoloModel() const {
  // Activation dtype from the "weight-act" pair in model_tensor_type.
  const std::string &mtt = MODEL_TENSOR_TYPE;
  const auto dash = mtt.find('-');
  const std::string act_dt =
    (dash == std::string::npos) ? mtt : mtt.substr(dash + 1);
  const bool fp16_act = (act_dt == "FP16");

  const auto in_fmt = INPUT_NHWC ? ml::train::TensorDim::Format::NHWC
                                 : ml::train::TensorDim::Format::NCHW;
  const auto in_dt = fp16_act ? ml::train::TensorDim::DataType::FP16
                              : ml::train::TensorDim::DataType::FP32;

  Tensor x(ml::train::TensorDim(BATCH_SIZE, 3, IMGSZ, IMGSZ, in_fmt, in_dt),
           "input0");

  // Conv weight quantization is a build-time concern separate from
  // model_tensor_type: a Q4_0/Q8_0 conv model keeps model_tensor_type at
  // FP32-FP{32,16} (the framework still allocates conv weights from the
  // per-layer weight_dtype property) and enables conv quantization via
  // conv_dtype. When set, the graph builder declares eligible convs with that
  // weight_dtype so the runtime allocates the [1,1,CRS,out_ch] packed shape the
  // saved file carries — otherwise the FP32 [out_ch,in_ch,kh,kw] declaration
  // mismatches the loaded Q4_0/Q8_0 weight. quantWeightDtype() selects the
  // precision; conv_q40 is the on/off gate (Q4_0 and Q8_0 share eligibility:
  // both QK=32 block-aligned).
  const bool conv_quantized =
    (CONV_DTYPE_STR == "Q4_0" || CONV_DTYPE_STR == "Q8_0");
  yolov11::quantWeightDtype() = conv_quantized ? CONV_DTYPE_STR : "Q4_0";
  const bool conv_q40 = conv_quantized;

  Tensor m4, m6;
  auto m10 = yolov11::buildBackbone(x, m4, m6, cfg_, conv_q40);
  auto outputs =
    yolov11::buildHead(m4, m6, m10, NC, cfg_, conv_q40); // {P3, P4, P5}
  return {x, outputs};
}

/**
 * @brief Collect the names of every Q-eligible conv the graph builder creates.
 *
 * The conv block builders (yolov11_graph.h) push a conv's layer name to
 * quantConvSink() only while it is non-null. This attaches a fresh vector,
 * rebuilds the symbolic graph (which runs the builders without compiling), and
 * returns the collected names. conv_q40 is left false here because the sink is
 * populated regardless of the weight_dtype the builder would assign — the
 * *name set* of eligible convs is identical for Q4_0 and Q8_0 (both require the
 * same 32-alignment), and the per-conv target dtype is decided by the caller
 * (nntr_quantize) via the layer_dtype_map, not by the builder.
 */
std::vector<std::string> Yolov11::getQuantizableLayerNames() const {
  std::vector<std::string> names;
  auto *prev = yolov11::quantConvSink();
  yolov11::quantConvSink() = &names;
  // Rebuild purely to collect names; the returned tensors are discarded. The
  // builders only construct LayerHandles, so no model mutation occurs.
  (void)constructYoloModel();
  yolov11::quantConvSink() =
    prev; // restore so normal inference builds don't collect
  return names;
}

/**
 * @brief Build and compile the YOLOv11 graph with three scale outputs.
 *
 * Overrides Transformer::initialize() because the base compiles a single
 * output whereas YOLOv11 emits P3/P4/P5. Uses the multi-output Model::compile
 * overload (model.h:178 — "supports models with multiple output heads, e.g.
 * YOLOv3 with 3 loss layers").
 */
void Yolov11::initialize() {
  registerCustomLayers();

  model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);

  std::vector<std::string> model_props = {
    withKey("batch_size", BATCH_SIZE), withKey("epochs", "1"),
    withKey("model_tensor_type", MODEL_TENSOR_TYPE)};
  if (INPUT_NHWC) {
    model_props.emplace_back(withKey("tensor_format", "NHWC"));
  }
  model->setProperty(model_props);

  auto [x, outputs] = constructYoloModel();

  if (model->compile(x, outputs, ml::train::ExecutionMode::INFERENCE)) {
    throw std::invalid_argument("YOLOv11 model compilation failed.");
  }

  is_initialized = true;
#ifdef DEBUG
  model->summarize(std::cout, ML_TRAIN_SUMMARY_MODEL);
#endif
}

/**
 * @brief Register the YOLOv11 custom layers, then the base transformer's.
 *
 * The psa_attention layer (C2PSA) is the only YOLO-specific custom layer; the
 * detect head uses standard nntrainer layers. Registered here per the standard
 * (each Factory-registered model registers its own custom layers).
 */
void Yolov11::registerCustomLayers() {
  auto &app_ctx = nntrainer::AppContext::Global();
  app_ctx.registerFactory(nntrainer::createLayer<yolov11::PSAAttentionLayer>);
}

/**
 * @brief Run YOLOv11 inference on an input Bin/image and emit detections.
 *
 * Loads the preprocessed FP32 input [.bin] from the prompt path, runs timed
 * inference (YOLO_BENCH_ITERS), then DFL decode + dist2bbox + NMS to emit JSON
 * detections. When YOLO_VERIFY is set, compares the raw logits and decoded
 * output against PyTorch reference .bin files (ref_p3/p4/p5.bin,
 * ref_decoded.bin); with YOLO_DUMP_RAW, also dumps them. Post-proc lives in
 * yolo_postprocess.h. The Model::inference contract takes float* inputs; FP16
 * activations are produced internally by the framework from the FP32 feed.
 */
void Yolov11::run(const WSTR prompt, bool do_sample,
                             const WSTR system_prompt, const WSTR tail_prompt,
                             bool log_output) {
  (void)do_sample;
  (void)system_prompt;
  (void)tail_prompt;
  (void)log_output;

  if (!is_initialized) {
    throw std::runtime_error("YOLOv11 model is not initialized. Please call "
                             "initialize() before run().");
  }

  // Load the preprocessed input Bin written by res/YOLOv11/run_pytorch.py (a
  // [1,3,IMGSZ,IMGSZ] float32 NCHW tensor — identical bytes both sides see).
  const std::string input_path(prompt);
  std::ifstream f(input_path, std::ios::binary);
  if (!f.is_open()) {
    throw std::runtime_error("Failed to open YOLO input: " + input_path);
  }
  f.seekg(0, std::ios::end);
  const size_t n_bytes = f.tellg();
  f.seekg(0, std::ios::beg);

  const size_t expect_n = (size_t)BATCH_SIZE * 3 * IMGSZ * IMGSZ;
  if (n_bytes != expect_n * sizeof(float)) {
    throw std::runtime_error("YOLO input size mismatch: got " +
                             std::to_string(n_bytes) + " bytes, expected " +
                             std::to_string(expect_n * sizeof(float)));
  }
  // When the graph is NHWC, the input bytes must be NHWC-ordered ([N,H,W,C]);
  // input bins are stored NCHW, so transpose here.
  std::vector<float> buf(expect_n);
  f.read(reinterpret_cast<char *>(buf.data()), n_bytes);
  if (INPUT_NHWC) {
    const int C = 3, H = IMGSZ, W = IMGSZ;
    std::vector<float> nhwc(buf.size());
    for (int c = 0; c < C; ++c)
      for (int h = 0; h < H; ++h)
        for (int w = 0; w < W; ++w)
          nhwc[(h * W + w) * C + c] = buf[(c * H + h) * W + w];
    buf.swap(nhwc);
  }
  std::vector<float *> in{buf.data()};

  // Timed inference — average over YOLO_BENCH_ITERS, last run feeds post-proc.
  std::vector<float *> outs;
  double total_ms = 0.0;
  for (unsigned int it = 0; it < YOLO_BENCH_ITERS; ++it) {
    auto t0 = std::chrono::steady_clock::now();
    outs = model->inference(BATCH_SIZE, in, std::vector<float *>());
    auto t1 = std::chrono::steady_clock::now();
    total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
  }
  std::cout << "Inference done (" << outs.size() << " scale outputs)."
            << std::endl;
  std::cout << "Inference time: " << (total_ms / YOLO_BENCH_ITERS)
            << " ms (avg over " << YOLO_BENCH_ITERS << " iters)" << std::endl;

  // DFL decode + dist2bbox + sigmoid -> [4+nc, N_total] then NMS.
  const int imgsz = IMGSZ;
  const int nc = NC;
  const int reg_max = REG_MAX;
  const int box_ch = 4 * reg_max;
  const int out_ch = box_ch + nc;
  const int s3 = imgsz / 8, s4 = imgsz / 16, s5 = imgsz / 32;
  std::vector<yolov11::ScaleInfo> scales = {
    {s3, s3, 8.0f}, {s4, s4, 16.0f}, {s5, s5, 32.0f}};
  const int N_total = s3 * s3 + s4 * s4 + s5 * s5;
  std::vector<float> anchors, strides;
  yolov11::makeAnchors(scales, anchors, strides);

  std::vector<float> decoded(static_cast<size_t>(4 + nc) * N_total, 0.0f);
  int off = 0;
  for (size_t i = 0; i < scales.size(); ++i) {
    const float *raw = outs[i];
    std::vector<float> nchw_buf;
    if (INPUT_NHWC) {
      const int N = scales[i].H * scales[i].W;
      nchw_buf.resize(static_cast<size_t>(out_ch) * N);
      for (int a = 0; a < N; ++a)
        for (int c = 0; c < out_ch; ++c)
          nchw_buf[static_cast<size_t>(c) * N + a] =
            raw[static_cast<size_t>(a) * out_ch + c];
      raw = nchw_buf.data();
    }
    yolov11::decodeOneScale(raw, scales[i].H, scales[i].W, scales[i].stride,
                            anchors, strides, off, N_total, box_ch, nc,
                            decoded);
    off += scales[i].H * scales[i].W;
  }

  auto dets = yolov11::nms(decoded, N_total, nc, NMS_CONF, NMS_IOU, 300);

  // JSON output — same field names as the PyTorch reference.
  std::cout << "\n[";
  for (size_t i = 0; i < dets.size(); ++i) {
    const auto &d = dets[i];
    if (i)
      std::cout << ",";
    std::printf("\n  {\"x1\": %.6g, \"y1\": %.6g, \"x2\": %.6g,"
                " \"y2\": %.6g, \"conf\": %.6g, \"cls\": %d}",
                d.x1, d.y1, d.x2, d.y2, d.conf, d.cls);
  }
  std::cout << (dets.empty() ? "" : "\n") << "]" << std::endl;

  // Optional: compare logits/decoded vs PyTorch reference .bin, and dump raw.
  if (YOLO_VERIFY) {
    std::cout << "\nVerification vs PyTorch references:" << std::endl;
    const size_t ns[3] = {static_cast<size_t>(out_ch) * s3 * s3,
                          static_cast<size_t>(out_ch) * s4 * s4,
                          static_cast<size_t>(out_ch) * s5 * s5};
    const char *names[3] = {"ref_p3.bin", "ref_p4.bin", "ref_p5.bin"};
    auto verifyAgainst = [&](const std::string &ref_name, const float *out,
                             size_t n) {
      const std::string rp = YOLO_REF_DIR + "/" + ref_name;
      std::ifstream rf(rp, std::ios::binary);
      if (!rf) {
        std::cout << "  [verify] " << ref_name << " not found, skipped"
                  << std::endl;
        return;
      }
      std::vector<float> ref(n);
      rf.read(reinterpret_cast<char *>(ref.data()), n * sizeof(float));
      // NaN must never masquerade as a 0 diff. This verify runs in a TU built
      // with -ffast-math (Android.mk), under which std::isnan is compiled away
      // to `false` — so it cannot detect NaN. Inspect the IEEE-754 bit pattern
      // directly instead: a float is NaN when its exponent bits are all 1 and
      // the mantissa is non-zero. The all-NaN output of a broken FP16 path
      // would otherwise print max_abs_diff=0 (diff of NaN is NaN, std::max
      // collapses it under fast-math) and look like a perfect PASS.
      auto is_nan_bits = [](float v) {
        uint32_t u;
        std::memcpy(&u, &v, sizeof(u));
        return ((u & 0x7F800000u) == 0x7F800000u) && (u & 0x007FFFFFu);
      };
      float max_diff = 0.0f;
      size_t out_nan = 0, ref_nan = 0, both_finite = 0;
      for (size_t i = 0; i < n; ++i) {
        if (is_nan_bits(out[i]))
          ++out_nan;
        if (is_nan_bits(ref[i]))
          ++ref_nan;
        if (!is_nan_bits(out[i]) && !is_nan_bits(ref[i])) {
          max_diff = std::max(max_diff, std::abs(out[i] - ref[i]));
          ++both_finite;
        }
      }
      const char *tag = (out_nan == n) ? "FAIL" : (out_nan > 0) ? "WARN" : "OK";
      std::cout << "  [verify] " << ref_name << ": " << tag
                << " max_abs_diff=" << max_diff << " out_nan=" << out_nan << "/"
                << n << " ref_nan=" << ref_nan << "/" << n
                << " both_finite=" << both_finite << std::endl;
    };
    std::vector<float> nchw_verify;
    for (int i = 0; i < 3; ++i) {
      const int N = ns[i] / out_ch;
      nchw_verify.resize(ns[i]);
      if (INPUT_NHWC) {
        for (int a = 0; a < N; ++a)
          for (int c = 0; c < out_ch; ++c)
            nchw_verify[static_cast<size_t>(c) * N + a] =
              outs[i][static_cast<size_t>(a) * out_ch + c];
      } else {
        std::copy(outs[i], outs[i] + ns[i], nchw_verify.begin());
      }
      verifyAgainst(names[i], nchw_verify.data(), ns[i]);
      if (YOLO_DUMP_RAW) {
        const std::string p = YOLO_REF_DIR + "/dump_" + names[i];
        std::ofstream ofs(p, std::ios::binary);
        ofs.write(reinterpret_cast<const char *>(nchw_verify.data()),
                  ns[i] * sizeof(float));
        if (ofs) {
          std::cout << "  [dump] " << p << " (" << ns[i] << " floats)"
                    << std::endl;
        } else {
          std::cout << "  [dump] FAILED to write " << p << std::endl;
        }
      }
    }
    verifyAgainst("ref_decoded.bin", decoded.data(), decoded.size());
    if (YOLO_DUMP_RAW) {
      const std::string p = YOLO_REF_DIR + "/dump_decoded.bin";
      std::ofstream ofs(p, std::ios::binary);
      ofs.write(reinterpret_cast<const char *>(decoded.data()),
                decoded.size() * sizeof(float));
      if (ofs) {
        std::cout << "  [dump] " << p << " (" << decoded.size() << " floats)"
                  << std::endl;
      } else {
        std::cout << "  [dump] FAILED to write " << p << std::endl;
      }
    }
  }
}

} // namespace quick_ai

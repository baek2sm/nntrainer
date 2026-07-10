// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   yolov11_transformer.h
 * @date   8 Jul 2026
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Factory-registered, config-driven YOLOv11 detection model.
 *
 * This brings the YOLOv11 inference path into conformance with the standard
 * Quick_AI app architecture: the model is registered with the Factory and
 * driven by nntr_config.json (no YOLO_* env vars), so the standard
 * nntrainer_quick_ai binary runs it and nntr_quantize quantizes it — mirroring
 * how the LLM (Qwen3/Gemma3/...) and TimmViT models are handled.
 *
 * The construction mirrors TimmViTTransformer: a conv-based, tokenizer-free
 * vision model built by overriding constructModel()/run() on the Transformer
 * base, with ModelType::MODEL + skip_tokenizer so the base ctor skips the
 * text/tokenizer setup (transformer.cpp). The detection graph produces three
 * scale outputs (P3/P4/P5); Transformer::initialize() is overridden to invoke
 * the multi-output Model::compile overload (model.h), which explicitly supports
 * detection heads (e.g. "YOLOv3 with 3 loss layers").
 */

#ifndef __YOLOV11_TRANSFORMER_H__
#define __YOLOV11_TRANSFORMER_H__

#include "model_base.h"

#include "yolov11_graph.h"

namespace quick_ai {

/**
 * @brief Yolov11 class — a Factory-registered detection model.
 */
class Yolov11 : public Model {

public:
  static constexpr const char *architectures = "YOLOv11ForDetection";

  /**
   * @brief Construct a Yolov11 object.
   */
  Yolov11(json &cfg, json &generation_cfg, json &nntr_cfg) {
    setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  /**
   * @brief Destroy the Yolov11 object.
   *
   * Inline `= default` (matching TimmViTTransformer): the vtable is emitted as
   * a weak COMDAT in each translation unit that needs it, so binaries that
   * reference the class via the Factory (e.g. nntr_quantize, which does not
   * compile this .cpp) still link. An out-of-line key function would force
   * every such binary to compile yolov11_transformer.cpp.
   */
  virtual ~Yolov11() = default;

  /**
   * @brief Names of every conv layer the graph builder marks Q-eligible.
   *
   * Drives the standard nntr_quantize per-layer dtype map: the conv block
   * builders (yolov11_graph.h) push each eligible conv's name to
   * quantConvSink() when it is set, so this rebuilds the graph with the sink
   * attached and returns the collected names. The eligibility (out_ch and
   * in_ch*k*k both 32-aligned, groups==1) matches Conv2DLayer::save's Q4_0/Q8_0
   * guard, so exactly the convs the saver will convert are named here.
   */
  std::vector<std::string> getQuantizableLayerNames() const override;

protected:
  /**
   * @brief Build and compile the YOLOv11 detection graph.
   *
   * Overridden because the detection head emits three scale outputs (P3/P4/P5)
   * — the base Transformer::initialize() compiles a single output; this calls
   * the multi-output Model::compile overload instead.
   */
  void initialize() override;

  /**
   * @brief Construct the symbolic YOLOv11 graph, returning the input and the
   *        three detect scale outputs.
   *
   * @return {input, {P3, P4, P5}} symbolic tensors.
   */
  std::pair<Tensor, std::vector<Tensor>> constructYoloModel() const;

  /**
   * @brief Set YOLOv11 parameters from HuggingFace and nntrainer configs.
   */
  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;

  /**
   * @brief Register the YOLOv11 custom layers (psa_attention) with the global
   *        AppContext, then the base transformer's custom layers.
   */
  void registerCustomLayers() override;

  /**
   * @brief Run the model on an image/bin input path and emit JSON detections.
   */
  void run(const WSTR prompt, bool do_sample = false,
           const WSTR system_prompt = WSTR(), const WSTR tail_prompt = WSTR(),
           bool log_output = true) override;

private:
  unsigned int IMGSZ = 1024;         /**< input image size (square) */
  unsigned int NC = 1;               /**< number of detection classes */
  unsigned int REG_MAX = 16;         /**< DFL reg_max (box bins per coord) */
  unsigned int BATCH_SIZE = 1;       /**< batch size */
  std::string YOLO_VARIANT = "v11m"; /**< model size preset: "v11m" | "v11s" */
  bool INPUT_NHWC = false;           /**< NHWC layout flag */
  float NMS_CONF = 0.25f;            /**< NMS confidence threshold */
  float NMS_IOU = 0.70f;             /**< NMS IoU threshold */

  bool YOLO_VERIFY = false;   /**< compare logits to ref_*.bin when set */
  bool YOLO_DUMP_RAW = false; /**< dump raw/decoded logits when set */
  unsigned int YOLO_BENCH_ITERS = 1; /**< timed-iteration count */
  std::string YOLO_REF_DIR;          /**< dir with ref_*.bin for verify/dump */
  std::string CONV_DTYPE_STR =
    "FP32"; /**< conv weight dtype (Q4_0/Q8_0/FP32) */

  std::string MODEL_TENSOR_TYPE = "FP32-FP32";
  std::string FC_LAYER_DTYPE = "FP32";
  std::string EMBEDDING_DTYPE = "FP32";
  unsigned int INIT_SEQ_LEN = 1;
  unsigned int MAX_SEQ_LEN = 1;
  unsigned int NUM_TO_GENERATE = 0;

  yolov11::ModelConfig cfg_ = yolov11::ModelConfig::v11s();
};

} // namespace quick_ai

#endif /* __YOLOV11_TRANSFORMER_H__ */

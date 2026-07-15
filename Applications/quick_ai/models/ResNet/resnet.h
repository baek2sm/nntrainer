// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   resnet.h
 * @date   14 July 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @brief  General ResNet vision backbone (including IR & Mona) model class.
 */

#ifndef __RESNET_H__
#define __RESNET_H__

#pragma once

#include "model_base.h"

namespace quick_ai {

/**
 * @brief ResNet class for image classification and feature extraction (standard
 * ResNet & IR configurations)
 */
class ResNet : public Model {
public:
  static constexpr const char *architectures = "ResNet";

  ResNet(json &cfg, json &generation_cfg, json &nntr_cfg);

  virtual ~ResNet() = default;

  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;

  void initialize() override;

  void registerCustomLayers() override;

  void run(const WSTR prompt, bool do_sample = false,
           const WSTR system_prompt = WSTR(), const WSTR tail_prompt = WSTR(),
           bool log_output = true) override;

protected:
  // Model hyperparameters
  unsigned int BATCH_SIZE = 1;
  std::string MODEL_TENSOR_TYPE = "FP32-FP32";
  std::string CONV_DTYPE_STR = "FP32";
  unsigned int IMGSZ = 112;

  // General ResNet block and channel specifications
  std::vector<int> BLOCK_DEPTHS = {3, 4, 14, 3}; // Default ResNet-50 / IR-50
  std::vector<int> BLOCK_WIDTHS = {64, 128, 256, 512};

  // Inference configs
  bool MONA_VERIFY = false;
  bool MONA_DUMP_RAW = false;
  unsigned int MONA_BENCH_ITERS = 1;
  std::string MONA_REF_DIR = ".";
  float USE_MONA_VAL =
    0.0f; // 0.0 for human face feature, 1.0 for pet face feature

  // Verification helper
  void verifyAgainstPyTorch(const std::vector<float *> &outs);
};

} // namespace quick_ai

#endif // __RESNET_H__

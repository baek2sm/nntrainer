// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   resnet_graph.h
 * @date   14 July 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @brief  Symbolic neural network graph builder for General ResNet.
 */

#ifndef __RESNET_GRAPH_H__
#define __RESNET_GRAPH_H__

#pragma once

#include <layer.h>
#include <tensor_api.h>
#include <vector>

namespace quick_ai {
namespace resnet {

using LayerHandle = ml::train::LayerHandle;
using Tensor = ml::train::Tensor;

// ===== Utility Graph Layer Helpers =====

inline LayerHandle createLayer(const std::string &type,
                               const std::vector<std::string> &properties) {
  return ml::train::createLayer(type, properties);
}

inline std::string pad(int p) { return std::to_string(p); }

inline std::string withKey(const std::string &key, const std::string &val) {
  return key + "=" + val;
}

inline std::string withKey(const std::string &key, int val) {
  return key + "=" + std::to_string(val);
}

// conv2d wrapper
inline Tensor conv2d(const std::string &name, int filters, int k, int stride,
                     const std::string &padding, const std::string &act,
                     bool bias, int groups, bool conv_q40, Tensor input) {
  std::vector<std::string> props = {
    withKey("name", name),
    withKey("filters", filters),
    withKey("kernel_size", std::to_string(k) + "," + std::to_string(k)),
    withKey("stride", std::to_string(stride) + "," + std::to_string(stride)),
    withKey("padding", padding + "," + padding),
    withKey("disable_bias", bias ? "false" : "true")};
  if (groups > 1) {
    props.emplace_back(withKey("groups", groups));
  }
  if (!act.empty()) {
    props.emplace_back(withKey("activation", act));
  }
  if (conv_q40) {
    props.emplace_back(withKey("weight_initializer", "ones"));
  }
  auto l = createLayer("conv2d", props);
  return l(input);
}

// cast wrapper
inline Tensor cast(const std::string &name, const std::string &dtype,
                   Tensor input) {
  auto l = createLayer("cast",
                       {withKey("name", name), withKey("tensor_dtype", dtype)});
  return l(input);
}

// batch_normalization wrapper
inline Tensor batch_normalization(const std::string &name, Tensor input,
                                  bool is_mixed = false) {
  auto l = createLayer("batch_normalization",
                       {withKey("name", name), withKey("epsilon", "1e-5")});
  auto h = l(input);
  if (is_mixed) {
    h = cast(name + "/cast_fp16", "FP16", h);
  }
  return h;
}

// prelu wrapper (PReLU custom layer)
inline Tensor prelu(const std::string &name, Tensor input) {
  auto l = createLayer("prelu", {withKey("name", name)});
  return l(input);
}

// max pooling wrapper
inline Tensor maxPool2D(const std::string &name, int k, int stride,
                        const std::string &padding, Tensor input) {
  auto pool = createLayer(
    "pooling2d",
    {withKey("name", name), withKey("pooling", "max"),
     withKey("pool_size", std::to_string(k) + "," + std::to_string(k)),
     withKey("stride", std::to_string(stride) + "," + std::to_string(stride)),
     withKey("padding", padding + "," + padding)});
  return pool(input);
}

// multiply wrapper
inline Tensor multiply(const std::string &name, Tensor input1, Tensor input2) {
  LayerHandle mul = createLayer("multiply", {withKey("name", name)});
  return mul({input1, input2});
}

// addition wrappers
inline Tensor add(const std::string &name, Tensor input1, Tensor input2) {
  LayerHandle addition(createLayer("Addition", {withKey("name", name)}));
  std::vector<Tensor> inputs = {input1, input2};
  return addition(inputs);
}

inline Tensor add3(const std::string &name, Tensor input1, Tensor input2,
                   Tensor input3) {
  LayerHandle addition(createLayer("Addition", {withKey("name", name)}));
  std::vector<Tensor> inputs = {input1, input2, input3};
  return addition(inputs);
}

// flatten wrapper
inline Tensor flatten(const std::string &name, Tensor input) {
  auto l = createLayer("flatten", {withKey("name", name)});
  return l(input);
}

// fully connected wrapper
inline Tensor fully_connected(const std::string &name, int unit, Tensor input) {
  auto l = createLayer("fully_connected",
                       {withKey("name", name), withKey("unit", unit),
                        withKey("disable_bias", "false")});
  return l(input);
}

// ===== Composite Blocks =====

inline Tensor convBnPrelu(const std::string &name, int out_ch, int k,
                          int stride, int padding, Tensor input,
                          bool conv_q40 = false, bool is_mixed = false) {
  auto h = conv2d(name + "/conv", out_ch, k, stride, pad(padding), "", false, 1,
                  conv_q40, input);
  h = batch_normalization(name + "/bn", h, is_mixed);
  return prelu(name + "/prelu", h);
}

// Mona Module Block (Optional, attached when use_mona_adapter == true)
inline Tensor monaBlock(const std::string &name, int channels, Tensor x_mid,
                        Tensor use_mona, bool conv_q40 = false,
                        bool is_mixed = false) {
  // 1) norm(x_mid) -> mona_norm
  auto norm_x = batch_normalization(name + "/mona_norm", x_mid, is_mixed);

  // 2) norm_x * gamma (represented as depthwise 1x1 conv)
  auto scaled_norm_x = conv2d(name + "/mona_mul_gamma", channels, 1, 1, "0", "",
                              false, channels, conv_q40, norm_x);

  // 3) x_mid * gammax (represented as depthwise 1x1 conv)
  auto scaled_x_mid = conv2d(name + "/mona_mul_gammax", channels, 1, 1, "0", "",
                             false, channels, conv_q40, x_mid);

  // 4) mona_in = scaled_norm_x + scaled_x_mid
  auto mona_in = add(name + "/mona_add_g", scaled_norm_x, scaled_x_mid);

  // 5) project1 (Conv2D 1x1 with bias!) -> [32, channels, 1, 1]
  auto proj1 =
    conv2d(name + "/mona_proj1", 32, 1, 1, "0", "", true, 1, conv_q40, mona_in);

  // 6) nonlinear (PReLU)
  auto nonlinear_out = prelu(name + "/mona_prelu", proj1);

  // 7) adapter_conv (MonaOp)
  // conv1: Depthwise 3x3 with bias!
  auto op_conv1 = conv2d(name + "/mona_op_conv1", 32, 3, 1, "1", "", true, 32,
                         conv_q40, nonlinear_out);
  // conv2: Depthwise 5x5 with bias!
  auto op_conv2 = conv2d(name + "/mona_op_conv2", 32, 5, 1, "2", "", true, 32,
                         conv_q40, nonlinear_out);
  // conv3: Depthwise 7x7 with bias!
  auto op_conv3 = conv2d(name + "/mona_op_conv3", 32, 7, 1, "3", "", true, 32,
                         conv_q40, nonlinear_out);

  // Average the three outputs: (conv1 + conv2 + conv3) * (1/3.0)
  auto op_sum = add3(name + "/mona_op_add", op_conv1, op_conv2, op_conv3);
  // scale by 0.3333333f via depthwise 1x1 conv (weights initialized to 1/3)
  auto op_avg = conv2d(name + "/mona_op_scale", 32, 1, 1, "0", "", false, 32,
                       conv_q40, op_sum);

  // Residual add of nonlinear_out
  auto op_mid = add(name + "/mona_op_res", op_avg, nonlinear_out);

  // projector (pointwise Conv2D 1x1 with bias!) -> [32, 32, 1, 1]
  auto op_proj = conv2d(name + "/mona_op_proj", 32, 1, 1, "0", "", true, 1,
                        conv_q40, op_mid);

  // Residual add of op_mid
  auto op_out = add(name + "/mona_op_final_add", op_proj, op_mid);

  // 8) project2 (Conv2D 1x1 with bias!) -> [channels, 32, 1, 1]
  auto proj2 = conv2d(name + "/mona_proj2", channels, 1, 1, "0", "", true, 1,
                      conv_q40, op_out);

  // 9) Gated path: proj2 * use_mona
  auto gated = multiply(name + "/mona_gated", proj2, use_mona);

  // 10) Final add: gated + x_mid
  return add(name + "/mona_add", gated, x_mid);
}

// Improved ResNet bottleneck block with optional Mona adapter
inline Tensor bottleneckIR(const std::string &name, int in_ch, int out_ch,
                           int stride, Tensor input, Tensor use_mona,
                           bool conv_q40 = false, bool is_mixed = false) {
  // 1) res_layer (residual path)
  auto h = batch_normalization(name + "/bn1", input, is_mixed);
  // first conv: 3x3, stride=1
  h = conv2d(name + "/conv1", in_ch, 3, 1, "1", "", false, 1, conv_q40, h);
  h = prelu(name + "/prelu", h);
  // second conv: 3x3, stride=stride
  h =
    conv2d(name + "/conv2", out_ch, 3, stride, "1", "", false, 1, conv_q40, h);
  auto res_out = batch_normalization(name + "/bn2", h, is_mixed);

  // 2) shortcut_layer
  Tensor shortcut_out;
  if (in_ch != out_ch || stride != 1) {
    // Stage first blocks downsample!
    if (name == "body/0") {
      // body.0 downsamples with max_pool2d
      shortcut_out = maxPool2D(name + "/shortcut", 1, 2, "0", input);
    } else {
      // body.3, body.7, body.21 downsample with projection conv + bn!
      auto proj_conv = conv2d(name + "/shortcut_conv", out_ch, 1, stride, "0",
                              "", false, 1, conv_q40, input);
      shortcut_out =
        batch_normalization(name + "/shortcut_bn", proj_conv, is_mixed);
    }
  } else {
    // Non-downsampling blocks: direct bypass (Identity)
    shortcut_out = input;
  }

  // 3) intermediate residual add: x_mid = res_out + shortcut_out
  auto x_mid = add(name + "/add", res_out, shortcut_out);

  // 4) Optional Mona block with use_mona gating (Mona Face feature models)
  // Standard ResNet bypasses Mona and returns x_mid directly
  return monaBlock(name, out_ch, x_mid, use_mona, conv_q40, is_mixed);
}

// ===== Main Graph Builder =====

struct ResNetConfig {
  std::vector<int> d = {
    3, 4, 14, 3}; // blocks count per stage (Default ResNet-50 / IR-50)
  std::vector<int> w = {64, 128, 256, 512}; // channels per stage
};

inline std::pair<std::vector<Tensor>, Tensor>
constructResNetGraph(const ResNetConfig &cfg, int imgsz, bool conv_q40 = false,
                     bool is_mixed = false) {
  // 1) Define inputs as Tensors directly with TensorDim and Name
  Tensor xIn(ml::train::TensorDim(1, 3, imgsz, imgsz), "input0");
  Tensor useMona(ml::train::TensorDim(1, 1, 1, 1), "use_mona");

  // 2) Input Layer
  auto h =
    convBnPrelu("input_layer", cfg.w[0], 3, 1, 1, xIn, conv_q40, is_mixed);

  // 3) Body Blocks
  int block_idx = 0;
  for (size_t s = 0; s < cfg.d.size(); ++s) {
    int in_ch = (s == 0) ? cfg.w[0] : cfg.w[s - 1];
    int out_ch = cfg.w[s];
    for (int b = 0; b < cfg.d[s]; ++b) {
      std::string block_name = "body/" + std::to_string(block_idx);
      int stride = (b == 0) ? 2 : 1;
      h = bottleneckIR(block_name, in_ch, out_ch, stride, h, useMona, conv_q40,
                       is_mixed);
      in_ch = out_ch;
      block_idx++;
    }
  }

  // 4) Output Layer
  auto out_bn2d = batch_normalization("output_layer/bn2d", h, is_mixed);
  auto out_flat = flatten("output_layer/flatten", out_bn2d);
  auto out_fc = fully_connected("output_layer/fc", 256, out_flat);
  auto out_feat = batch_normalization("output_layer/bn1d", out_fc, is_mixed);

  std::vector<Tensor> inputs = {xIn, useMona};
  return {inputs, out_feat};
}

} // namespace resnet
} // namespace quick_ai

#endif // __RESNET_GRAPH_H__

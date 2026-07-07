// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   yolov11_graph.h
 * @date   25 June 2026
 * @brief  YOLOv11m graph block builders (inline, header-only).
 *
 * Extracted from main.cpp so the quantization-aware model class
 * (YOLOv11Model) can reuse the same graph construction without
 * duplicating code.
 *
 * Usage:
 *   #include "yolov11_graph.h"
 *   bool q40 = (std::getenv("YOLO_CONV_Q40") != nullptr);
 *   Tensor m4, m6;
 *   auto m10 = yolov11::buildBackbone(input, m4, m6, q40);
 *   auto outputs = yolov11::buildHead(m4, m6, m10, q40);
 *
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 */

#ifndef __YOLOV11_GRAPH_H__
#define __YOLOV11_GRAPH_H__

#include <string>
#include <vector>

#include <layer.h>
#include <model.h>
#include <tensor_api.h>

using ml::train::createLayer;
using ml::train::LayerHandle;
using ml::train::Tensor;

namespace yolov11 {

// When set (non-null), the conv block builders append the layer name of every
// Q4_0-eligible conv filter (out_ch and in_ch*k*k both 32-aligned) to this
// sink. The offline quantizer (main.cpp --quantize mode) uses it to build the
// per-layer dtype map for model->save() without enumerating the compiled
// graph. Default null = no collection. Eligibility here MUST match the
// conv_q40 gate below and quantize_q4_0_conv.py.
inline std::vector<std::string> *&quantConvSink() {
  static std::vector<std::string> *sink = nullptr;
  return sink;
}

// ===== Primitive graph block builders =====

/**
 * @brief Build a (BN-fused) Conv2d + SiLU sub-graph block.
 *
 * BatchNorm is folded into the convolution at convert time, so this is a
 * single biased conv followed by a standalone SiLU.
 *
 * @param name      Layer-name prefix
 * @param in_ch     Input channels
 * @param out_ch    Output channels
 * @param k         Kernel size (square)
 * @param stride    Stride
 * @param padding   Padding
 * @param input     Input symbolic tensor
 * @param conv_q40  If true, eligible convs get weight_dtype=Q4_0.
 *                  Eligibility: groups==1 && out_ch>1 && out_ch%32==0 &&
 *                  (in_ch*k*k)%32==0 (must match quantize_q4_0_conv.py).
 * @return Output symbolic tensor after conv+SiLU
 */
inline Tensor convBnSilu(const std::string &name, int in_ch, int out_ch, int k,
                         int stride, int padding, Tensor input,
                         bool conv_q40 = false) {
  std::vector<std::string> conv_props = {
    nntrainer::withKey("name", name + "/conv"),
    nntrainer::withKey("kernel_size", {k, k}),
    nntrainer::withKey("filters", out_ch),
    nntrainer::withKey("stride", {stride, stride}),
    nntrainer::withKey("padding", padding)};
  if (out_ch > 1 && out_ch % 32 == 0 && (in_ch * k * k) % 32 == 0) {
    if (conv_q40)
      conv_props.push_back(nntrainer::withKey("weight_dtype", "Q4_0"));
    if (quantConvSink() != nullptr)
      quantConvSink()->push_back(name + "/conv");
  }
  LayerHandle conv(createLayer("conv2d", conv_props));
  auto h = conv(input);

  LayerHandle act(
    createLayer("activation", {nntrainer::withKey("name", name + "/act"),
                               nntrainer::withKey("activation", "swish")}));
  return act(h);
}

/**
 * @brief Build a Bottleneck sub-graph (cv1 3x3 + cv2 3x3 + residual add).
 */
inline Tensor bottleneck(const std::string &name, int in_ch, int hidden_ch,
                         int out_ch, Tensor input, bool conv_q40 = false) {
  auto h =
    convBnSilu(name + "/cv1", in_ch, hidden_ch, 3, 1, 1, input, conv_q40);
  h = convBnSilu(name + "/cv2", hidden_ch, out_ch, 3, 1, 1, h, conv_q40);
  LayerHandle add(
    createLayer("Addition", {nntrainer::withKey("name", name + "/add")}));
  return add({h, input});
}

/**
 * @brief Build a C3k sub-graph.
 */
inline Tensor c3kBlock(const std::string &name, int in_ch, int inner_ch,
                       int out_ch, Tensor input, bool conv_q40 = false) {
  auto inner =
    convBnSilu(name + "/cv1", in_ch, inner_ch, 1, 1, 0, input, conv_q40);
  inner =
    bottleneck(name + "/inner0", inner_ch, inner_ch, inner_ch, inner, conv_q40);
  inner =
    bottleneck(name + "/inner1", inner_ch, inner_ch, inner_ch, inner, conv_q40);
  auto skip =
    convBnSilu(name + "/cv2", in_ch, inner_ch, 1, 1, 0, input, conv_q40);

  LayerHandle cat(
    createLayer("concat", {nntrainer::withKey("name", name + "/cat"),
                           nntrainer::withKey("axis", 1)}));
  auto concat_out = cat({inner, skip});

  return convBnSilu(name + "/cv3", 2 * inner_ch, out_ch, 1, 1, 0, concat_out,
                    conv_q40);
}

/**
 * @brief Build a C3k2 sub-graph.
 */
inline Tensor c3k2Block(const std::string &name, int in_ch, int out_ch, int c,
                        bool c3k, Tensor input, bool conv_q40 = false) {
  auto y = convBnSilu(name + "/cv1", in_ch, 2 * c, 1, 1, 0, input, conv_q40);

  LayerHandle sliceA(
    createLayer("slice", {nntrainer::withKey("name", name + "/slice_a"),
                          nntrainer::withKey("axis", 1),
                          nntrainer::withKey("start_index", 1),
                          nntrainer::withKey("end_index", c + 1)}));
  auto y_a = sliceA(y);

  LayerHandle sliceB(
    createLayer("slice", {nntrainer::withKey("name", name + "/slice_b"),
                          nntrainer::withKey("axis", 1),
                          nntrainer::withKey("start_index", c + 1),
                          nntrainer::withKey("end_index", 2 * c + 1)}));
  auto y_b = sliceB(y);

  Tensor y_c;
  if (c3k) {
    y_c = c3kBlock(name + "/m0", c, c / 2, c, y_b, conv_q40);
  } else {
    y_c = bottleneck(name + "/m0", c, c / 2, c, y_b, conv_q40);
  }

  LayerHandle cat(
    createLayer("concat", {nntrainer::withKey("name", name + "/cat"),
                           nntrainer::withKey("axis", 1)}));
  auto concat_out = cat({y_a, y_b, y_c});

  return convBnSilu(name + "/cv2", 3 * c, out_ch, 1, 1, 0, concat_out,
                    conv_q40);
}

/**
 * @brief Build a (BN-fused) Conv2d sub-graph block (NO activation).
 *
 * @param conv_q40  If true, eligible convs get weight_dtype=Q4_0.
 */
inline Tensor convBnOnly(const std::string &name, int in_ch, int out_ch, int k,
                         int stride, int padding, Tensor input,
                         bool conv_q40 = false) {
  std::vector<std::string> conv_props = {
    nntrainer::withKey("name", name + "/conv"),
    nntrainer::withKey("kernel_size", {k, k}),
    nntrainer::withKey("filters", out_ch),
    nntrainer::withKey("stride", {stride, stride}),
    nntrainer::withKey("padding", padding)};
  if (out_ch > 1 && out_ch % 32 == 0 && (in_ch * k * k) % 32 == 0) {
    if (conv_q40)
      conv_props.push_back(nntrainer::withKey("weight_dtype", "Q4_0"));
    if (quantConvSink() != nullptr)
      quantConvSink()->push_back(name + "/conv");
  }
  LayerHandle conv(createLayer("conv2d", conv_props));
  return conv(input);
}

/**
 * @brief Build a MaxPool2d layer with explicit padding.
 */
inline Tensor maxPool(const std::string &name, int k, Tensor input) {
  int p = k / 2;
  std::string pad_str = std::to_string(p) + "," + std::to_string(p) + "," +
                        std::to_string(p) + "," + std::to_string(p);
  LayerHandle pool(
    createLayer("pooling2d", {nntrainer::withKey("name", name),
                              nntrainer::withKey("pooling", "max"),
                              nntrainer::withKey("pool_size", {k, k}),
                              nntrainer::withKey("stride", {1, 1}),
                              nntrainer::withKey("padding", pad_str)}));
  return pool(input);
}

/**
 * @brief Build an SPPF sub-graph.
 */
inline Tensor sppfBlock(const std::string &name, int in_ch, Tensor input,
                        bool conv_q40 = false) {
  int half = in_ch / 2;
  auto x = convBnOnly(name + "/cv1", in_ch, half, 1, 1, 0, input, conv_q40);

  auto p1 = maxPool(name + "/pool1", 5, x);
  auto p2 = maxPool(name + "/pool2", 5, p1);
  auto p3 = maxPool(name + "/pool3", 5, p2);

  LayerHandle cat(
    createLayer("concat", {nntrainer::withKey("name", name + "/cat"),
                           nntrainer::withKey("axis", 1)}));
  auto concat_out = cat({x, p1, p2, p3});

  return convBnSilu(name + "/cv2", half * 4, in_ch, 1, 1, 0, concat_out,
                    conv_q40);
}

// ===== Detect head graph builders =====

/**
 * @brief 1x1 Conv2d with bias, no BN, no activation (detect output conv).
 *
 * @param conv_q40  If true, eligible 1x1 convs get weight_dtype=Q4_0.
 *                  Note: out_ch for box branch is 64 (divisible by 32) but
 *                  in_ch (64) * 1 * 1 = 64 is also divisible by 32, so it
 *                  can be Q4_0. cls branch out_ch=1 is NOT eligible
 *                  (out_ch>1 check).
 */
inline Tensor convBias1x1(const std::string &name, int out_ch, int in_ch,
                          Tensor input, bool conv_q40 = false) {
  std::vector<std::string> conv_props = {
    nntrainer::withKey("name", name + "/conv"),
    nntrainer::withKey("kernel_size", {1, 1}),
    nntrainer::withKey("filters", out_ch), nntrainer::withKey("stride", {1, 1}),
    nntrainer::withKey("padding", 0)};
  if (out_ch > 1 && out_ch % 32 == 0 && (in_ch * 1 * 1) % 32 == 0) {
    if (conv_q40)
      conv_props.push_back(nntrainer::withKey("weight_dtype", "Q4_0"));
    if (quantConvSink() != nullptr)
      quantConvSink()->push_back(name + "/conv");
  }
  LayerHandle conv(createLayer("conv2d", conv_props));
  return conv(input);
}

/**
 * @brief Depthwise 3x3 (pad 1) + SiLU, BN folded. Always FP32 (never Q4_0).
 */
inline Tensor dwConvBnSilu(const std::string &name, int ch, Tensor input) {
  LayerHandle dw(createLayer(
    "conv2d",
    {nntrainer::withKey("name", name + "/dw"),
     nntrainer::withKey("kernel_size", {3, 3}),
     nntrainer::withKey("filters", ch), nntrainer::withKey("groups", ch),
     nntrainer::withKey("stride", {1, 1}), nntrainer::withKey("padding", 1)}));
  auto h = dw(input);
  LayerHandle act(
    createLayer("activation", {nntrainer::withKey("name", name + "/act"),
                               nntrainer::withKey("activation", "swish")}));
  return act(h);
}

/**
 * @brief Depthwise 3x3 (pad 1), NO activation. Always FP32 (never Q4_0).
 */
inline Tensor dwConvBnOnly(const std::string &name, int ch, Tensor input) {
  LayerHandle dw(createLayer(
    "conv2d",
    {nntrainer::withKey("name", name + "/dw"),
     nntrainer::withKey("kernel_size", {3, 3}),
     nntrainer::withKey("filters", ch), nntrainer::withKey("groups", ch),
     nntrainer::withKey("stride", {1, 1}), nntrainer::withKey("padding", 1)}));
  return dw(input);
}

struct YOLOv11Params {
  std::string scale;
  int nc;

  // Backbone channels
  int c0, c1, c2, c3, m4_in, m4_out, c5, m6_in, m6_out, c7, m8_in, m8_out;
  int m10_c;

  // Backbone C3k2/C2PSA configs
  bool m2_c3k, m4_c3k, m6_c3k, m8_c3k;
  int m2_inner_c, m4_inner_c, m6_inner_c, m8_inner_c;

  // Head channels
  int m13_in, m13_out, m13_c;
  int m16_in, m16_out, m16_c;
  int m17_out;
  int m19_in, m19_out, m19_c;
  int m20_out;
  int m22_in, m22_out, m22_c;

  bool m13_c3k, m16_c3k, m19_c3k, m22_c3k;

  // Detect head channels
  int det0_pi, det1_pi, det2_pi;
};

inline YOLOv11Params getParams(const std::string &scale, int nc = 1) {
  YOLOv11Params p;
  p.scale = scale;
  p.nc = nc;

  if (scale == "s" || scale == "S") {
    p.c0 = 32;
    p.c1 = 64;
    p.c2 = 128;
    p.c3 = 128;
    p.m4_in = 128;
    p.m4_out = 256;
    p.c5 = 256;
    p.m6_in = 256;
    p.m6_out = 256;
    p.c7 = 512;
    p.m8_in = 512;
    p.m8_out = 512;

    p.m2_c3k = false;
    p.m2_inner_c = 32;
    p.m4_c3k = false;
    p.m4_inner_c = 64;
    p.m6_c3k = true;
    p.m6_inner_c = 128;
    p.m8_c3k = true;
    p.m8_inner_c = 256;

    p.m10_c = 512;

    p.m13_in = 768;
    p.m13_out = 256;
    p.m13_c = 128;
    p.m13_c3k = false;
    p.m16_in = 512;
    p.m16_out = 128;
    p.m16_c = 64;
    p.m16_c3k = false;
    p.m17_out = 128;
    p.m19_in = 384;
    p.m19_out = 256;
    p.m19_c = 128;
    p.m19_c3k = false;
    p.m20_out = 256;
    p.m22_in = 768;
    p.m22_out = 512;
    p.m22_c = 256;
    p.m22_c3k = true;

    p.det0_pi = 128;
    p.det1_pi = 256;
    p.det2_pi = 512;
  } else {
    // Default to medium ("m")
    p.c0 = 64;
    p.c1 = 128;
    p.c2 = 256;
    p.c3 = 256;
    p.m4_in = 256;
    p.m4_out = 512;
    p.c5 = 512;
    p.m6_in = 512;
    p.m6_out = 512;
    p.c7 = 512;
    p.m8_in = 512;
    p.m8_out = 512;

    p.m2_c3k = true;
    p.m2_inner_c = 64;
    p.m4_c3k = true;
    p.m4_inner_c = 128;
    p.m6_c3k = true;
    p.m6_inner_c = 256;
    p.m8_c3k = true;
    p.m8_inner_c = 256;

    p.m10_c = 512;

    p.m13_in = 1024;
    p.m13_out = 512;
    p.m13_c = 256;
    p.m13_c3k = true;
    p.m16_in = 1024;
    p.m16_out = 256;
    p.m16_c = 128;
    p.m16_c3k = true;
    p.m17_out = 256;
    p.m19_in = 768;
    p.m19_out = 512;
    p.m19_c = 256;
    p.m19_c3k = true;
    p.m20_out = 512;
    p.m22_in = 1024;
    p.m22_out = 512;
    p.m22_c = 256;
    p.m22_c3k = true;

    p.det0_pi = 256;
    p.det1_pi = 512;
    p.det2_pi = 512;
  }

  return p;
}

/**
 * @brief Build one Detect scale -> raw logits [1, 64+nc, H, W].
 *
 * @param s       name prefix (e.g. "det0")
 * @param pi_ch   input feature channels (256 for P3, 512 for P4/P5)
 * @param c3      classification branch channels (max(ch[0], nc))
 * @param nc      number of classes
 * @param in      input feature map
 * @param conv_q40  If true, eligible convs use Q4_0.
 */
inline Tensor detectScale(const std::string &s, int pi_ch, int c3, int nc,
                          Tensor in, bool conv_q40 = false) {
  // box branch (cv2): pi_ch->64->64->64 (1x1 output)
  auto x = convBnSilu(s + "/cv2_0", pi_ch, 64, 3, 1, 1, in, conv_q40);
  x = convBnSilu(s + "/cv2_1", 64, 64, 3, 1, 1, x, conv_q40);
  auto box = convBias1x1(s + "/cv2_2", 64, 64, x, conv_q40);

  // cls branch (cv3): depthwise-separable x2 then 1x1 (out_ch=nc, never Q4_0)
  auto c = dwConvBnSilu(s + "/cv3_0_dw", pi_ch, in);
  c = convBnSilu(s + "/cv3_0_pw", pi_ch, c3, 1, 1, 0, c, conv_q40);
  c = dwConvBnSilu(s + "/cv3_1_dw", c3, c);
  c = convBnSilu(s + "/cv3_1_pw", c3, c3, 1, 1, 0, c, conv_q40);
  auto cls = convBias1x1(s + "/cv3_2", nc, c3, c, conv_q40); // out_ch=nc: FP32

  LayerHandle cat(createLayer("concat", {nntrainer::withKey("name", s + "/out"),
                                         nntrainer::withKey("axis", 1)}));
  return cat({box, cls});
}

// ===== Utility builders used by C2PSA and head =====

/** @brief Channel-axis slice [start0, end0) — slice layer uses 1-indexed. */
inline Tensor sliceCh(const std::string &name, int start0, int end0,
                      Tensor in) {
  LayerHandle s(createLayer(
    "slice", {nntrainer::withKey("name", name), nntrainer::withKey("axis", 1),
              nntrainer::withKey("start_index", start0 + 1),
              nntrainer::withKey("end_index", end0 + 1)}));
  return s(in);
}

/** @brief Elementwise addition of two tensors. */
inline Tensor addT(const std::string &name, Tensor a, Tensor b) {
  LayerHandle l(createLayer("Addition", {nntrainer::withKey("name", name)}));
  return l({a, b});
}

/** @brief Upsample (nearest, x2). */
inline Tensor upsampleX2(const std::string &name, Tensor in) {
  LayerHandle l(
    createLayer("upsample2d", {nntrainer::withKey("name", name),
                               nntrainer::withKey("upsample", "nearest"),
                               nntrainer::withKey("kernel_size", "2,2")}));
  return l(in);
}

/** @brief Channel-axis concat. */
inline Tensor concatCh(const std::string &name,
                       const std::vector<Tensor> &ins) {
  LayerHandle l(createLayer("concat", {nntrainer::withKey("name", name),
                                       nntrainer::withKey("axis", 1)}));
  return l(ins);
}

/**
 * @brief Build the C2PSA block (model.10) from standard layers + psa_attention.
 *
 * @param conv_q40  If true, eligible group=1 convs use Q4_0.
 */
inline Tensor buildC2PSA(const std::string &n, int ch, Tensor x,
                         bool conv_q40 = false) {
  auto cv1 = yolov11::convBnSilu(n + "/cv1", ch, ch, 1, 1, 0, x, conv_q40);
  auto a = sliceCh(n + "/slice_a", 0, ch / 2, cv1);
  auto b = sliceCh(n + "/slice_b", ch / 2, ch, cv1);

  auto qkv = yolov11::convBnOnly(n + "/qkv", ch / 2, ch, 1, 1, 0, b, conv_q40);

  std::vector<Tensor> v_parts;
  int head_dim = ch / 4;
  int v_dim = head_dim / 2;
  for (int h = 0; h < 4; ++h)
    v_parts.push_back(sliceCh(n + "/slice_v" + std::to_string(h),
                              h * head_dim + v_dim, h * head_dim + head_dim,
                              qkv));
  LayerHandle vcat(
    createLayer("concat", {nntrainer::withKey("name", n + "/vcat"),
                           nntrainer::withKey("axis", 1)}));
  auto v = vcat(v_parts);
  auto pe = yolov11::dwConvBnOnly(n + "/pe", ch / 2, v);

  LayerHandle att(
    createLayer("psa_attention", {nntrainer::withKey("name", n + "/attn")}));
  auto attn = att(qkv);
  auto attn_pe = addT(n + "/add_pe", attn, pe);
  auto proj = yolov11::convBnOnly(n + "/proj", ch / 2, ch / 2, 1, 1, 0, attn_pe,
                                  conv_q40);
  auto b1 = addT(n + "/res1", b, proj);

  auto ffn0 =
    yolov11::convBnSilu(n + "/ffn0", ch / 2, ch, 1, 1, 0, b1, conv_q40);
  auto ffn1 =
    yolov11::convBnOnly(n + "/ffn1", ch, ch / 2, 1, 1, 0, ffn0, conv_q40);
  auto b2 = addT(n + "/res2", b1, ffn1);

  LayerHandle cat(createLayer("concat", {nntrainer::withKey("name", n + "/cat"),
                                         nntrainer::withKey("axis", 1)}));
  auto cc = cat({a, b2});
  return yolov11::convBnSilu(n + "/cv2", ch, ch, 1, 1, 0, cc, conv_q40);
}

// ===== Top-level whole-network builders =====

/**
 * @brief Build the backbone (model.0-10) and return m10 (C2PSA output).
 *
 * Also outputs m4 and m6 (FPN skip connections).
 *
 * @param xIn       Input symbolic tensor [1, 3, imgsz, imgsz]
 * @param m4_out    [out] C3k2 block 4 output (P3 skip)
 * @param m6_out    [out] C3k2 block 6 output (P4 skip)
 * @param p         YOLOv11 scaling and layout parameters
 * @param conv_q40  If true, eligible convs use Q4_0.
 * @return m10 (C2PSA output, P5 feature)
 */
inline Tensor buildBackbone(Tensor xIn, Tensor &m4_out, Tensor &m6_out,
                            const YOLOv11Params &p, bool conv_q40 = false) {
  auto h = yolov11::convBnSilu("conv0", 3, p.c0, 3, 2, 1, xIn, conv_q40);
  h = yolov11::convBnSilu("conv1", p.c0, p.c1, 3, 2, 1, h, conv_q40);
  h = yolov11::c3k2Block("m2", p.c1, p.c2, p.m2_inner_c, p.m2_c3k, h, conv_q40);
  h = yolov11::convBnSilu("conv3", p.c2, p.c3, 3, 2, 1, h, conv_q40);
  m4_out = yolov11::c3k2Block("m4", p.c3, p.m4_out, p.m4_inner_c, p.m4_c3k, h,
                              conv_q40);
  h = yolov11::convBnSilu("conv5", p.m4_out, p.c5, 3, 2, 1, m4_out, conv_q40);
  m6_out = yolov11::c3k2Block("m6", p.c5, p.m6_out, p.m6_inner_c, p.m6_c3k, h,
                              conv_q40);
  h = yolov11::convBnSilu("conv7", p.m6_out, p.c7, 3, 2, 1, m6_out, conv_q40);
  h = yolov11::c3k2Block("m8", p.c7, p.m8_out, p.m8_inner_c, p.m8_c3k, h,
                         conv_q40);
  h = yolov11::sppfBlock("m9", p.m8_out, h, conv_q40);
  return buildC2PSA("m10", p.m10_c, h, conv_q40);
}

/**
 * @brief Build the FPN head (model.11-22) and 3-scale Detect head (model.23).
 *
 * @param m4       C3k2 block 4 output (P3 skip from backbone)
 * @param m6       C3k2 block 6 output (P4 skip from backbone)
 * @param m10      C2PSA output (P5 feature from backbone)
 * @param p        YOLOv11 scaling and layout parameters
 * @param conv_q40 If true, eligible convs use Q4_0.
 * @return {P3, P4, P5} raw detection logits [1, 64+nc, H, W] each
 */
inline std::vector<Tensor> buildHead(Tensor m4, Tensor m6, Tensor m10,
                                     const YOLOv11Params &p,
                                     bool conv_q40 = false) {
  auto m11 = upsampleX2("m11", m10);
  auto m12 = concatCh("m12", {m11, m6});
  auto m13 = yolov11::c3k2Block("m13", p.m13_in, p.m13_out, p.m13_c, p.m13_c3k,
                                m12, conv_q40);

  auto m14 = upsampleX2("m14", m13);
  auto m15 = concatCh("m15", {m14, m4});
  auto m16 = yolov11::c3k2Block("m16", p.m16_in, p.m16_out, p.m16_c, p.m16_c3k,
                                m15, conv_q40);

  auto m17 =
    yolov11::convBnSilu("m17", p.m16_out, p.m17_out, 3, 2, 1, m16, conv_q40);
  auto m18 = concatCh("m18", {m17, m13});
  auto m19 = yolov11::c3k2Block("m19", p.m19_in, p.m19_out, p.m19_c, p.m19_c3k,
                                m18, conv_q40);

  auto m20 =
    yolov11::convBnSilu("m20", p.m19_out, p.m20_out, 3, 2, 1, m19, conv_q40);
  auto m21 = concatCh("m21", {m20, m10});
  auto m22 = yolov11::c3k2Block("m22", p.m22_in, p.m22_out, p.m22_c, p.m22_c3k,
                                m21, conv_q40);

  return {detectScale("det0", p.det0_pi, p.det0_pi, p.nc, m16, conv_q40),
          detectScale("det1", p.det1_pi, p.det0_pi, p.nc, m19, conv_q40),
          detectScale("det2", p.det2_pi, p.det0_pi, p.nc, m22, conv_q40)};
}

} // namespace yolov11

#endif // __YOLOV11_GRAPH_H__

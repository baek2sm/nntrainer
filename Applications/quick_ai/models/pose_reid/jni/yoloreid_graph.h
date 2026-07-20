// SPDX-License-Identifier: Apache-2.0
/**
 * @file   yoloreid_graph.h
 * @brief  YOLOv7ReIDtiny graph block builders (backbone + heads)
 */
#ifndef __YOLOREID_GRAPH_H__
#define __YOLOREID_GRAPH_H__

#include <layer.h>
#include <model.h>
#include <string>
#include <tensor_api.h>
#include <tensor_dim.h>
#include <util_func.h>
#include <vector>

namespace yoloreid {

using Tensor = ml::train::Tensor;
using LayerHandle = ml::train::LayerHandle;

// Helper to create layers
inline LayerHandle
createLayer(const std::string &type,
            const std::vector<std::string> &properties = {}) {
  return ml::train::createLayer(type, properties);
}

// 1x1 or 3x3 Conv Only
inline Tensor convOnly(const std::string &name, int in_ch, int out_ch, int k,
                       int stride, int padding, int groups, Tensor input) {
  std::vector<std::string> props = {
    nntrainer::withKey("name", name + "/conv"),
    nntrainer::withKey("kernel_size", {k, k}),
    nntrainer::withKey("filters", out_ch),
    nntrainer::withKey("stride", {stride, stride}),
    nntrainer::withKey("padding", padding),
    nntrainer::withKey("groups", groups)};
  return createLayer("conv2d", props)(input);
}

// Conv + BN + SiLU (activation="swish")
inline Tensor convSiLU(const std::string &name, int in_ch, int out_ch, int k,
                       int stride, int padding, int groups, Tensor input) {
  std::vector<std::string> props = {
    nntrainer::withKey("name", name + "/conv"),
    nntrainer::withKey("kernel_size", {k, k}),
    nntrainer::withKey("filters", out_ch),
    nntrainer::withKey("stride", {stride, stride}),
    nntrainer::withKey("padding", padding),
    nntrainer::withKey("groups", groups),
    nntrainer::withKey("fused_activation", "swish")};
  return createLayer("conv2d", props)(input);
}

// Concatenate multiple tensors along channel axis (axis 1)
inline Tensor concatCh(const std::string &name,
                       const std::vector<Tensor> &inputs) {
  std::vector<std::string> props = {nntrainer::withKey("name", name),
                                    nntrainer::withKey("axis", 1)};
  return createLayer("concat", props)(inputs);
}

// Slice along channel dimension (axis 1)
inline Tensor sliceCh(const std::string &name, int start, int end,
                      Tensor input) {
  std::vector<std::string> props = {
    nntrainer::withKey("name", name), nntrainer::withKey("axis", 1),
    nntrainer::withKey("start_index",
                       start + 1), // 1-indexed in nntrainer slice
    nntrainer::withKey("end_index", end)};
  return createLayer("slice", props)(input);
}

// Slice along width dimension (axis 3)
inline Tensor sliceWidth(const std::string &name, int start, int end,
                         Tensor input) {
  std::vector<std::string> props = {
    nntrainer::withKey("name", name), nntrainer::withKey("axis", 3),
    nntrainer::withKey("start_index", start + 1),
    nntrainer::withKey("end_index", end)};
  return createLayer("slice", props)(input);
}

// ELAN Block (YOLOv7-tiny style)
inline Tensor elanBlock(const std::string &name, int c_in, int c_out,
                        int c_bottleneck, int n_blocks, Tensor input) {
  auto x1 =
    convSiLU(name + "/elan/conv1", c_in, c_bottleneck, 1, 1, 0, 1, input);
  auto x2 =
    convSiLU(name + "/elan/conv2", c_in, c_bottleneck, 1, 1, 0, 1, input);

  std::vector<Tensor> cat_inputs = {x1, x2};
  auto prev = x2;
  for (int i = 0; i < n_blocks; ++i) {
    auto next = convSiLU(name + "/elan/conv_blocks/" + std::to_string(i),
                         c_bottleneck, c_bottleneck, 3, 1, 1, 1, prev);
    cat_inputs.push_back(next);
    prev = next;
  }

  // Reverse list order
  std::vector<Tensor> cat_inputs_rev(cat_inputs.rbegin(), cat_inputs.rend());
  auto cat = concatCh(name + "/elan/cat", cat_inputs_rev);
  return convSiLU(name + "/elan/last_conv", c_bottleneck * (2 + n_blocks),
                  c_out, 1, 1, 0, 1, cat);
}

// SPPCSPC Block (YOLOv7-tiny style)
inline Tensor sppcspcBlock(const std::string &name, int c_in, int c_out,
                           Tensor input) {
  int c_hidden = c_out; // 256
  auto x1 = convSiLU(name + "/spp/cv1", c_in, c_hidden, 1, 1, 0, 1, input);

  // SPP pools: MaxPool2d with kernels 5, 9, 13
  auto p5 =
    createLayer("pooling2d", {nntrainer::withKey("name", name + "/spp/p5"),
                              nntrainer::withKey("pooling", "max"),
                              nntrainer::withKey("pool_size", {5, 5}),
                              nntrainer::withKey("stride", {1, 1}),
                              nntrainer::withKey("padding", "same")})(x1);
  auto p9 =
    createLayer("pooling2d", {nntrainer::withKey("name", name + "/spp/p9"),
                              nntrainer::withKey("pooling", "max"),
                              nntrainer::withKey("pool_size", {9, 9}),
                              nntrainer::withKey("stride", {1, 1}),
                              nntrainer::withKey("padding", "same")})(x1);
  auto p13 =
    createLayer("pooling2d", {nntrainer::withKey("name", name + "/spp/p13"),
                              nntrainer::withKey("pooling", "max"),
                              nntrainer::withKey("pool_size", {13, 13}),
                              nntrainer::withKey("stride", {1, 1}),
                              nntrainer::withKey("padding", "same")})(x1);

  auto cat_spp = concatCh(name + "/spp/cat_spp", {x1, p5, p9, p13});
  auto cv3 =
    convSiLU(name + "/spp/cv3", c_hidden * 4, c_hidden, 1, 1, 0, 1, cat_spp);
  auto cv2 = convSiLU(name + "/spp/cv2", c_in, c_hidden, 1, 1, 0, 1, input);

  auto cat_all = concatCh(name + "/spp/cat_all", {cv3, cv2});
  return convSiLU(name + "/spp/cv4", c_hidden * 2, c_out * 2, 1, 1, 0, 1,
                  cat_all);
}

// Downsample block
// Downsample block
inline Tensor downsampleBlock(const std::string &name, int c_in, int c_out,
                              Tensor input, Tensor route) {
  auto x = convSiLU(name + "/base/conv", c_in, c_out / 2, 3, 2, 1, 1, input);
  std::vector<Tensor> cat_inputs = {x, route};
  return concatCh(name + "/cat", cat_inputs);
}

// Upsample block
inline Tensor upsampleBlock(const std::string &name, int c_in, int c_out,
                            int route_ch, Tensor input, Tensor route) {
  auto x1 = convSiLU(name + "/base/conv1", c_in, c_out / 2, 1, 1, 0, 1, input);
  auto up =
    createLayer("upsample2d", {nntrainer::withKey("name", name + "/up"),
                               nntrainer::withKey("kernel_size", {2, 2})})(x1);
  auto x2 =
    convSiLU(name + "/base/conv2", route_ch, c_out / 2, 1, 1, 0, 1, route);
  std::vector<Tensor> cat_inputs = {x2, up};
  return concatCh(name + "/cat", cat_inputs);
}

// Backbone Graph Builder
inline std::vector<Tensor> buildBackbone(Tensor xIn) {
  // Stem0: Conv + BN + SiLU
  auto x = convSiLU("backbone/blocks/0/0", 3, 48, 3, 2, 1, 1,
                    xIn); // -> [1,48,160,160]

  // Stage 0 -> Stem1 + ELAN
  auto stem1_base = convSiLU("backbone/blocks/1/base", 48, 96, 3, 2, 1, 1,
                             x); // -> [1,96,80,80]
  auto stem1_elan = elanBlock("backbone/blocks/1", 96, 96, 48, 2,
                              stem1_base); // -> [1,96,80,80]

  // Stage 1 -> MP + ELAN
  auto stem2_mp = createLayer(
    "pooling2d",
    {nntrainer::withKey("name", "backbone/blocks/2/mp"),
     nntrainer::withKey("pooling", "max"),
     nntrainer::withKey("pool_size", {2, 2}),
     nntrainer::withKey("stride", {2, 2})})(stem1_elan); // -> [1,96,40,40]
  auto stem2_elan = elanBlock("backbone/blocks/2", 96, 192, 96, 2,
                              stem2_mp); // -> [1,192,40,40]

  // Stage 2 -> MP + ELAN
  auto stem3_mp = createLayer(
    "pooling2d",
    {nntrainer::withKey("name", "backbone/blocks/3/mp"),
     nntrainer::withKey("pooling", "max"),
     nntrainer::withKey("pool_size", {2, 2}),
     nntrainer::withKey("stride", {2, 2})})(stem2_elan); // -> [1,192,20,20]
  auto stem3_elan = elanBlock("backbone/blocks/3", 192, 384, 192, 2,
                              stem3_mp); // -> [1,384,20,20]

  // Stage 3 -> MP + ELAN
  auto stem4_mp = createLayer(
    "pooling2d",
    {nntrainer::withKey("name", "backbone/blocks/4/mp"),
     nntrainer::withKey("pooling", "max"),
     nntrainer::withKey("pool_size", {2, 2}),
     nntrainer::withKey("stride", {2, 2})})(stem3_elan); // -> [1,384,10,10]
  auto stem4_elan = elanBlock("backbone/blocks/4", 384, 768, 384, 2,
                              stem4_mp); // -> [1,768,10,10]

  // Returns backbone endpoints at multiple scales
  return {stem2_elan, stem3_elan, stem4_elan};
}

// FPN Features Neck Builder (Stage 3 -> 1 output endpoint)
inline Tensor buildFeatureFPN(const std::string &prefix,
                              const std::vector<Tensor> &backbone_feats) {
  auto stem2_elan = backbone_feats[0]; // [1,192,40,40]
  auto stem3_elan = backbone_feats[1]; // [1,384,20,20]
  auto stem4_elan = backbone_feats[2]; // [1,768,10,10]

  // SPPCSPC neck
  auto spp =
    sppcspcBlock(prefix + "/spp", 768, 256, stem4_elan); // -> [1,512,10,10]

  // Upsample Stage 2
  auto up0 = upsampleBlock(prefix + "/feature_up/0", 512, 256, 384, spp,
                           stem3_elan); // -> [1,256,20,20]
  auto up0_elan = elanBlock(prefix + "/feature_up/0", 256, 256, 128, 2,
                            up0); // -> [1,256,20,20]

  // Upsample Stage 1
  auto up1 = upsampleBlock(prefix + "/feature_up/1", 256, 128, 192, up0_elan,
                           stem2_elan); // -> [1,128,40,40]
  auto up1_elan = elanBlock(prefix + "/feature_up/1", 128, 128, 64, 2,
                            up1); // -> [1,128,40,40]

  // Downsample Neck Stage 1
  auto down0 = downsampleBlock(prefix + "/feature_down/0", 128, 256, up1_elan,
                               up0_elan); // -> [1,256,20,20]
  auto down0_elan = elanBlock(prefix + "/feature_down/0", 384, 256, 128, 2,
                              down0); // -> [1,256,20,20]

  // Downsample Neck Stage 2
  auto down1 = downsampleBlock(prefix + "/feature_down/1", 256, 512, down0_elan,
                               spp); // -> [1,512,10,10]
  auto down1_elan = elanBlock(prefix + "/feature_down/1", 768, 512, 256, 2,
                              down1); // -> [1,512,10,10]

  // Ends
  return convSiLU(prefix + "/ends/0", 512, 1024, 3, 1, 1, 1,
                  down1_elan); // -> [1,1024,10,10]
}

// RTMCC keypoint prediction head
inline Tensor buildRTMCCHead(const std::string &name, Tensor input, int nkpt) {
  // final_layer: Conv2d(1024 -> nkpt, k7, s1, p3)
  auto final_conv = convOnly(name + "/final_layer", 1024, nkpt, 7, 1, 3, 1,
                             input); // -> [1, nkpt, 10, 10]

  // Reshape spatial dims: [1, nkpt, 10, 10] -> [1, nkpt, 1, 100]
  auto flat = createLayer(
    "reshape", {nntrainer::withKey("name", name + "/flat"),
                nntrainer::withKey("target_shape", std::to_string(nkpt) +
                                                     ":1:100")})(final_conv);

  // mlp: Linear(100 -> 256)
  // We model mlp as fully_connected with unit=256
  auto mlp = createLayer(
    "fully_connected",
    {nntrainer::withKey("name", name + "/mlp"), nntrainer::withKey("unit", 256),
     nntrainer::withKey("bias", false)})(flat); // -> [1, nkpt, 1, 256]

  // Gated Attention Unit (GAU)
  // uv: Linear(256 -> 512) + SiLU
  auto uv = createLayer(
    "fully_connected",
    {nntrainer::withKey("name", name + "/gau_proj"),
     nntrainer::withKey("unit", 512), nntrainer::withKey("bias", false),
     nntrainer::withKey("activation", "swish")})(mlp); // -> [1, nkpt, 1, 512]

  // Slice uv into u and v (each 256 channel elements along width axis 3!)
  auto u =
    sliceWidth(name + "/gau_slice_u", 0, 256, uv); // -> [1, nkpt, 1, 256]
  auto v =
    sliceWidth(name + "/gau_slice_v", 256, 512, uv); // -> [1, nkpt, 1, 256]

  // Elementwise multiplication: u * v
  std::vector<Tensor> mul_inputs = {u, v};
  auto u_v =
    createLayer("multiply", {nntrainer::withKey("name", name + "/gau_mul")})(
      mul_inputs); // -> [1, nkpt, 1, 256]

  // o: Linear(256 -> 256)
  auto gau_out = createLayer("fully_connected",
                             {nntrainer::withKey("name", name + "/gau_out"),
                              nntrainer::withKey("unit", 256),
                              nntrainer::withKey("bias", false)})(
    u_v); // -> [1, nkpt, 1, 256]

  // cls_x: Linear(256 -> 640)
  auto cls_x =
    createLayer("fully_connected", {nntrainer::withKey("name", name + "/cls_x"),
                                    nntrainer::withKey("unit", 640),
                                    nntrainer::withKey("bias", false)})(
      gau_out); // -> [1, nkpt, 1, 640]

  // cls_y: Linear(256 -> 640)
  auto cls_y =
    createLayer("fully_connected", {nntrainer::withKey("name", name + "/cls_y"),
                                    nntrainer::withKey("unit", 640),
                                    nntrainer::withKey("bias", false)})(
      gau_out); // -> [1, nkpt, 1, 640]

  // Concatenate cls_x and cls_y: [1, 2 * nkpt, 1, 640] (axis 1)
  std::vector<Tensor> concat_inputs = {cls_x, cls_y};
  auto concat_cls = concatCh(name + "/concat_cls", concat_inputs);

  return concat_cls;
}

// ReID embedding prediction head
inline Tensor buildReIDHead(const std::string &name, Tensor input,
                            int embed_dim) {
  // Global average pool: [1, 1024, 10, 10] -> [1, 1024, 1, 1]
  LayerHandle gap_h(
    createLayer("reduce_mean", {nntrainer::withKey("name", name + "/gap_h"),
                                nntrainer::withKey("axis", 2)}));
  LayerHandle gap_w(
    createLayer("reduce_mean", {nntrainer::withKey("name", name + "/gap_w"),
                                nntrainer::withKey("axis", 3)}));
  auto pooled = gap_w(gap_h(input));

  // Reshape to [1, 1, 1, 1024] so that fully_connected layer acts on the last
  // dimension
  auto flat = createLayer(
    "reshape", {nntrainer::withKey("name", name + "/flat"),
                nntrainer::withKey("target_shape", "1:1:1024")})(pooled);

  // fc: Linear(1024 -> embed_dim)
  auto fc =
    createLayer("fully_connected",
                {nntrainer::withKey("name", name + "/fc"),
                 nntrainer::withKey("unit", embed_dim),
                 nntrainer::withKey("bias", false)})(flat); // -> [1, 128]

  // L2 normalization: fc / sqrt(sum(fc^2) + eps)
  // Standard l2-normalization layer in nntrainer, or simply return fc for
  // app-level normalization
  return fc;
}

} // namespace yoloreid

#endif // __YOLOREID_GRAPH_H__

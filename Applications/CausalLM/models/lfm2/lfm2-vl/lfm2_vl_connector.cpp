// SPDX-License-Identifier: Apache-2.0
/**
 * @file   lfm2_vl_connector.cpp
 * @date   04 June 2026
 * @brief  Pixel-unshuffle + MLP connector for LFM2-VL.
 * @author SeungBaek Hong <baek2sm@naver.com>
 * @bug    No known bugs except for NYI items
 */

#include "lfm2_vl_connector.h"

#include <cmath>
#include <fstream>
#include <stdexcept>

namespace causallm {

Lfm2VlConnector::Lfm2VlConnector(unsigned int in_features,
                                 unsigned int hidden_size,
                                 unsigned int out_features)
  : in_features_(in_features),
    hidden_size_(hidden_size),
    out_features_(out_features) {}

/* static */ float Lfm2VlConnector::gelu(float x) {
  // tanh GELU approximation (matches PyTorch gelu_pytorch_tanh)
  constexpr float kAlpha = 0.7978845608028654f; // sqrt(2/pi)
  constexpr float kBeta  = 0.044715f;
  float inner = kAlpha * (x + kBeta * x * x * x);
  return 0.5f * x * (1.0f + std::tanh(inner));
}

/* static */ std::vector<float>
Lfm2VlConnector::linearForward(const std::vector<float> &W,
                               const std::vector<float> &b,
                               const std::vector<float> &x,
                               unsigned int rows,
                               unsigned int cols) {
  std::vector<float> y(rows, 0.0f);
  for (unsigned int r = 0; r < rows; ++r) {
    float s = b[r];
    for (unsigned int c = 0; c < cols; ++c) {
      s += W[r * cols + c] * x[c];
    }
    y[r] = s;
  }
  return y;
}

void Lfm2VlConnector::loadWeights(const std::string &weight_path) {
  std::ifstream f(weight_path, std::ios::binary);
  if (!f)
    throw std::runtime_error("Lfm2VlConnector: cannot open " + weight_path);

  auto readFloats = [&](std::vector<float> &buf, size_t n) {
    buf.resize(n);
    f.read(reinterpret_cast<char *>(buf.data()),
           static_cast<std::streamsize>(n * sizeof(float)));
    if (!f)
      throw std::runtime_error("Lfm2VlConnector: unexpected EOF in " +
                               weight_path);
  };

  readFloats(fc1_weight_, static_cast<size_t>(hidden_size_) * in_features_);
  readFloats(fc1_bias_,   hidden_size_);
  readFloats(fc2_weight_, static_cast<size_t>(out_features_) * hidden_size_);
  readFloats(fc2_bias_,   out_features_);

  weights_loaded_ = true;
}

std::vector<float>
Lfm2VlConnector::forward(const std::vector<float> &x,
                         unsigned int n_patches) const {
  if (!weights_loaded_)
    throw std::runtime_error("Lfm2VlConnector: call loadWeights() first");

  std::vector<float> out;
  out.reserve(n_patches * out_features_);

  for (unsigned int p = 0; p < n_patches; ++p) {
    // Slice one patch vector
    std::vector<float> xp(x.begin() + p * in_features_,
                          x.begin() + (p + 1) * in_features_);

    // fc1 + gelu
    auto h = linearForward(fc1_weight_, fc1_bias_, xp,
                           hidden_size_, in_features_);
    for (auto &v : h)
      v = gelu(v);

    // fc2
    auto y = linearForward(fc2_weight_, fc2_bias_, h,
                           out_features_, hidden_size_);
    out.insert(out.end(), y.begin(), y.end());
  }
  return out;
}

std::vector<float> pixelUnshuffle(const std::vector<float> &features,
                                  unsigned int n_patches,
                                  unsigned int embed_dim,
                                  unsigned int patch_h,
                                  unsigned int patch_w,
                                  unsigned int factor) {
  if (patch_h % factor != 0 || patch_w % factor != 0)
    throw std::invalid_argument(
      "pixelUnshuffle: patch grid dimensions must be divisible by factor");

  unsigned int out_h   = patch_h / factor;
  unsigned int out_w   = patch_w / factor;
  unsigned int out_n   = out_h * out_w;
  unsigned int out_dim = embed_dim * factor * factor;

  std::vector<float> out(static_cast<size_t>(out_n) * out_dim, 0.0f);

  for (unsigned int oh = 0; oh < out_h; ++oh) {
    for (unsigned int ow = 0; ow < out_w; ++ow) {
      unsigned int out_idx = oh * out_w + ow;
      unsigned int out_offset = out_idx * out_dim;
      unsigned int ch_block = 0;
      for (unsigned int dh = 0; dh < factor; ++dh) {
        for (unsigned int dw = 0; dw < factor; ++dw) {
          unsigned int ih = oh * factor + dh;
          unsigned int iw = ow * factor + dw;
          unsigned int in_idx = ih * patch_w + iw;
          unsigned int in_offset = in_idx * embed_dim;
          for (unsigned int c = 0; c < embed_dim; ++c) {
            out[out_offset + ch_block * embed_dim + c] =
              features[in_offset + c];
          }
          ++ch_block;
        }
      }
    }
  }
  return out;
}

} // namespace causallm

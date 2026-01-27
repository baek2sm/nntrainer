// SPDX-License-Identifier: Apache-2.0
/**
 *
 * @file   llm_util.cpp
 * @brief  util functions for llm (refactored from main.cpp)
 * @date   21 August 2024
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#define STB_IMAGE_IMPLEMENTATION
#include <llm_util.hpp>

std::vector<unsigned int> generate_multi_tokens(
  float *logits, unsigned int NUM_VOCAB, unsigned int NUM_TARGET_TOKENS,
  float repetition_penalty, unsigned int *input_ids, unsigned int NUM_INPUT_IDS,
  unsigned int *bad_words_ids, unsigned int NUM_BAD_WORDS_IDS) {

  std::vector<unsigned int> outputs;

  // apply repetition penalty
  if (repetition_penalty != 1 && input_ids != nullptr && NUM_INPUT_IDS != 0) {
    applyRepetitionPenalty(logits, input_ids, NUM_INPUT_IDS,
                           repetition_penalty);
  }

  // apply bad words penalty
  if (bad_words_ids != nullptr && NUM_BAD_WORDS_IDS != 0)
    applyBadWordsPenalty(logits, bad_words_ids, NUM_BAD_WORDS_IDS);

  // Sort and generate multiple tokens
  std::vector<std::pair<unsigned int, float>> top_indices_and_logits;
  for (unsigned int i = 0; i < NUM_VOCAB; ++i) {
    top_indices_and_logits.push_back({i, logits[i]});
  }
  std::partial_sort(top_indices_and_logits.begin(),
                    top_indices_and_logits.begin() + NUM_TARGET_TOKENS,
                    top_indices_and_logits.end(),
                    [](auto &a, auto &b) { return a.second > b.second; });

  // add sampled words
  for (unsigned int i = 0; i < NUM_TARGET_TOKENS; ++i) {
    outputs.push_back(top_indices_and_logits[i].first);
  }

  return outputs;
}

void applyRepetitionPenalty(float *logits, unsigned int *input_ids,
                            unsigned int NUM_INPUT_IDS,
                            float repetition_penalty) {
  for (unsigned int i = 0; i < NUM_INPUT_IDS; ++i) {
    if (logits[input_ids[i]] < 0) {
      logits[input_ids[i]] *= repetition_penalty;
    } else {
      logits[input_ids[i]] /= repetition_penalty;
    }
  }
}

void applyBadWordsPenalty(float *logits, unsigned int *bad_words_ids,
                          unsigned int NUM_BAD_WORDS_IDS) {
  for (unsigned int i = 0; i < NUM_BAD_WORDS_IDS; ++i) {
    logits[bad_words_ids[i]] = -INFINITY;
  }
}

/**
 * @brief Apply temperature & top-k & top-p to logits
 * @return Max logit for softmax
 */
float applyTKP(float *logits, int len, float temperature, unsigned int top_k,
               float top_p) {

  // Apply temperature & Sort logits
  std::vector<std::pair<int, float>> top_indices_and_logits;
  for (int i = 0; i < len; ++i) {
    if (temperature > 1e-5)
      logits[i] = logits[i] / temperature;
    top_indices_and_logits.push_back({i, logits[i]});
  }
  std::partial_sort(top_indices_and_logits.begin(),
                    top_indices_and_logits.begin() + 1,
                    top_indices_and_logits.end(),
                    [](auto &a, auto &b) { return a.second > b.second; });

  return top_indices_and_logits[0].second;
}

/**
 * @brief Resize image using bilinear interpolation
 */
std::vector<unsigned char> resizeImage(const unsigned char *src, int src_w,
                                       int src_h, int channels, int dst_w,
                                       int dst_h) {
  std::vector<unsigned char> dst(dst_w * dst_h * channels);

  float x_ratio = (float)src_w / (float)dst_w;
  float y_ratio = (float)src_h / (float)dst_h;

  for (int y = 0; y < dst_h; ++y) {
    for (int x = 0; x < dst_w; ++x) {
      float px = x * x_ratio;
      float py = y * y_ratio;

      int x0 = (int)std::floor(px);
      int y0 = (int)std::floor(py);
      int x1 = std::min(x0 + 1, src_w - 1);
      int y1 = std::min(y0 + 1, src_h - 1);

      float fx = px - x0;
      float fy = py - y0;

      for (int c = 0; c < channels; ++c) {
        float v00 = src[(y0 * src_w + x0) * channels + c];
        float v10 = src[(y0 * src_w + x1) * channels + c];
        float v01 = src[(y1 * src_w + x0) * channels + c];
        float v11 = src[(y1 * src_w + x1) * channels + c];

        float v0 = v00 * (1 - fx) + v10 * fx;
        float v1 = v01 * (1 - fx) + v11 * fx;
        float v = v0 * (1 - fy) + v1 * fy;

        dst[(y * dst_w + x) * channels + c] = (unsigned char)std::round(v);
      }
    }
  }

  return dst;
}

std::vector<float> loadAndPreprocessImage(const std::string &filepath,
                                          int target_width, int target_height,
                                          bool normalize) {
  int width, height, channels;

  unsigned char *data =
    stbi_load(filepath.c_str(), &width, &height, &channels, STBI_default);

  if (data == nullptr) {
    throw std::runtime_error("Failed to load image: " + filepath);
  }

  std::vector<unsigned char> rgb_data;
  std::vector<unsigned char> resized_data;

  // Resize image
  unsigned char *ptr_to_stbi_free = data;
  if (width != target_width || height != target_height) {
    resized_data =
      resizeImage(data, width, height, channels, target_width, target_height);
    data = resized_data.data();
  }

  // Convert to RGB
  unsigned char *data_to_process = data;

  if (channels == 1) {
    // Grayscale -> RGB
    rgb_data.resize(target_width * target_height * 3);
    for (int i = 0; i < target_width * target_height; ++i) {
      unsigned char val = data[i];
      rgb_data[i * 3] = val;
      rgb_data[i * 3 + 1] = val;
      rgb_data[i * 3 + 2] = val;
    }
    data_to_process = rgb_data.data();
    channels = 3;
  } else if (channels == 4) {
    // RGBA -> RGB (discard alpha)
    rgb_data.resize(target_width * target_height * 3);
    for (int i = 0; i < target_width * target_height; ++i) {
      rgb_data[i * 3] = data[i * 4];
      rgb_data[i * 3 + 1] = data[i * 4 + 1];
      rgb_data[i * 3 + 2] = data[i * 4 + 2];
    }
    data_to_process = rgb_data.data();
    channels = 3;
  } else if (channels != 3) {
    stbi_image_free(ptr_to_stbi_free);
    throw std::runtime_error("Unsupported number of channels: " +
                             std::to_string(channels));
  }

  int out_channels = 3;
  int out_width = target_width;
  int out_height = target_height;

  std::vector<float> output(out_channels * out_height * out_width);

  if (normalize) {
    for (int c = 0; c < out_channels; ++c) {
      for (int y = 0; y < out_height; ++y) {
        for (int x = 0; x < out_width; ++x) {
          unsigned char val = data_to_process[y * out_width * out_channels +
                                              x * out_channels + c];
          output[c * out_height * out_width + y * out_width + x] = val / 255.0f;
        }
      }
    }
  } else {
    for (int c = 0; c < out_channels; ++c) {
      for (int y = 0; y < out_height; ++y) {
        for (int x = 0; x < out_width; ++x) {
          unsigned char val = data_to_process[y * out_width * out_channels +
                                              x * out_channels + c];
          output[c * out_height * out_width + y * out_width + x] = (float)val;
        }
      }
    }
  }

  stbi_image_free(ptr_to_stbi_free);

  return output;
}

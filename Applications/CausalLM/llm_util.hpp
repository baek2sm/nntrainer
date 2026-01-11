// SPDX-License-Identifier: Apache-2.0
/**
 *
 * @file   llm_util.hpp
 * @brief  util functions for llm (refactored from main.cpp)
 * @date   21 August 2024
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __LLM_UTIL_HPP__
#define __LLM_UTIL_HPP__ __LLM_UTIL_HPP__

#include <algorithm> // sort
#include <math.h>    // INFINITY
#include <optional>
#include <iostream>

#include <base_properties.h>
#include <common.h>
#include <layer.h>
#include <model.h>
#include <tensor.h>
/***************** ALAIS *******************/
using LayerHandle = std::shared_ptr<ml::train::Layer>;
using ModelHandle = std::unique_ptr<ml::train::Model>;
using ml::train::createLayer;

/****************** UTIL *******************/
/**
 * @brief util functio to make "key=value" from key and value
 *
 * @tparam T type of a value
 * @param key key
 * @param value value
 * @return std::string with "key=value"
 */
template <typename T>
static std::string withKey(const std::string &key, const T &value) {
  std::stringstream ss;
  ss << key << "=" << value;
  return ss.str();
}

/**
 * @brief util function to make "key=value1,value2, ..."  from key and value

 * @tparam T type of a value
 * @param key key
 * @param value list of value
 * @return std::string with "key=value1, value, ...."
 */
template <typename T>
static std::string withKey(const std::string &key,
                           std::initializer_list<T> value) {
  if (std::empty(value)) {
    throw std::invalid_argument("empty data cannot be converted");
  }

  std::stringstream ss;
  ss << key << "=";

  auto iter = value.begin();
  for (; iter != value.end() - 1; ++iter) {
    ss << *iter << ',';
  }
  ss << *iter;

  return ss.str();
}

/**
 * @brief
 */
template <typename T>
T unwrap(std::optional<T> &&value, const std::string &error_msg) {
  if (value.has_value()) {
    return value.value();
  } else {
    throw std::runtime_error(error_msg);
  }
}

/**
 * @brief generate multi tokens from logits
 * @note This function apply repetition penalty, bad words penalty, and sort to
 * generate multiple tokens
 */
std::vector<unsigned int> generate_multi_tokens(
  float *logits, unsigned int NUM_VOCAB = 0, unsigned int NUM_TARGET_TOKENS = 1,
  float repetition_penalty = 1, unsigned int *input_ids = nullptr,
  unsigned int NUM_INPUT_IDS = 0, unsigned int *bad_words_ids = nullptr,
  unsigned int NUM_BAD_WORDS_IDS = 0);

/**
 * @brief Apply repetition penalty to logits
 */
void applyRepetitionPenalty(float *logits, unsigned int *input_ids,
                            unsigned int NUM_INPUT_IDS,
                            float repetition_penalty = 1);

/**
 * @brief Apply bad words penalty
 */
void applyBadWordsPenalty(float *logits, unsigned int *bad_words_ids,
                          unsigned int NUM_BAD_WORDS_IDS);

/**
 * @brief Apply temperature & top-k & top-p to logits
 * @return Max logit for softmax
 */
float applyTKP(float *logits, int len, float temperature, unsigned int top_k,
               float top_p);

/**
 * @brief print tensor for comparison
 */
inline void print_compare(const std::string &name, nntrainer::Tensor &tensor) {
  auto dim = tensor.getDim();
  std::cout << "[" << name << "] Shape: " << dim << std::endl;
  size_t len = tensor.size();

  // Check for (1, 1, seq_len, hidden_dim) pattern common for tokens
  // dim.height() is sequence length, dim.width() is hidden dimension
  if (dim.batch() == 1 && dim.channel() == 1 && dim.height() > 1 && dim.width() > 1) {
    unsigned int seq_len = dim.height();
    unsigned int hidden_dim = dim.width();

    if (tensor.getDataType() == ml::train::TensorDim::DataType::FP32) {
      float *data = tensor.getData<float>();
      for (unsigned int t = 0; t < seq_len; ++t) {
        float *token_data = data + t * hidden_dim;
        std::cout << "[" << name << "] Token " << t << " First 3: [";
        for (size_t i = 0; i < std::min((unsigned int)3, hidden_dim); ++i)
          std::cout << token_data[i] << " ";
        std::cout << "]" << std::endl;

        std::cout << "[" << name << "] Token " << t << " Last 3: [";
        if (hidden_dim >= 3) {
          for (size_t i = 0; i < 3; ++i)
            std::cout << token_data[hidden_dim - 3 + i] << " ";
        }
        std::cout << "]" << std::endl;
      }
    }
#ifdef ENABLE_FP16
    else if (tensor.getDataType() == ml::train::TensorDim::DataType::FP16) {
      _FP16 *data = tensor.getData<_FP16>();
      for (unsigned int t = 0; t < seq_len; ++t) {
        _FP16 *token_data = data + t * hidden_dim;
        std::cout << "[" << name << "] Token " << t << " First 3: [";
        for (size_t i = 0; i < std::min((unsigned int)3, hidden_dim); ++i)
          std::cout << (float)token_data[i] << " ";
        std::cout << "]" << std::endl;

        std::cout << "[" << name << "] Token " << t << " Last 3: [";
        if (hidden_dim >= 3) {
          for (size_t i = 0; i < 3; ++i)
            std::cout << (float)token_data[hidden_dim - 3 + i] << " ";
        }
        std::cout << "]" << std::endl;
      }
    }
#endif
  } else {
    // Fallback for weights or other shapes (flattened)
    if (tensor.getDataType() == ml::train::TensorDim::DataType::FP32) {
      float *data = tensor.getData<float>();
      std::cout << "[" << name << "] First 3: [";
      for (size_t i = 0; i < std::min((size_t)3, len); ++i)
        std::cout << data[i] << " ";
      std::cout << "]" << std::endl;
      std::cout << "[" << name << "] Last 3: [";
      if (len >= 3) {
        for (size_t i = 0; i < 3; ++i)
          std::cout << data[len - 3 + i] << " ";
      }
      std::cout << "]" << std::endl;
    }
#ifdef ENABLE_FP16
    else if (tensor.getDataType() == ml::train::TensorDim::DataType::FP16) {
      _FP16 *data = tensor.getData<_FP16>();
      std::cout << "[" << name << "] First 3: [";
      for (size_t i = 0; i < std::min((size_t)3, len); ++i)
        std::cout << (float)data[i] << " ";
      std::cout << "]" << std::endl;
      std::cout << "[" << name << "] Last 3: [";
      if (len >= 3) {
        for (size_t i = 0; i < 3; ++i)
          std::cout << (float)data[len - 3 + i] << " ";
      }
      std::cout << "]" << std::endl;
    }
#endif
  }
  std::cout << "--------------------" << std::endl;
}

#endif // __LLM_UTIL_HPP__

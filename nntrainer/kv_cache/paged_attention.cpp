// SPDX-License-Identifier: Apache-2.0
/** @file paged_attention.cpp
 * @brief Direct CPU attention over checked KV segments using online softmax.
 * @author Codex (OpenAI)
 */
#include <kv_cache/kv_cache_manager.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace nntrainer::kv {
namespace {
// Decode IEEE binary16 without requiring compiler or platform half support.
float load(const char *data, size_t index, Encoding encoding) {
  if (encoding == Encoding::Float32) {
    float value;
    std::memcpy(&value, data + index * sizeof(float), sizeof(float));
    return value;
  }
  uint16_t bits;
  std::memcpy(&bits, data + index * sizeof(bits), sizeof(bits));
  const unsigned int exponent = (bits >> 10) & 31;
  const unsigned int mantissa = bits & 1023;
  float value;
  if (!exponent)
    value = std::ldexp(static_cast<float>(mantissa), -24);
  else if (exponent != 31)
    value = std::ldexp(static_cast<float>(1024 + mantissa),
                       static_cast<int>(exponent) - 25);
  else
    value = mantissa ? std::numeric_limits<float>::quiet_NaN()
                     : std::numeric_limits<float>::infinity();
  return bits & 0x8000 ? -value : value;
}
} // namespace

std::vector<float> KvCacheManager::attention(const StepLease &step,
                                             size_t layer,
                                             const std::vector<float> &queries,
                                             const AttentionOptions &options,
                                             size_t batch) {
  checkStep(step);
  const auto &s = layerSpec(layer);
  if (s.capacity != s.max_tokens)
    throw std::invalid_argument(
      "Segmented attention requires non-ring storage");
  if (!options.query_heads || options.query_heads % s.num_kv_heads ||
      !std::isfinite(options.scale) || options.scale < 0 ||
      !std::isfinite(options.logit_softcap) || options.logit_softcap < 0)
    throw std::invalid_argument("Invalid segmented attention options");
  const size_t width = checkedMultiply(options.query_heads, s.head_dim);
  const size_t elements = checkedMultiply(step.to() - step.from(), width);
  if (queries.size() != elements)
    throw std::invalid_argument("Attention query shape differs from KV step");
  for (float value : queries)
    if (!std::isfinite(value))
      throw std::invalid_argument("Attention query must be finite");
  const auto keys = readKey(step, layer, batch);
  const auto values = readValue(step, layer, batch);
  std::vector<float> output(elements);
  std::vector<double> accumulator(s.head_dim);
  const double scale =
    options.scale ? options.scale : 1.0 / std::sqrt(s.head_dim);
  const size_t group = options.query_heads / s.num_kv_heads;
  const size_t kv_width = s.num_kv_heads * s.head_dim;
  for (size_t q = 0; q < step.to() - step.from(); ++q) {
    const size_t end = options.causal ? step.from() + q + 1 : step.to();
    const size_t start =
      options.window && end > options.window ? end - options.window : 0;
    for (size_t h = 0; h < options.query_heads; ++h) {
      std::fill(accumulator.begin(), accumulator.end(), 0.0);
      double maximum = -std::numeric_limits<double>::infinity();
      double denominator = 0;
      size_t token = 0;
      for (size_t p = 0; p < keys.size() && token < end; ++p) {
        const auto *k = static_cast<const char *>(keys[p].data());
        const auto *v = static_cast<const char *>(values[p].data());
        const size_t rows = keys[p].sizeBytes() / s.rowBytes();
        for (size_t row = 0; row < rows && token < end; ++row, ++token) {
          if (token < start)
            continue;
          const size_t offset = row * kv_width + (h / group) * s.head_dim;
          double score = 0;
          for (size_t d = 0; d < s.head_dim; ++d) {
            const float key = load(k, offset + d, s.encoding);
            if (!std::isfinite(key))
              throw std::invalid_argument("Attention key must be finite");
            score +=
              static_cast<double>(queries[q * width + h * s.head_dim + d]) *
              key;
          }
          score *= scale;
          if (options.logit_softcap)
            score =
              options.logit_softcap * std::tanh(score / options.logit_softcap);
          if (!std::isfinite(score))
            throw std::invalid_argument(
              "Attention key produces a nonfinite score");
          const double next_max = std::max(maximum, score);
          const double old_weight = std::exp(maximum - next_max);
          const double weight = std::exp(score - next_max);
          denominator = denominator * old_weight + weight;
          for (size_t d = 0; d < s.head_dim; ++d) {
            const float value = load(v, offset + d, s.encoding);
            if (!std::isfinite(value))
              throw std::invalid_argument("Attention value must be finite");
            accumulator[d] = accumulator[d] * old_weight + weight * value;
          }
          maximum = next_max;
        }
      }
      for (size_t d = 0; d < s.head_dim; ++d)
        output[q * width + h * s.head_dim + d] =
          static_cast<float>(accumulator[d] / denominator);
    }
  }
  return output;
}
} // namespace nntrainer::kv

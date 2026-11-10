// SPDX-License-Identifier: Apache-2.0
/** @file turboquant.h
 * @brief Calibration-free Hadamard/Lloyd-Max keys and uniform four-bit values.
 * @author NNTrainer contributors
 * @bug Experimental; model quality must be evaluated for each checkpoint.
 */
#ifndef NNTRAINER_KV_TURBOQUANT_H
#define NNTRAINER_KV_TURBOQUANT_H

#include <array>
#include <cstddef>
#include <cstdint>

namespace nntrainer::kv {

/** Algorithm follows vLLM's turboquant_4bit_nc preset. This is a portable
 * implementation, not its GPU kernels or binary format. Metadata uses FP32
 * to avoid FP16 norm/scale overflow. RoPE precedes encoding. */
class TurboQuant4BitNC {
public:
  explicit TurboQuant4BitNC(size_t head_dim);
  size_t headBytes() const { return dim_ / 2 + 8; }
  size_t headDim() const { return dim_; }
  /** Orthonormal Sylvester transform; self-inverse. */
  void rotate(float *row) const;
  void encodeKey(const float *row, uint8_t *packed) const;
  void encodeValue(const float *row, uint8_t *packed) const;
  /** Dequantize keys in the rotated basis with norm correction. */
  void decodeKeyRotated(const uint8_t *packed, float *row) const;
  void decodeValue(const uint8_t *packed, float *row) const;
  const std::array<float, 16> &centroids() const { return centroids_; }

private:
  size_t dim_;
  std::array<float, 16> centroids_;
  std::array<float, 15> boundaries_;
};
} // namespace nntrainer::kv
#endif

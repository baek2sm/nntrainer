// SPDX-License-Identifier: Apache-2.0
/** @file turboquant.cpp
 * @brief Portable implementation of the turboquant_4bit_nc algorithm.
 * @author NNTrainer contributors
 * @bug Experimental; no checkpoint quality guarantee.
 */
#include <kv_cache/turboquant.h>
#include <kv_cache/turboquant_codec.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace nntrainer::kv {
namespace {
// App builds may enable -ffast-math. Inspect IEEE bits so validation is not
// removed by the compiler's assumption that floating inputs are finite.
bool finite(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits & 0x7f800000u) != 0x7f800000u;
}
void storeMeta(uint8_t *p, float a, float b) {
  std::memcpy(p, &a, sizeof(a));
  std::memcpy(p + sizeof(a), &b, sizeof(b));
}
float loadMeta(const uint8_t *p) {
  float value;
  std::memcpy(&value, p, sizeof(value));
  return value;
}
unsigned nibble(const uint8_t *p, size_t i) {
  return (p[i / 2] >> ((i % 2) * 4)) & 15u;
}
void finiteRow(const float *row, size_t dim) {
  if (!row)
    throw std::invalid_argument("Null TurboQuant row");
  for (size_t i = 0; i < dim; ++i)
    if (!finite(row[i]))
      throw std::invalid_argument("TurboQuant requires finite input");
}
} // namespace

TurboQuant4BitNC::TurboQuant4BitNC(size_t head_dim) : dim_(head_dim) {
  if (dim_ < 64 || dim_ > 1024 || (dim_ & (dim_ - 1)))
    throw std::invalid_argument(
      "TurboQuant head_dim must be a power of two in [64, 1024]");
  using Tables = std::pair<std::array<float, 16>, std::array<float, 15>>;
  static std::mutex mutex;
  static std::map<size_t, Tables> tables;
  std::lock_guard<std::mutex> lock(mutex);
  const auto cached = tables.find(dim_);
  if (cached != tables.end()) {
    centroids_ = cached->second.first;
    boundaries_ = cached->second.second;
    return;
  }
  // Lloyd-Max for N(0, 1/D). Use the upstream preset's initialization,
  // integration range, 200 trapezoid intervals and 200 refinement steps.
  const double sigma = 1.0 / std::sqrt(static_cast<double>(dim_));
  std::array<double, 16> levels;
  for (size_t i = 0; i < levels.size(); ++i)
    levels[i] = sigma * (-3.5 + 7.0 * (i + 0.5) / 16.0);
  for (unsigned iteration = 0; iteration < 200; ++iteration) {
    auto next = levels;
    double delta = 0;
    for (size_t i = 0; i < levels.size(); ++i) {
      const double lo = i ? (levels[i - 1] + levels[i]) * 0.5 : -10.5 * sigma;
      const double hi =
        i < 15 ? (levels[i] + levels[i + 1]) * 0.5 : 10.5 * sigma;
      double mass = 0, moment = 0;
      for (unsigned j = 0; j <= 200; ++j) {
        const double x = lo + (hi - lo) * j / 200.0;
        const double weight = (j == 0 || j == 200) ? 0.5 : 1.0;
        const double p = weight * std::exp(-0.5 * x * x * dim_);
        mass += p;
        moment += x * p;
      }
      next[i] = moment / mass;
      delta = std::max(delta, std::abs(next[i] - levels[i]));
    }
    if (delta < 1e-10)
      break;
    levels = next;
  }
  for (size_t i = 0; i < levels.size(); ++i)
    centroids_[i] = static_cast<float>(levels[i]);
  for (size_t i = 0; i < boundaries_.size(); ++i)
    boundaries_[i] = static_cast<float>((levels[i] + levels[i + 1]) * 0.5);
  tables.emplace(dim_, Tables{centroids_, boundaries_});
}

void TurboQuant4BitNC::rotate(float *row) const {
  for (size_t span = 1; span < dim_; span *= 2)
    for (size_t base = 0; base < dim_; base += 2 * span)
      for (size_t i = 0; i < span; ++i) {
        const float a = row[base + i], b = row[base + span + i];
        row[base + i] = a + b;
        row[base + span + i] = a - b;
      }
  const float scale = 1.0f / std::sqrt(static_cast<float>(dim_));
  for (size_t i = 0; i < dim_; ++i)
    row[i] *= scale;
}

void TurboQuant4BitNC::encodeKey(const float *row, uint8_t *packed) const {
  nntr_tq_codec codec;
  codec.dim = static_cast<uint32_t>(dim_);
  std::copy(centroids_.begin(), centroids_.end(), codec.centroids);
  std::copy(boundaries_.begin(), boundaries_.end(), codec.boundaries);
  thread_local std::vector<float> scratch;
  scratch.resize(dim_);
  if (nntr_tq_encode_key(&codec, row, packed, scratch.data()))
    throw std::invalid_argument(
      "TurboQuant requires a finite representable key");
}

void TurboQuant4BitNC::encodeValue(const float *row, uint8_t *packed) const {
  finiteRow(row, dim_);
  const auto range = std::minmax_element(row, row + dim_);
  const float lo = *range.first;
  const float scale =
    static_cast<float>((static_cast<double>(*range.second) - lo) / 15.0);
  if (!finite(scale))
    throw std::invalid_argument("TurboQuant value range exceeds FP32");
  std::memset(packed, 0, headBytes());
  for (size_t i = 0; i < dim_; ++i) {
    const double value =
      scale > 0 ? (static_cast<double>(row[i]) - lo) / scale : 0;
    const unsigned index = static_cast<unsigned>(
      std::min(15.0, std::max(0.0, std::floor(value + 0.5))));
    packed[i / 2] |= static_cast<uint8_t>(index << ((i % 2) * 4));
  }
  storeMeta(packed + dim_ / 2, scale, lo);
}

void TurboQuant4BitNC::decodeKeyRotated(const uint8_t *packed,
                                        float *row) const {
  const float norm = loadMeta(packed + dim_ / 2);
  float scale = loadMeta(packed + dim_ / 2 + sizeof(float));
  if (!finite(norm) || norm < 0 || !finite(scale) || scale < 0)
    throw std::invalid_argument("Invalid TurboQuant key metadata");
  // Older pages left the second metadata slot zero. New pages keep the
  // norm-correction multiplier computed at append, eliminating a scan/sqrt.
  double squared = 0;
  for (size_t i = 0; i < dim_; ++i) {
    row[i] = centroids_[nibble(packed, i)];
    if (scale == 0 && norm > 0)
      squared += static_cast<double>(row[i]) * row[i];
  }
  if (scale == 0 && norm > 0)
    scale = norm / static_cast<float>(std::sqrt(squared));
  for (size_t i = 0; i < dim_; ++i)
    row[i] *= scale;
}

void TurboQuant4BitNC::decodeValue(const uint8_t *packed, float *row) const {
  const float scale = loadMeta(packed + dim_ / 2);
  const float lo = loadMeta(packed + dim_ / 2 + sizeof(float));
  if (!finite(scale) || scale < 0 || !finite(lo))
    throw std::invalid_argument("Invalid TurboQuant value metadata");
  for (size_t i = 0; i < dim_; ++i)
    row[i] =
      static_cast<float>(static_cast<double>(nibble(packed, i)) * scale + lo);
}
} // namespace nntrainer::kv

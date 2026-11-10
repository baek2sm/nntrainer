// SPDX-License-Identifier: Apache-2.0
/** @file sparse_prefix.h
 * @brief Ordered token deletion matching and position-only key correction.
 * @author Codex <noreply@openai.com>
 */
#ifndef NNTRAINER_SPARSE_PREFIX_H
#define NNTRAINER_SPARSE_PREFIX_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace nntrainer::kv {

/** Bit check remains valid in builds using -ffast-math. */
inline bool sparseFinite(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits & 0x7f800000u) != 0x7f800000u;
}

/** A contiguous destination run gathered from an immutable source template. */
struct SparsePrefixRun {
  size_t source = 0, destination = 0, count = 0;
};

/** Greedy ordered subsequence, stopping at the first unmatched request token.
 * A source token is used at most once. Earliest matches maximize the reusable
 * request prefix, including repeated IDs. No token from another template is
 * ever spliced into the same plan. Optional vocabulary must be sorted unique
 * IDs containing every source token; conservative extra IDs are allowed. It
 * only accelerates rejection and does not replace the ordered match. */
inline std::vector<SparsePrefixRun>
matchSparseTokens(const uint32_t *source, size_t source_count,
                  const uint32_t *request, size_t request_count,
                  const std::vector<uint32_t> *vocabulary = nullptr) {
  if ((source_count && !source) || (request_count && !request))
    throw std::invalid_argument("Sparse prefix requires token IDs");
  std::vector<SparsePrefixRun> runs;
  size_t src = 0;
  for (size_t dst = 0; dst < request_count; ++dst) {
    // A published source may cache sorted unique IDs. Test only at a mismatch,
    // keeping contiguous hits linear and rejecting absent IDs without scanning
    // the rest of a long source. The vocabulary may conservatively include
    // unavailable rows; it must never omit IDs in source[0:source_count].
    if (src < source_count && source[src] != request[dst] && vocabulary &&
        !std::binary_search(vocabulary->begin(), vocabulary->end(),
                            request[dst]))
      break;
    while (src < source_count && source[src] != request[dst])
      ++src;
    if (src == source_count)
      break;
    if (!runs.empty() && runs.back().source + runs.back().count == src)
      ++runs.back().count;
    else
      runs.push_back({src, dst, 1});
    ++src;
  }
  return runs;
}

inline size_t sparsePrefixLength(const std::vector<SparsePrefixRun> &runs) {
  return runs.empty() ? 0 : runs.back().destination + runs.back().count;
}

/** Frequencies MUST come from the producing model's actual RoPE parameters.
 * Zero frequencies preserve non-rotary pairs (partial RoPE). Empty means the
 * layer has no RoPE. No magnitude scaling is applied twice. */
struct SparseRope {
  std::vector<float> frequencies;
  bool interleaved = false;

  void validate(size_t head_dim) const {
    if (frequencies.size() > head_dim / 2)
      throw std::invalid_argument("Sparse RoPE exceeds head dimension");
    for (float frequency : frequencies)
      if (!sparseFinite(frequency))
        throw std::invalid_argument("Nonfinite sparse RoPE frequency");
  }
  /** One position delta is shared by every token/head of a gathered run. */
  std::vector<float> rotation(size_t source, size_t destination) const {
    const double delta =
      static_cast<double>(destination) - static_cast<double>(source);
    std::vector<float> coefficients(2 * frequencies.size());
    for (size_t i = 0; i < frequencies.size(); ++i) {
      const double angle = delta * frequencies[i];
      coefficients[2 * i] = static_cast<float>(std::cos(angle));
      coefficients[2 * i + 1] = static_cast<float>(std::sin(angle));
    }
    return coefficients;
  }
  void reposition(float *key, const std::vector<float> &coefficients) const {
    if (coefficients.size() != 2 * frequencies.size())
      throw std::invalid_argument("Sparse RoPE coefficient geometry mismatch");
    for (size_t i = 0; i < frequencies.size(); ++i) {
      const size_t a = interleaved ? 2 * i : i;
      const size_t b = interleaved ? 2 * i + 1 : i + frequencies.size();
      const float c = coefficients[2 * i], s = coefficients[2 * i + 1];
      const float x = key[a], y = key[b];
      key[a] = x * c - y * s;
      key[b] = y * c + x * s;
    }
  }
  void reposition(float *key, size_t source, size_t destination) const {
    const double delta =
      static_cast<double>(destination) - static_cast<double>(source);
    for (size_t i = 0; i < frequencies.size(); ++i) {
      const size_t a = interleaved ? 2 * i : i;
      const size_t b = interleaved ? 2 * i + 1 : i + frequencies.size();
      const double angle = delta * frequencies[i];
      const float c = static_cast<float>(std::cos(angle));
      const float s = static_cast<float>(std::sin(angle));
      const float x = key[a], y = key[b];
      key[a] = x * c - y * s;
      key[b] = y * c + x * s;
    }
  }
};

struct SparsePrefixResult {
  size_t tokens = 0;
  size_t exact_tokens = 0; /**< Leading rows before the first deletion. */
  bool approximate() const { return tokens > exact_tokens; }
};

} // namespace nntrainer::kv
#endif

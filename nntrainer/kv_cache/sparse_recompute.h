// SPDX-License-Identifier: Apache-2.0
/** @file sparse_recompute.h
 * @brief Contiguous block selection under a strict token recompute budget.
 * @author Codex <noreply@openai.com>
 */
#ifndef NNTRAINER_SPARSE_RECOMPUTE_H
#define NNTRAINER_SPARSE_RECOMPUTE_H

#include <kv_cache/sparse_prefix.h>
#include <limits>

namespace nntrainer::kv {

inline bool sparseRecomputeFinite(double value) {
  uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}

struct SparseRecomputeOptions {
  bool enabled = false;
  size_t block_tokens = 32;
  double ratio = 0.15;
  void validate() const {
    if (!block_tokens || !sparseRecomputeFinite(ratio) ||
        !(ratio >= 0 && ratio <= 1))
      throw std::invalid_argument("Invalid sparse recompute budget");
  }
};

struct SparseRecomputeBlock {
  size_t from = 0, count = 0;
  double error = 0;
};

struct SparseRecomputeStats {
  size_t prefix_tokens = 0, budget_tokens = 0, selected_tokens = 0;
  size_t full_first_layer_tokens = 0, probe_key_tokens = 0;
  size_t replay_calls = 0;
  double probe_ms = 0, replay_ms = 0;
  std::vector<double> block_errors;
  std::vector<SparseRecomputeBlock> blocks;
};

/** Scores are sums of squared K differences after position alignment. A
 * block is indivisible, including the short final block. Budget is based on
 * all retained tokens; leading exact rows are ineligible. Stable ties use
 * earlier positions. This is greedy block ranking, not a knapsack optimum.
 */
inline SparseRecomputeStats
selectSparseRecomputeBlocks(const std::vector<double> &errors,
                            const SparseRecomputeOptions &options,
                            size_t exact_tokens = 0) {
  options.validate();
  SparseRecomputeStats result;
  const size_t n = errors.size();
  if (exact_tokens > n)
    throw std::invalid_argument("Exact sparse prefix exceeds token count");
  result.prefix_tokens = n;
  result.budget_tokens = static_cast<size_t>(n * options.ratio);
  std::vector<SparseRecomputeBlock> ranked;
  for (size_t from = 0; from < n;) {
    const size_t count = std::min(options.block_tokens, n - from);
    double error = 0;
    for (size_t i = from; i < from + count; ++i) {
      if (!sparseRecomputeFinite(errors[i]) || !(errors[i] >= 0))
        throw std::invalid_argument("Invalid sparse K error");
      if (i >= exact_tokens)
        error += errors[i];
    }
    if (!sparseRecomputeFinite(error))
      throw std::invalid_argument("Sparse block error overflow");
    result.block_errors.push_back(error);
    if (from + count > exact_tokens)
      ranked.push_back({from, count, error});
    from += count;
  }
  if (!options.enabled)
    return result;
  std::stable_sort(
    ranked.begin(), ranked.end(),
    [](const auto &a, const auto &b) { return a.error > b.error; });
  for (const auto &block : ranked)
    if (block.count <= result.budget_tokens - result.selected_tokens) {
      result.blocks.push_back(block);
      result.selected_tokens += block.count;
    }
  std::sort(result.blocks.begin(), result.blocks.end(),
            [](const auto &a, const auto &b) { return a.from < b.from; });
  return result;
}

} // namespace nntrainer::kv
#endif

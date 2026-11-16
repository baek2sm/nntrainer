// SPDX-License-Identifier: Apache-2.0
/** @file unittest_causallm_sparse.cpp
 * @brief Real graph checks for immutable sparse gather and block replay.
 * @author Codex <noreply@openai.com>
 */
#include <causallm_test_utils.h>
#include <cmath>
#include <cstring>
#include <fp16.h>
#include <gtest/gtest.h>
#include <kv_cache/sparse_recompute.h>
#include <layer_node.h>
#include <qwen3_causallm.h>

namespace {
using namespace nntrainer::kv;
using Adapter = causallm_test::CausalLMTestAdapter<causallm::Qwen3CausalLM>;

class SparseModel final : public Adapter {
public:
  SparseModel(causallm::json &c, causallm::json &g, causallm::json &n) :
    causallm::Transformer(c, g, n, causallm::ModelType::CAUSALLM),
    Adapter(c, g, n) {}
  void prefill(const std::vector<uint32_t> &ids) {
    prefillLogitsFromIds(ids);
    setKVCachePosition(ids.size());
    global_token_len = ids.size();
    sparse_kv_approximate_ = false;
  }
  size_t restore(std::shared_ptr<const SparseKVPrefix> source,
                 const std::vector<uint32_t> &ids,
                 SparseRecomputeOptions options) {
    setSparseKVPrefix(std::move(source));
    setSparseKVRecompute(options);
    setKVCachePosition(0);
    global_token_len = 0;
    sparse_recompute_stats_ = {};
    auto request = ids;
    request.push_back(0); // production reserves this final row for logits
    return restoreArmedSparseKVPrefix(request);
  }
  std::vector<uint16_t> cache(unsigned layer, bool key, unsigned rows) {
    auto &t = key ? kv_cache.getKeyCache(layer) : kv_cache.getValueCache(layer);
    if (t.getDim().getDataTypeSize() != 2)
      throw std::runtime_error("Test expects half cache");
    return {t.getData<uint16_t>(), t.getData<uint16_t>() + rows * t.width()};
  }
  void seed() {
    unsigned serial = 1;
    forEachLayer(
      [&](ml::train::Layer &layer, nntrainer::RunLayerContext &c, void *) {
        for (unsigned w = 0; w < c.getNumWeights(); ++w) {
          auto &t = c.getWeight(w);
          if (t.getDataType() != ml::train::TensorDim::DataType::FP32)
            continue;
          for (size_t i = 0; i < t.size(); ++i)
            t.getData()[i] =
              causallm_test::isRmsNormLayerType(layer.getType())
                ? 1.0f
                : 0.06f * std::sin(static_cast<float>(++serial) * 0.173f);
        }
      });
  }
  void repairChunk(unsigned rows) { INIT_SEQ_LEN = rows; }
};

std::unique_ptr<SparseModel> makeModel(const char *name) {
  auto files =
    causallm_test::makeTinyCausalLMFiles("SparseKVReplay", name, "FP32");
  causallm::json c = {{"architectures", {"Qwen3ForCausalLM"}},
                      {"bos_token_id", 0},
                      {"eos_token_id", {31}},
                      {"head_dim", 8},
                      {"hidden_size", 64},
                      {"intermediate_size", 96},
                      {"is_causal", true},
                      {"max_position_embeddings", 128},
                      {"num_attention_heads", 8},
                      {"num_hidden_layers", 3},
                      {"num_key_value_heads", 4},
                      {"rms_norm_eps", 1e-5},
                      {"rope_theta", 10000},
                      {"tie_word_embeddings", true},
                      {"vocab_size", 32}};
  auto g = causallm_test::makeTinyGenerationConfig();
  auto n = causallm_test::makeTinyNntrainerConfig(
    files.tokenizer_path, causallm_test::makeTinyFp32DataType());
  n["init_seq_len"] = 64;
  n["max_seq_len"] = 128;
  auto model = std::make_unique<SparseModel>(c, g, n);
  model->initializeModel();
  model->seed();
  return model;
}

std::vector<uint32_t> sourceIds() {
  std::vector<uint32_t> ids;
  for (unsigned i = 0; i < 64; ++i)
    ids.push_back(1 + (i * 7) % 29);
  return ids;
}
std::vector<uint32_t> subsetIds() {
  auto s = sourceIds();
  s.erase(s.begin() + 8, s.begin() + 32);
  return s;
}

TEST(SparseKVBlocks, StrictBudgetAndShortTail) {
  SparseRecomputeOptions o{true, 32, 0.15};
  std::vector<double> e(1000, 0);
  e[160] = 10;
  e[800] = 9;
  e[500] = 8;
  auto r = selectSparseRecomputeBlocks(e, o);
  EXPECT_LE(r.selected_tokens, 150u);
  for (size_t expected : {160u, 480u, 800u})
    EXPECT_TRUE(std::any_of(r.blocks.begin(), r.blocks.end(),
                            [&](auto b) { return b.from == expected; }));
  for (const auto &b : r.blocks)
    EXPECT_EQ(b.from % 32, 0u);
  auto tiny = selectSparseRecomputeBlocks(std::vector<double>(10, 1), o);
  EXPECT_EQ(tiny.selected_tokens, 0u);
}

TEST(SparseKVBlocks, InvalidAndDisabledOptions) {
  EXPECT_THROW(selectSparseRecomputeBlocks({1}, {true, 0, .15}),
               std::invalid_argument);
  EXPECT_THROW(selectSparseRecomputeBlocks({1}, {true, 32, -1}),
               std::invalid_argument);
  EXPECT_THROW(selectSparseRecomputeBlocks({1}, {true, 32, NAN}),
               std::invalid_argument);
  EXPECT_THROW(selectSparseRecomputeBlocks({NAN}, {true, 32, .15}),
               std::invalid_argument);
  EXPECT_THROW(selectSparseRecomputeBlocks({INFINITY}, {true, 32, .15}),
               std::invalid_argument);
  EXPECT_EQ(selectSparseRecomputeBlocks({1, 2}, {false, 1, 1}).selected_tokens,
            0u);
  EXPECT_EQ(
    selectSparseRecomputeBlocks({1, 2}, {true, 1, 1}, 2).selected_tokens, 0u);
}

TEST(SparseKVReplay, RopeOnlyPreservesValuesAndSource) {
  auto m = makeModel("rope");
  const auto source = sourceIds(), subset = subsetIds();
  m->prefill(source);
  auto asset = m->captureSparseKVPrefix(source);
  const auto original = asset->kv->layers[0].pages[0]->key;
  auto v = m->cache(1, false, 64);
  ASSERT_EQ(m->restore(asset, subset, {false, 4, .15}), 40u);
  auto actual = m->cache(1, false, 40);
  const auto runs = matchSparseTokens(source.data(), source.size(),
                                      subset.data(), subset.size());
  for (const auto &run : runs)
    for (size_t t = 0; t < run.count; ++t)
      for (size_t d = 0; d < 32; ++d)
        EXPECT_EQ(actual[(run.destination + t) * 32 + d],
                  v[(run.source + t) * 32 + d]);
  EXPECT_EQ(m->getSparseKVRecomputeStats().full_first_layer_tokens, 0u);
  EXPECT_EQ(asset->kv->layers[0].pages[0]->key, original);
}

TEST(SparseKVReplay, OnlySelectedLaterRowsChange) {
  auto m = makeModel("partial");
  const auto source = sourceIds(), subset = subsetIds();
  m->prefill(source);
  auto asset = m->captureSparseKVPrefix(source);
  auto original = asset->kv->layers[2].pages[0]->key;
  m->restore(asset, subset, {false, 4, .15});
  auto basek = m->cache(2, true, 40), basev = m->cache(2, false, 40);
  ASSERT_EQ(m->restore(asset, subset, {true, 4, .15}), 40u);
  const auto &r = m->getSparseKVRecomputeStats();
  EXPECT_EQ(r.full_first_layer_tokens, 40u);
  EXPECT_EQ(r.probe_key_tokens, 40u);
  EXPECT_EQ(r.selected_tokens, 4u);
  EXPECT_EQ(r.budget_tokens, 6u);
  std::vector<bool> selected(40, false);
  for (auto b : r.blocks)
    for (size_t t = b.from; t < b.from + b.count; ++t)
      selected[t] = true;
  auto k = m->cache(2, true, 40), v = m->cache(2, false, 40);
  size_t changed = 0;
  for (size_t t = 0; t < 40; ++t)
    for (size_t d = 0; d < 32; ++d) {
      if (!selected[t]) {
        EXPECT_EQ(k[t * 32 + d], basek[t * 32 + d]);
        EXPECT_EQ(v[t * 32 + d], basev[t * 32 + d]);
      } else
        changed += k[t * 32 + d] != basek[t * 32 + d] ||
                   v[t * 32 + d] != basev[t * 32 + d];
    }
  EXPECT_GT(changed, 0u);
  EXPECT_EQ(asset->kv->layers[2].pages[0]->key, original);
  EXPECT_THROW(m->captureSparseKVPrefix(subset), std::invalid_argument);
}

TEST(SparseKVReplay, FullBudgetMatchesColdSubset) {
  auto m = makeModel("full");
  const auto source = sourceIds(), subset = subsetIds();
  m->prefill(source);
  auto asset = m->captureSparseKVPrefix(source);
  m->prefill(subset);
  std::vector<std::vector<uint16_t>> expected;
  for (unsigned l = 0; l < 3; ++l)
    for (bool k : {true, false})
      expected.push_back(m->cache(l, k, 40));
  m->restore(asset, subset, {true, 4, 1.0});
  EXPECT_EQ(m->getSparseKVRecomputeStats().replay_calls, 1u);
  size_t idx = 0;
  for (unsigned l = 0; l < 3; ++l)
    for (bool k : {true, false}) {
      auto actual = m->cache(l, k, 40);
      double error = 0, energy = 0;
      for (size_t i = 0; i < actual.size(); ++i) {
        double a = nntrainer::compute_fp16_to_fp32(actual[i]);
        double b = nntrainer::compute_fp16_to_fp32(expected[idx][i]);
        error += (a - b) * (a - b);
        energy += b * b;
      }
      EXPECT_LT(std::sqrt(error / std::max(energy, 1e-20)), 0.003)
        << "layer=" << l << " K=" << k;
      ++idx;
    }
}

TEST(SparseKVReplay, ChunkedProbeKeepsAbsolutePositions) {
  auto m = makeModel("chunks");
  const auto source = sourceIds(), subset = subsetIds();
  m->prefill(source);
  auto asset = m->captureSparseKVPrefix(source);
  m->prefill(subset);
  auto expected = m->cache(2, true, 40);
  // Activation capacity stays 64; this forces the executor to drive 7-row
  // calls.
  m->repairChunk(7);
  m->restore(asset, subset, {true, 4, 1.0});
  EXPECT_EQ(m->getSparseKVRecomputeStats().replay_calls, 5u);
  auto actual = m->cache(2, true, 40);
  double error = 0, energy = 0;
  for (size_t i = 0; i < actual.size(); ++i) {
    double a = nntrainer::compute_fp16_to_fp32(actual[i]);
    double b = nntrainer::compute_fp16_to_fp32(expected[i]);
    error += (a - b) * (a - b);
    energy += b * b;
  }
  EXPECT_LT(std::sqrt(error / std::max(energy, 1e-20)), 0.003);
}

TEST(SparseKVReplay, ZeroBudgetProbesWithoutUpdatingLaterKV) {
  auto m = makeModel("zero");
  const auto source = sourceIds(), subset = subsetIds();
  m->prefill(source);
  auto asset = m->captureSparseKVPrefix(source);
  m->restore(asset, subset, {false, 4, 0});
  auto before = m->cache(1, true, 40);
  m->restore(asset, subset, {true, 4, 0});
  EXPECT_EQ(m->getSparseKVRecomputeStats().selected_tokens, 0u);
  EXPECT_EQ(m->getSparseKVRecomputeStats().probe_key_tokens, 40u);
  EXPECT_EQ(m->cache(1, true, 40), before);
}
TEST(SparseKVReplay, RejectsSourceFromDifferentModel) {
  auto a = makeModel("owner_a");
  const auto ids = sourceIds();
  a->prefill(ids);
  auto source = a->captureSparseKVPrefix(ids);
  auto b = makeModel("owner_b");
  EXPECT_THROW(b->setSparseKVPrefix(source), std::invalid_argument);
}

} // namespace

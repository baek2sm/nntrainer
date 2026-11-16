// SPDX-License-Identifier: Apache-2.0
/** @file causal_lm_sparse.cpp
 * @brief Superset gather, position alignment and contiguous block repair.
 * @author Codex <noreply@openai.com>
 */
#include <causal_lm.h>
#include <fp16.h>
#include <layer_context.h>
#include <layer_node.h>
#include <llm_util.hpp>
#include <mha_core.h>
#include <neuralnet.h>

#include <chrono>
#include <cstring>
#include <limits>
#include <map>
#include <set>

namespace causallm {
namespace {
using Clock = std::chrono::steady_clock;
using Node = nntrainer::LayerNode;

int decoderIndex(const std::string &name) {
  if (name.compare(0, 5, "layer") != 0)
    return -1;
  size_t end = 5;
  while (end < name.size() && name[end] >= '0' && name[end] <= '9')
    ++end;
  if (end == 5 || end == name.size() || (name[end] != '_' && name[end] != ':'))
    return -1;
  return std::stoi(name.substr(5, end - 5));
}

bool cacheInput(const std::string &name) {
  return name.compare(0, 9, "cache_k_l") == 0 ||
         name.compare(0, 9, "cache_v_l") == 0;
}

size_t rowBytes(const nntrainer::Tensor &tensor) {
  if (tensor.batch() != 1 || tensor.channel() != 1 || tensor.isClMem())
    throw std::invalid_argument(
      "Sparse block repair requires batch-one host-addressable row tensors");
  return static_cast<size_t>(tensor.width()) *
         tensor.getDim().getDataTypeSize();
}

double elapsed(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
    .count();
}
} // namespace

void CausalLM::setSparseKVRecompute(
  const nntrainer::kv::SparseRecomputeOptions &options) {
  options.validate();
  sparse_recompute_options_ = options;
}

std::shared_ptr<const CausalLM::SparseKVPrefix>
CausalLM::captureSparseKVPrefix(const std::vector<uint32_t> &tokens) {
  if (tokens.empty() || !kv_cache.isAllocated() ||
      tokens.size() > getSessionTokenPosition() || sparse_kv_approximate_ ||
      (device_kv_manager_ && device_kv_manager_->approximate()))
    throw std::invalid_argument(
      "Capture requires a freshly computed KV prefix");
  auto result = std::make_shared<SparseKVPrefix>();
  result->tokens = tokens;
  result->identity_ = sparse_kv_identity_;
  result->kv =
    kv_cache.capturePrefix(static_cast<unsigned>(tokens.size()), nullptr, true);
  return result;
}

void CausalLM::setSparseKVPrefix(std::shared_ptr<const SparseKVPrefix> source) {
  if (source && source->identity_ != sparse_kv_identity_)
    throw std::invalid_argument(
      "Sparse source belongs to a different model instance");
  if (source && (!source->kv || source->tokens.empty() ||
                 source->tokens.size() != source->kv->position))
    throw std::invalid_argument("Sparse source tokens and KV rows disagree");
  if (source && device_kv_manager_)
    throw std::invalid_argument(
      "Use the device manager's existing source lookup on device sessions");
  sparse_kv_source_ = std::move(source);
}

void CausalLM::sparseKVGeometry(std::vector<unsigned> &dims,
                                std::vector<nntrainer::kv::SparseRope> &rope) {
  model->forEachLayer([&](ml::train::Layer &layer,
                          nntrainer::RunLayerContext &context, void *) {
    auto *node = dynamic_cast<Node *>(&layer);
    auto *mha = node ? dynamic_cast<MHACoreLayer *>(node->getLayer()) : nullptr;
    if (mha) {
      if (context.getNumInputs() < 5)
        throw std::invalid_argument("Sparse reuse requires external KV inputs");
      dims.push_back(mha->sparsePrefixHeadDim());
      rope.push_back(mha->sparsePrefixRope(true));
    }
  });
  if (dims.size() != kv_cache.getNumLayers())
    throw std::invalid_argument("Sparse KV geometry requires MHA cache layers");
}

size_t
CausalLM::restoreArmedSparseKVPrefix(const std::vector<uint32_t> &tokens) {
  if (tokens.size() < 2)
    return 0;
  if (causallm_engine() != "cpu" || REPETITION_PENALTY != 1.0f)
    throw std::invalid_argument(
      "Armed sparse prefix requires CPU and repetition_penalty=1");
  const auto &source = *sparse_kv_source_;
  auto runs =
    nntrainer::kv::matchSparseTokens(source.tokens.data(), source.tokens.size(),
                                     tokens.data(), tokens.size() - 1);
  const size_t count = nntrainer::kv::sparsePrefixLength(runs);
  if (!count)
    return 0;
  std::vector<unsigned> dims;
  std::vector<nntrainer::kv::SparseRope> rope;
  sparseKVGeometry(dims, rope);
  {
    const auto gathered =
      KVCacheManager::gatherSparsePrefix({source.kv}, runs, dims, rope);
    kv_cache.restorePrefix(*gathered);
  }
  setKVCachePosition(static_cast<unsigned>(count));
  global_token_len = static_cast<unsigned>(count);
  const size_t exact = runs.front().source == 0 ? runs.front().count : 0;
  sparse_kv_approximate_ = count > exact;
  sparse_recompute_stats_.prefix_tokens = count;
  if (sparse_kv_approximate_ && sparse_recompute_options_.enabled)
    repairSparseKVPrefix(
      std::vector<uint32_t>(tokens.begin(), tokens.begin() + count), exact);
  return count;
}

void CausalLM::repairSparseKVPrefix(const std::vector<uint32_t> &tokens,
                                    size_t exact_tokens) {
  using namespace nntrainer::kv;
  const auto &options = sparse_recompute_options_;
  options.validate();
  if (!options.enabled || tokens.empty())
    return;
  // A compacted sliding snapshot cannot supply arbitrary early repair queries.
  // Reject before executing any graph node instead of reading missing history.
  for (unsigned l = 0; l < kv_cache.getNumLayers(); ++l)
    if (kv_cache.getBase(l) || kv_cache.getRows(l) < tokens.size())
      throw std::invalid_argument(
        "Sparse block repair requires complete retained KV history per layer");
  // Accelerator graph replay and host staging require different ownership
  // contracts. The portable executor below is synchronous; never enter a
  // resident DSP/GPU graph using partially materialized host activations.
  if (htp_execution_profile_ || causallm_engine() != "cpu")
    throw std::invalid_argument(
      "Sparse block repair currently requires the synchronous CPU engine");

  auto *network = dynamic_cast<nntrainer::NeuralNetwork *>(model.get());
  if (!network || BATCH_SIZE != 1 || tokens.size() > MAX_SEQ_LEN)
    throw std::invalid_argument("Unsupported sparse repair graph");

  std::map<std::string, Node *> nodes;
  Node *probe = nullptr;
  model->forEachLayer(
    [&](ml::train::Layer &layer, nntrainer::RunLayerContext &, void *) {
      auto *node = dynamic_cast<Node *>(&layer);
      if (!node)
        throw std::invalid_argument("Sparse replay requires graph nodes");
      nodes.emplace(node->getName(), node);
      if (decoderIndex(node->getName()) == 1 &&
          dynamic_cast<MHACoreLayer *>(node->getLayer()))
        probe = node;
    });
  if (!probe)
    throw std::invalid_argument("Sparse repair requires a second MHA layer");

  struct Boundary {
    Node *producer;
    unsigned output;
    size_t bytes = 0;
    std::vector<unsigned char> rows;
  };
  using Edge = std::pair<std::string, unsigned>;
  std::map<Edge, Boundary> boundaries;
  for (const auto &[name, node] : nodes) {
    if (decoderIndex(name) < 1)
      continue;
    for (unsigned i = 0; i < node->getInputConnections().size(); ++i) {
      const auto &parent = node->getInputConnectionName(i);
      if (cacheInput(parent) || decoderIndex(parent) >= 1)
        continue;
      auto found = nodes.find(parent);
      if (found == nodes.end())
        throw std::invalid_argument("Sparse replay boundary producer missing");
      Edge edge{parent, node->getInputConnectionIndex(i)};
      boundaries.emplace(edge, Boundary{found->second, edge.second, 0, {}});
    }
  }
  if (boundaries.empty())
    throw std::invalid_argument("Sparse replay has no first-layer boundary");

  std::set<std::string> probe_required;
  std::function<void(const std::string &)> require_node =
    [&](const auto &name) {
      if (!probe_required.insert(name).second)
        return;
      auto found = nodes.find(name);
      if (found == nodes.end())
        throw std::invalid_argument("Sparse probe dependency missing");
      for (unsigned i = 0; i < found->second->getInputConnections().size(); ++i)
        require_node(found->second->getInputConnectionName(i));
    };
  require_node(probe->getInputConnectionName(1));
  for (const auto &[edge, boundary] : boundaries)
    require_node(edge.first);

  const size_t width = kv_cache.getKeyCache(1).width();
  auto &old_key = kv_cache.getKeyCache(1);
  const size_t key_bytes = old_key.getDim().getDataTypeSize();
  if (key_bytes != 2 && key_bytes != 4)
    throw std::invalid_argument("Sparse K probe requires FP16 or FP32 cache");
  const auto *cached = old_key.getData<unsigned char>();
  std::vector<double> errors(tokens.size());
  std::vector<float> prompt(MAX_SEQ_LEN, 0);

  auto invoke = [&](unsigned from, unsigned rows,
                    nntrainer::NeuralNetwork::IncrementalNodeHook hook) {
    for (unsigned i = 0; i < rows; ++i)
      prompt[i] = static_cast<float>(tokens[from + i]);
    const auto in_dims = network->getInputDimension();
    const auto &names = network->getInputNames();
    nntrainer::sharedConstTensors inputs;
    for (size_t i = 0; i < names.size(); ++i) {
      if (cacheInput(names[i])) {
        size_t end = 9;
        while (end < names[i].size() && names[i][end] >= '0' &&
               names[i][end] <= '9')
          ++end;
        const unsigned layer = std::stoul(names[i].substr(9, end - 9));
        auto &cache = names[i][6] == 'k' ? kv_cache.getKeyCache(layer)
                                         : kv_cache.getValueCache(layer);
        inputs.emplace_back(MAKE_SHARED_TENSOR(cache));
      } else {
        if (in_dims[i].getDataLen() > prompt.size())
          throw std::invalid_argument(
            "Sparse replay input exceeds prompt buffer");
        inputs.emplace_back(MAKE_SHARED_TENSOR(nntrainer::Tensor::Map(
          prompt.data(), prompt.size() * sizeof(float), in_dims[i], 0)));
      }
    }
    setKVCachePosition(from);
    network->setIncrementalNodeHook(std::move(hook));
    struct ClearHook {
      nntrainer::NeuralNetwork *network;
      ~ClearHook() { network->setIncrementalNodeHook({}); }
    } clear{network};
    // Results are deliberately ignored: prefix repair never evaluates lm_head.
    network->incremental_inference(inputs, rows, from, from + rows);
  };

  bool stopped = false;
  auto probe_hook = [&](Node &node, unsigned from, unsigned to, bool after) {
    const size_t rows = to - from;
    if (!after) {
      if (cacheInput(node.getName()))
        return true;
      if (stopped)
        return false;
      if (&node != probe)
        return probe_required.count(node.getName()) != 0;
      auto *mha = dynamic_cast<MHACoreLayer *>(node.getLayer());
      auto fresh = mha->sparsePrefixProbeKey(node.getRunContext(), from, rows);
      if (fresh.size() != rows * width)
        throw std::invalid_argument("Sparse K probe geometry mismatch");
      for (size_t t = 0; t < rows; ++t) {
        double error = 0;
        for (size_t d = 0; d < width; ++d) {
          float old;
          const auto *p = cached + ((from + t) * width + d) * key_bytes;
          if (key_bytes == 4)
            std::memcpy(&old, p, 4);
          else {
            uint16_t bits;
            std::memcpy(&bits, p, 2);
            old = nntrainer::compute_fp16_to_fp32(bits);
          }
          float updated = fresh[t * width + d];
          if (key_bytes == 2)
            updated = nntrainer::compute_fp16_to_fp32(
              nntrainer::compute_fp32_to_fp16(updated));
          const double diff = static_cast<double>(updated) - old;
          error += diff * diff;
        }
        errors[from + t] = error;
      }
      stopped = true;
      return false; // No second-layer attention/FFN on the full prompt.
    }
    for (auto &[edge, boundary] : boundaries)
      if (boundary.producer == &node) {
        auto &output = node.getOutput(boundary.output);
        const size_t bytes = rowBytes(output);
        if (!boundary.bytes) {
          boundary.bytes = bytes;
          boundary.rows.resize(checkedMultiply(bytes, tokens.size()));
        }
        if (bytes != boundary.bytes || rows > output.height())
          throw std::invalid_argument("Sparse replay boundary shape changed");
        std::memcpy(boundary.rows.data() + from * bytes,
                    output.getData<unsigned char>(), rows * bytes);
      }
    return true;
  };

  const auto probe_start = Clock::now();
  const unsigned chunk = INIT_SEQ_LEN;
  try {
    for (unsigned from = 0; from < tokens.size();) {
      const unsigned rows = std::min<size_t>(chunk, tokens.size() - from);
      stopped = false;
      invoke(from, rows, probe_hook);
      if (!stopped)
        throw std::logic_error("Sparse probe layer was not reached");
      from += rows;
    }
    auto result = selectSparseRecomputeBlocks(errors, options, exact_tokens);
    result.probe_ms = elapsed(probe_start);
    result.full_first_layer_tokens = tokens.size();
    result.probe_key_tokens = tokens.size();

    auto replay_hook = [&](Node &node, unsigned from, unsigned to, bool after) {
      if (after)
        return true;
      if (cacheInput(node.getName()))
        return true;
      if (decoderIndex(node.getName()) < 1)
        return false;
      for (unsigned i = 0; i < node.getInputConnections().size(); ++i) {
        Edge edge{node.getInputConnectionName(i),
                  node.getInputConnectionIndex(i)};
        auto found = boundaries.find(edge);
        if (found == boundaries.end())
          continue;
        auto &boundary = found->second;
        auto &input = node.getRunContext().getInput(i);
        if (!boundary.bytes || boundary.bytes != rowBytes(input) ||
            to - from > input.height())
          throw std::invalid_argument("Sparse replay boundary unavailable");
        std::memcpy(input.getData<unsigned char>(),
                    boundary.rows.data() + from * boundary.bytes,
                    (to - from) * boundary.bytes);
      }
      return true;
    };
    const auto replay_start = Clock::now();
    // Adjacent selected blocks share one call (subject to the activation
    // plane). Disjoint blocks retain absolute RoPE positions and causal masks.
    for (size_t i = 0; i < result.blocks.size();) {
      size_t from = result.blocks[i].from, count = result.blocks[i++].count;
      while (i < result.blocks.size() &&
             result.blocks[i].from == from + count) {
        count += result.blocks[i++].count;
      }
      while (count) {
        const unsigned rows = std::min<size_t>(chunk, count);
        invoke(static_cast<unsigned>(from), rows, replay_hook);
        ++result.replay_calls;
        from += rows;
        count -= rows;
      }
    }
    result.replay_ms = elapsed(replay_start);
    sparse_recompute_stats_ = std::move(result);
    setKVCachePosition(static_cast<unsigned>(tokens.size()));
  } catch (...) {
    setKVCachePosition(0);
    global_token_len = 0;
    throw;
  }
}

} // namespace causallm

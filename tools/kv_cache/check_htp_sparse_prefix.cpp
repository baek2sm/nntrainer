// SPDX-License-Identifier: Apache-2.0
/** @file check_htp_sparse_prefix.cpp
 * @brief Real-model deletion reuse, exact source preservation and timings.
 * Usage: htp_sparse_prefix_check MODEL_DIR OUTPUT_JSON
 */
#include <causal_lm.h>
#include <filesystem>
#include <fstream>
#include <gemma4_htp_moe_causallm.h>
#include <htp_mha_core.h>
#include <htp_model_config.h>
#include <htp_runtime_config.h>
#include <iostream>
#include <layer_node.h>
#include <model_registry.h>

namespace causallm {
class SparsePrefixCheck final : public Gemma4HtpMoECausalLM {
public:
  SparsePrefixCheck(json &c, json &g, json &n) :
    Transformer(sanitizeConfig(c), sanitizeGenerationConfig(g, c), n,
                ModelType::CAUSALLM),
    Gemma4HtpMoECausalLM(c, g, n) {}
  json checkProducerRope() {
    json layers = json::array();
    model->forEachLayer(
      [&](ml::train::Layer &layer, nntrainer::RunLayerContext &, void *) {
        auto *node = dynamic_cast<nntrainer::LayerNode *>(&layer);
        auto *mha =
          node ? dynamic_cast<HtpMHACoreLayer *>(node->getLayer()) : nullptr;
        if (!mha)
          return;
        const auto local = mha->sparsePrefixRope();
        const auto upstream = mha->sparsePrefixRope(true);
        const size_t dim = mha->sparsePrefixHeadDim();
        if (upstream.frequencies.size() != dim / 2)
          throw std::runtime_error("Fused QKV sparse RoPE is missing");
        size_t nonzero = 0;
        for (float frequency : upstream.frequencies)
          nonzero += frequency != 0;
        if (!nonzero)
          throw std::runtime_error("Fused QKV sparse RoPE has no rotation");
        layers.push_back({{"name", layer.getName()},
                          {"head_dim", dim},
                          {"local_rope", !local.frequencies.empty()},
                          {"rotary_pairs", nonzero}});
      },
      nullptr);
    if (layers.empty())
      throw std::runtime_error("No attention producers checked");
    return layers;
  }
  std::unique_ptr<DeviceKVManager> suspendLookup() {
    device_prefill_capture_ = false;
    return std::move(device_kv_manager_);
  }
  void resumeLookup(std::unique_ptr<DeviceKVManager> manager) {
    device_kv_manager_ = std::move(manager);
    device_prefill_capture_ = true;
  }
};
} // namespace causallm

using json = causallm::json;
static json read(const std::filesystem::path &p) {
  std::ifstream file(p);
  json value;
  file >> value;
  return value;
}
static void require(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}
int main(int argc, char **argv) {
  if (argc != 3)
    return 2;
  json report{{"passed", false}};
  auto write = [&] {
    std::ofstream out(argv[2]);
    out << report.dump(2) << '\n';
  };
  try {
    const std::filesystem::path dir = argv[1];
    auto cfg = read(dir / "config.json"),
         gen = read(dir / "generation_config.json"),
         nc = read(dir / "nntr_config.json");
    nc["htp"]["kv_turboquant"] = false;
    nc["htp"]["kv_session"] = true;
    nc["htp"]["kv_paged"] = true;
    nc["htp"]["kv_prefix_cache"] = true;
    nc["htp"]["kv_sparse_prefix_cache"] = true;
    nc["htp"]["kv_seed_q"] = false;
    nc["num_to_generate"] = 24;
    nc["skip_prefill"] = true;
    // Paths resolve relative to the supplied model, independent of run cwd.
    for (const char *key : {"model_file_name", "tokenizer_file",
                            "moe_hadamard_signs", "act_scale_json"})
      if (nc.contains(key)) {
        const std::filesystem::path p = nc.at(key).get<std::string>();
        if (p.is_relative())
          nc[key] = (std::filesystem::absolute(dir) / p).string();
      }
    nntrainer::configureHtpRuntime(causallm::normalizeHtpRuntimeConfig(nc));
    causallm::registerAllModels();
    auto model = std::make_unique<causallm::SparsePrefixCheck>(cfg, gen, nc);
    model->initialize();
    model->load_weight(nc.at("model_file_name"));
    model->repack_weight();
    auto *c = dynamic_cast<causallm::CausalLM *>(model.get());
    require(c, "CausalLM unavailable");
    report["producer_rope"] = model->checkProducerRope();
    auto *tokenizer = model->getTokenizer();
    std::string source = "<bos><|turn>user\nThe secret code is ORCHID-842.\n";
    for (unsigned i = 0; i < 48; ++i)
      source += "Note " + std::to_string(i) +
                ": The village has an orchard, a library and a stone bridge.\n";
    source += "\nWhat is the secret code? Reply only with the code."
              "<turn|>\n<|turn>model\n";
    const auto source_ids = tokenizer->Encode(source, true);
    require(source_ids.size() > 700 && source_ids.size() < 1800,
            "Unexpected source length");
    auto subset_ids = source_ids;
    // Two non-page-aligned holes, defined on token IDs to prevent matching
    // different text-token boundaries. A novel suffix is recomputed normally.
    subset_ids.erase(subset_ids.begin() + 403, subset_ids.begin() + 450);
    subset_ids.erase(subset_ids.begin() + 131, subset_ids.begin() + 258);
    const auto subset = tokenizer->Decode(subset_ids);
    require(tokenizer->Encode(subset, true) == subset_ids,
            "Deleted-token prompt does not roundtrip");
    auto run = [&](const std::string &text) {
      c->setSessionTokenPosition(0);
      const auto before = c->deviceKVManagerStats();
      model->run(text, false, "", "", false);
      const auto after = c->deviceKVManagerStats();
      const auto &metrics = c->getPerformanceMetrics();
      require(!c->getLastGeneratedTokenIds().empty() &&
                !metrics.prompt_truncated_tokens,
              "Generation is empty or prompt truncated");
      return json{{"output", c->getOutput()},
                  {"ids", c->getLastGeneratedTokenIds()},
                  {"prefill_tokens", metrics.prefill_tokens},
                  {"prefill_ms", metrics.prefill_duration_ms},
                  {"ttft_ms", metrics.ttft_ms},
                  {"decode_ms_per_token", metrics.decode_steady_ms},
                  {"peak_memory_kb", metrics.peak_memory_kb},
                  {"reused_tokens", after.reused_tokens - before.reused_tokens},
                  {"sparse_hits", after.sparse_hits - before.sparse_hits},
                  {"prefixes", after.prefixes}};
    };
    report["source_tokens"] = source_ids.size();
    report["subset_tokens"] = subset_ids.size();
    report["deleted_tokens"] = source_ids.size() - subset_ids.size();
    report["cold"] = run(subset);
    write();
    c->setSessionTokenPosition(0);
    c->enableDeviceKVManager(4, 4);
    report["source"] = run(source);
    write();
    auto held = model->suspendLookup();
    report["cold_warm"] = run(subset);
    model->resumeLookup(std::move(held));
    require(report["cold_warm"]["ids"] == report["cold"]["ids"],
            "Cold recomputation is not stable");
    report["sparse"] = run(subset);
    require(report["sparse"]["sparse_hits"] == 1 &&
              report["sparse"]["reused_tokens"].get<size_t>() > 131,
            "Sparse deletion reuse did not occur");
    require(report["sparse"]["prefixes"] == report["source"]["prefixes"],
            "Approximate result contaminated exact prefix cache");
    write();
    auto suffix_ids = subset_ids;
    suffix_ids.resize(suffix_ids.size() - 24);
    const auto extra = tokenizer->Encode(
      "\nAdditional task: check the first sentence. What is the secret code? "
      "Reply only with the code.<turn|>\n<|turn>model\n",
      false);
    suffix_ids.insert(suffix_ids.end(), extra.begin(), extra.end());
    const auto suffix_text = tokenizer->Decode(suffix_ids);
    require(tokenizer->Encode(suffix_text, true) == suffix_ids,
            "Suffix prompt does not roundtrip");
    held = model->suspendLookup();
    report["suffix_cold"] = run(suffix_text);
    model->resumeLookup(std::move(held));
    report["suffix_sparse"] = run(suffix_text);
    require(report["suffix_sparse"]["sparse_hits"] == 1 &&
              report["suffix_sparse"]["prefill_tokens"].get<unsigned>() > 1 &&
              report["suffix_sparse"]["prefixes"] ==
                report["source"]["prefixes"],
            "Sparse prefix plus fresh prefill suffix failed");
    write();
    report["exact_source_repeat"] = run(source);
    require(report["exact_source_repeat"]["sparse_hits"] == 0 &&
              report["exact_source_repeat"]["ids"] == report["source"]["ids"],
            "Sparse reuse changed the original exact prefix");
    const auto cold = report["cold"]["output"].get<std::string>();
    const auto sparse = report["sparse"]["output"].get<std::string>();
    report["quality"] = {
      {"same_tokens", report["cold"]["ids"] == report["sparse"]["ids"]},
      {"cold_has_code", cold.find("ORCHID-842") != std::string::npos},
      {"sparse_has_code", sparse.find("ORCHID-842") != std::string::npos},
      {"suffix_same_tokens",
       report["suffix_cold"]["ids"] == report["suffix_sparse"]["ids"]},
      {"suffix_has_code",
       report["suffix_sparse"]["output"].get<std::string>().find(
         "ORCHID-842") != std::string::npos}};
    require(report["quality"]["cold_has_code"] == true &&
              report["quality"]["sparse_has_code"] == true &&
              report["quality"]["suffix_has_code"] == true,
            "Sparse prefix failed the retained-fact quality fixture");
    report["passed"] = true;
    write();
    std::cout << "Sparse prefix model check: PASS\n";
  } catch (const std::exception &e) {
    report["error"] = e.what();
    write();
    std::cerr << e.what() << '\n';
    return 1;
  }
}

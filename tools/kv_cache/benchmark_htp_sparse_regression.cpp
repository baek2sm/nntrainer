// SPDX-License-Identifier: Apache-2.0
/** @file benchmark_htp_sparse_regression.cpp
 * @brief Common baseline/current benchmark for ordinary and exact KV paths.
 * Usage: htp_sparse_regression MODEL_DIR MODE REPEATS OUTPUT_JSON
 * MODE: plain-q8, plain-tq, exact-off, exact-on. Baseline supports all but on.
 */
#include <causal_lm.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <gemma4_htp_moe_causallm.h>
#include <htp_model_config.h>
#include <htp_runtime_config.h>
#include <iostream>
#include <model_registry.h>

namespace causallm {
class RegressionModel final : public Gemma4HtpMoECausalLM {
public:
  RegressionModel(json &c, json &g, json &n) :
    Transformer(sanitizeConfig(c), sanitizeGenerationConfig(g, c), n,
                ModelType::CAUSALLM),
    Gemma4HtpMoECausalLM(c, g, n) {}
  void freezeProducer() {
    device_prefill_capture_ = false;
    frozen_ = true;
  }

protected:
  void beforePrefillBlock(unsigned from, unsigned to) override {
    Gemma4HtpMoECausalLM::beforePrefillBlock(from, to);
    if (frozen_)
      device_prefill_history_.reset();
  }

private:
  bool frozen_ = false;
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
  if (argc != 5)
    return 2;
  json report{{"passed", false}, {"samples", json::array()}};
  auto write = [&] { std::ofstream(argv[4]) << report.dump(2) << '\n'; };
  try {
    const std::filesystem::path dir = argv[1];
    const std::string mode = argv[2];
    const unsigned repeats = std::stoul(argv[3]);
    require(repeats && repeats <= 20, "Invalid repeat count");
    const bool prefix = mode == "exact-off" || mode == "exact-on";
    require(prefix || mode == "plain-q8" || mode == "plain-tq", "Invalid mode");
    auto cfg = read(dir / "config.json"),
         gen = read(dir / "generation_config.json"),
         nc = read(dir / "nntr_config.json");
    nc["htp"]["kv_session"] = prefix;
    nc["htp"]["kv_paged"] = prefix;
    nc["htp"]["kv_prefix_cache"] = prefix;
    nc["htp"].erase("kv_sparse_prefix_cache");
    if (mode == "exact-on")
      nc["htp"]["kv_sparse_prefix_cache"] = true;
    nc["htp"]["kv_turboquant"] = mode == "plain-tq";
    nc["attention_kv_dtype"] = mode == "plain-tq" ? "turboquant_4bit_nc" : "q8";
    nc["htp"]["kv_seed_q"] = false;
    nc["skip_prefill"] = true;
    nc["num_to_generate"] = 32;
    for (const char *key : {"model_file_name", "tokenizer_file",
                            "moe_hadamard_signs", "act_scale_json"})
      if (nc.contains(key)) {
        const std::filesystem::path p = nc.at(key).get<std::string>();
        if (p.is_relative())
          nc[key] = (std::filesystem::absolute(dir) / p).string();
      }
    nntrainer::configureHtpRuntime(causallm::normalizeHtpRuntimeConfig(nc));
    causallm::registerAllModels();
    auto model = std::make_unique<causallm::RegressionModel>(cfg, gen, nc);
    model->initialize();
    model->load_weight(nc.at("model_file_name"));
    model->repack_weight();
    auto *c = dynamic_cast<causallm::CausalLM *>(model.get());
    require(c, "CausalLM unavailable");
    if (prefix)
      c->enableDeviceKVManager(4, 4);
    auto *tokenizer = model->getTokenizer();
    std::string source = "<bos><|turn>user\nThe secret code is ORCHID-842.\n";
    for (unsigned i = 0; i < 48; ++i)
      source += "Note " + std::to_string(i) +
                ": The village has an orchard, a library and a stone bridge.\n";
    source += "\nWhat is the secret code? Reply only with the "
              "code.<turn|>\n<|turn>model\n";
    const std::string miss =
      "<bos><|turn>user\nXylophone music: explain how sound travels through "
      "air in a few sentences.<turn|>\n<|turn>model\n";
    auto source_ids = tokenizer->Encode(source, true);
    auto branch_ids =
      std::vector<int>(source_ids.begin(), source_ids.begin() + 512);
    const auto tail =
      tokenizer->Encode("\nIdentify the secret code from the first sentence. "
                        "Reply only with that code.<turn|>\n<|turn>model\n",
                        false);
    branch_ids.insert(branch_ids.end(), tail.begin(), tail.end());
    const auto branch = tokenizer->Decode(branch_ids);
    require(tokenizer->Encode(branch, true) == branch_ids,
            "Branch does not roundtrip");
    report["mode"] = mode;
    report["source_tokens"] = source_ids.size();
    auto run = [&](const std::string &name, const std::string &text,
                   bool keep) {
      c->setSessionTokenPosition(0);
      const auto before = c->deviceKVManagerStats();
      const auto started = std::chrono::steady_clock::now();
      model->run(text, false, "", "", false);
      const auto wall = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - started)
                          .count();
      const auto after = c->deviceKVManagerStats();
      const auto &m = c->getPerformanceMetrics();
      require(!m.prompt_truncated_tokens &&
                !c->getLastGeneratedTokenIds().empty(),
              "Invalid generation");
      json sample{{"case", name},
                  {"ids", c->getLastGeneratedTokenIds()},
                  {"ttft_ms", m.ttft_ms},
                  {"prefill_ms", m.prefill_duration_ms},
                  {"prefill_tokens", m.prefill_tokens},
                  {"decode_ms_per_token", m.decode_steady_ms},
                  {"decode_tokens", m.decode_steady_tokens},
                  {"peak_memory_kb", m.peak_memory_kb},
                  {"kv_bytes", c->getDeviceKVCacheBytes()},
                  {"prefix_host_bytes", c->getDeviceKVPrefixBytes()},
                  {"reused_tokens", after.reused_tokens - before.reused_tokens},
                  {"wall_ms", wall}};
      if (keep) {
        report["samples"].push_back(sample);
        write();
      }
      return sample;
    };
    run("warmup", source, false);
    model->freezeProducer();
    for (unsigned i = 0; i < repeats; ++i) {
      if (!prefix)
        run("ordinary_prefill_decode", source, true);
      else {
        auto hit = run("exact_full", source, true);
        require(hit["reused_tokens"].get<unsigned>() == source_ids.size() - 1,
                "Full exact hit changed");
        auto partial = run("exact_partial", branch, true);
        require(partial["reused_tokens"] == 512, "Partial exact hit changed");
        auto missed = run("prefix_miss", miss, true);
        require(missed["reused_tokens"] == 0, "Unexpected reuse on miss");
      }
    }
    report["passed"] = true;
    write();
    std::cout << "KV existing path regression benchmark: PASS\n";
  } catch (const std::exception &e) {
    report["error"] = e.what();
    write();
    std::cerr << e.what() << '\n';
    return 1;
  }
}

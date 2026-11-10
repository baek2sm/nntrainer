// SPDX-License-Identifier: Apache-2.0
/** @file benchmark_htp_sparse_benefit.cpp
 * @brief Same Q8 request: KV-cold, exact prefix, sparse prefix TTFT matrix.
 * Usage: htp_sparse_benefit MODEL_DIR OUTPUT_JSON REPEATS [CASE]
 * [all|cold|cache] K means 1024 tokens. Model load and source preparation are
 * reported apart.
 */
#include <chrono>
#include <filesystem>
#include <fstream>
#include <gemma4_htp_moe_causallm.h>
#include <htp_model_config.h>
#include <htp_runtime_config.h>
#include <iostream>
#include <model_registry.h>
#include <set>
#include <streamer.h>

namespace causallm {
class SparseBenefitModel final : public Gemma4HtpMoECausalLM {
public:
  SparseBenefitModel(json &c, json &g, json &n) :
    Transformer(sanitizeConfig(c), sanitizeGenerationConfig(g, c), n,
                ModelType::CAUSALLM),
    Gemma4HtpMoECausalLM(c, g, n) {}
  void freezeProducer() {
    device_prefill_capture_ = false;
    frozen_ = true;
  }
  std::unique_ptr<DeviceKVManager> suspendLookup() {
    return std::move(device_kv_manager_);
  }
  void resumeLookup(std::unique_ptr<DeviceKVManager> manager) {
    device_kv_manager_ = std::move(manager);
  }
  void sparse(bool enabled) {
#ifdef ENABLE_HEXKL
    if (enabled)
      device_kv_manager_->setSparseRestore(
        [this](uint32_t handle,
               const std::vector<nntrainer::kv::SparsePrefixRun> &runs,
               size_t rows) {
          restoreSparseDevicePrefill(handle, runs, rows);
        });
    else
      device_kv_manager_->setSparseRestore({});
#else
    (void)enabled;
    throw std::runtime_error("HTP is unavailable");
#endif
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
using Clock = std::chrono::steady_clock;
static json read(const std::filesystem::path &path) {
  std::ifstream in(path);
  json j;
  in >> j;
  return j;
}
static void require(bool value, const char *why) {
  if (!value)
    throw std::runtime_error(why);
}
struct Observer {
  BaseStreamer base;
  Clock::time_point first{};
  static int put(BaseStreamer *b, const char *s) {
    auto *o = reinterpret_cast<Observer *>(b);
    if (s && *s && o->first == Clock::time_point{})
      o->first = Clock::now();
    return 0;
  }
  static void end(BaseStreamer *) {}
};
int main(int argc, char **argv) {
  if (argc < 4 || argc > 6)
    return 2;
  json report{{"passed", false}, {"samples", json::array()}};
  auto write = [&] { std::ofstream(argv[2]) << report.dump(2) << '\n'; };
  try {
    const std::filesystem::path dir = argv[1];
    const unsigned repeats = std::stoul(argv[3]);
    require(repeats && repeats <= 10, "Invalid repeat count");
    const std::string filter = argc >= 5 ? argv[4] : "all";
    const std::string experiment = argc == 6 ? argv[5] : "all";
    require(experiment == "all" || experiment == "cold" ||
              experiment == "cache",
            "Invalid experiment mode");
    const bool cold_only = experiment == "cold";
    report["experiment"] = experiment;
    auto cfg = read(dir / "config.json"),
         gen = read(dir / "generation_config.json"),
         nc = read(dir / "nntr_config.json");
    nc["attention_kv_dtype"] = "q8";
    nc["skip_prefill"] = true;
    nc["init_seq_len"] = 1024;
    nc["max_seq_len"] = 28672;
    nc["num_to_generate"] = 16;
    for (const char *key : {"kv_session", "kv_paged", "kv_prefix_cache",
                            "kv_sparse_prefix_cache"})
      nc["htp"][key] = !cold_only;
    nc["htp"]["kv_turboquant"] = false;
    nc["htp"]["kv_seed_q"] = false;
    nc["htp"]["kv_request_reserve"] = true;
    nc["htp"]["page_cache"] = 0;
    nc["moe"]["cache_grow"] = 0;
    for (const char *key : {"model_file_name", "tokenizer_file",
                            "moe_hadamard_signs", "act_scale_json"})
      if (nc.contains(key)) {
        const std::filesystem::path p = nc.at(key).get<std::string>();
        if (p.is_relative())
          nc[key] = (std::filesystem::absolute(dir) / p).string();
      }
    nntrainer::configureHtpRuntime(causallm::normalizeHtpRuntimeConfig(nc));
    causallm::registerAllModels();
    const auto load_start = Clock::now();
    auto model = std::make_unique<causallm::SparseBenefitModel>(cfg, gen, nc);
    model->initialize();
    model->load_weight(nc.at("model_file_name"));
    model->repack_weight();
    report["model_load_ms"] =
      std::chrono::duration<double, std::milli>(Clock::now() - load_start)
        .count();
    report["config"] = nc;
    auto *tokenizer = model->getTokenizer();
    std::string filler = "<bos><|turn>user\nThe secret code is ORCHID-842.\n";
    for (unsigned i = 0; i < 3500; ++i)
      filler += "Record " + std::to_string(i) +
                ": The village has an orchard, a library and a stone bridge.\n";
    const auto body = tokenizer->Encode(filler, true);
    constexpr size_t context = 24576;
    require(body.size() > context + 1, "Insufficient source body");
    const std::vector<int> source_ids(body.begin(), body.begin() + context + 1);
    const std::vector<int> cached(source_ids.begin(),
                                  source_ids.begin() + context);
    auto textOf = [&](const std::vector<int> &ids) {
      const auto text = tokenizer->Decode(ids);
      require(tokenizer->Encode(text, true) == ids, "Token roundtrip differs");
      return text;
    };
    (void)textOf(source_ids);
    const auto tail = tokenizer->Encode("\nWhat is the secret code? Reply only "
                                        "with the code.<turn|>\n<|turn>model\n",
                                        false);
    struct Case {
      std::string name, text;
      size_t context_rows, exact_rows, sparse_rows;
      json holes;
    };
    std::vector<Case> cases;
    auto add = [&](const std::string &name,
                   std::vector<std::pair<size_t, size_t>> holes,
                   const std::vector<int> &suffix, size_t expected_exact) {
      for (size_t shift = 0; shift < 64; ++shift) {
        auto ids = cached;
        json cuts = json::array();
        for (auto it = holes.rbegin(); it != holes.rend(); ++it) {
          const size_t start = it->first + shift, end = start + it->second;
          require(end <= ids.size(), "Invalid deletion range");
          ids.erase(ids.begin() + start, ids.begin() + end);
          cuts.push_back({start, end});
        }
        const size_t kept = ids.size();
        ids.insert(ids.end(), suffix.begin(), suffix.end());
        const auto text = tokenizer->Decode(ids);
        if (tokenizer->Encode(text, true) != ids)
          continue;
        size_t common = 0;
        while (common < ids.size() && common < cached.size() &&
               ids[common] == cached[common])
          ++common;
        if (common / 4096 * 4096 != expected_exact)
          continue;
        // The policy should reuse the entire retained context. A few shared
        // suffix tokens may match too, so actual counts are reported
        // separately.
        cases.push_back({name, text, kept, expected_exact, kept, cuts});
        return;
      }
      throw std::runtime_error("Cannot construct stable case " + name);
    };
    add("24k_exact_control", {}, tail, 24576);
    add("4k_at_0k", {{512, 20480}}, tail, 0);
    add("8k_at_4k", {{4096, 16384}}, tail, 4096);
    add("20k_at_4k", {{4096, 4096}}, tail, 4096);
    add("20k_at_12k", {{12288, 4096}}, tail, 12288);
    add("20k_multi_holes",
        {{4096, 1024}, {8192, 1024}, {12288, 1024}, {16384, 1024}}, tail, 4096);
    add("24k_one_token_at_4k", {{4096, 1}}, tail, 4096);
    auto miss = tokenizer->Encode(
      "<bos><|turn>user\nXylophone harmonics describe musical vibrations.\n",
      true);
    miss.insert(miss.end(), cached.begin() + miss.size(),
                cached.begin() + 4096);
    miss.insert(miss.end(), tail.begin(), tail.end());
    const auto miss_text = textOf(miss);
    const auto miss_runs = nntrainer::kv::matchSparseTokens(
      reinterpret_cast<const uint32_t *>(cached.data()), cached.size(),
      reinterpret_cast<const uint32_t *>(miss.data()), miss.size() - 1);
    require(nntrainer::kv::sparsePrefixLength(miss_runs) < 64,
            "Miss fixture unexpectedly sparse-matches");
    cases.push_back({"4k_cache_miss", miss_text, 4096, 0, 0, json::array()});
    for (const auto &c : cases)
      report["cases"].push_back(
        {{"name", c.name},
         {"context_tokens", c.context_rows},
         {"input_tokens", tokenizer->Encode(c.text, true).size()},
         {"expected_exact_rows", c.exact_rows},
         {"expected_sparse_context_rows", c.sparse_rows},
         {"deleted_source_ranges", c.holes}});
    report["source_tokens"] = source_ids.size();
    report["source_cached_context_tokens"] = context;
    static const BaseStreamerVTable vt = {Observer::put, Observer::end};
    auto run = [&](const std::string &text, const std::string &mode) {
      model->setSessionTokenPosition(0);
      std::unique_ptr<causallm::DeviceKVManager> held;
      if (mode == "cold")
        held = model->suspendLookup();
      else if (mode != "producer" && mode != "warmup")
        model->sparse(mode == "sparse");
      Observer observer;
      observer.base.vtable = &vt;
      model->setStreamer(&observer.base);
      const auto b = model->deviceKVManagerStats();
      const auto fb = model->deviceKVPrefixFlashStats();
      const auto started = Clock::now();
      try {
        model->run(text, false, "", "", false);
      } catch (...) {
        if (held)
          model->resumeLookup(std::move(held));
        throw;
      }
      const auto ended = Clock::now();
      const auto a = model->deviceKVManagerStats();
      const auto fa = model->deviceKVPrefixFlashStats();
      if (held)
        model->resumeLookup(std::move(held));
      model->setStreamer(nullptr);
      const auto m = model->getPerformanceMetrics();
      if (cold_only)
        require(model->getDeviceKVPrefixBytes() == 0 && a.reused_tokens == 0,
                "No-cache baseline retained or reused prefix KV");
      require(!m.prompt_truncated_tokens &&
                observer.first != Clock::time_point{},
              "Truncated request or no first token");
      return json{
        {"mode", mode},
        {"ids", model->getLastGeneratedTokenIds()},
        {"output", model->getOutput()},
        {"ttft_ms", m.ttft_ms},
        {"external_ttft_ms",
         std::chrono::duration<double, std::milli>(observer.first - started)
           .count()},
        {"wall_ms",
         std::chrono::duration<double, std::milli>(ended - started).count()},
        {"prefill_ms", m.prefill_duration_ms},
        {"prefill_tokens", m.prefill_tokens},
        {"reused_tokens", a.reused_tokens - b.reused_tokens},
        {"sparse_hits", a.sparse_hits - b.sparse_hits},
        {"flash_hits", fa.flash_hits - fb.flash_hits},
        {"flash_read_ms", fa.read_ms - fb.read_ms},
        {"decode_ms_per_token", m.decode_steady_ms},
        {"decode_tokens", m.decode_steady_tokens},
        {"kv_bytes", model->getDeviceKVCacheBytes()},
        {"prefix_host_bytes", model->getDeviceKVPrefixBytes()},
        {"peak_memory_kb", m.peak_memory_kb},
        {"position", model->getKvLen()}};
    };
    write();
    std::cout << "BENEFIT warmup\n" << std::flush;
    report["warmup"] =
      run("<bos><|turn>user\nSay READY.<turn|>\n<|turn>model\n", "warmup");
    if (cold_only) {
      // Warm model weights/workspaces on the same long source, then discard
      // request KV normally. No prefix manager or retained source is created.
      std::cout << "BENEFIT cold weight warmup\n" << std::flush;
      report["cold_weight_warmup"] = run(textOf(source_ids), "cold");
      report["cache_tier"] = "none; prefix caching and TurboQuant disabled";
    } else {
      model->setSessionTokenPosition(0);
      model->enableDeviceKVManager(1, 3);
      report["producer"] = json::array();
      for (size_t boundary : {size_t(4096), size_t(12288), size_t(24576)}) {
        bool prepared = false;
        for (size_t extra = 1; extra < 64; ++extra) {
          std::vector<int> ids(body.begin(), body.begin() + boundary + extra);
          const auto text = tokenizer->Decode(ids);
          if (tokenizer->Encode(text, true) != ids)
            continue;
          std::cout << "BENEFIT producer " << boundary << "\n" << std::flush;
          auto result = run(text, "producer");
          result["native_boundary"] = boundary;
          result["input_tokens"] = ids.size();
          report["producer"].push_back(std::move(result));
          prepared = true;
          break;
        }
        require(prepared, "Cannot prepare a canonical source boundary");
      }
      report["cache_tier"] = "DRAM; common producer frames at 4K, 12K, 24K";
    }
    model->freezeProducer();
    write();
    for (const auto &c : cases) {
      if (filter != "all" && filter != c.name)
        continue;
      for (unsigned round = 0; round < repeats; ++round) {
        json samples = json::object();
        const std::vector<std::string> order =
          cold_only ? std::vector<std::string>{"cold"}
          : experiment == "cache"
            ? (round % 2 ? std::vector<std::string>{"sparse", "exact"}
                         : std::vector<std::string>{"exact", "sparse"})
          : round % 2 ? std::vector<std::string>{"sparse", "exact", "cold"}
                      : std::vector<std::string>{"cold", "exact", "sparse"};
        for (const auto &mode : order) {
          std::cout << "BENEFIT " << c.name << " round=" << round << " " << mode
                    << "\n"
                    << std::flush;
          samples[mode] = run(c.text, mode);
          const auto reused = samples[mode]["reused_tokens"].get<size_t>();
          require(mode != "cold" || !reused, "Cold path reused KV");
          require(mode != "exact" || reused == c.exact_rows,
                  "Exact reuse differs from fixture");
          require(mode != "sparse" || reused >= c.sparse_rows,
                  "Sparse context did not hit");
          report["samples"].push_back(
            {{"case", c.name}, {"round", round}, {"result", samples[mode]}});
          write();
        }
        if (experiment == "all")
          require(samples["cold"]["ids"] == samples["exact"]["ids"],
                  "Exact output differs from cold");
      }
    }
    report["passed"] = true;
    write();
    std::cout << "BENEFIT_RESULT PASS\n";
  } catch (const std::exception &e) {
    report["error"] = e.what();
    write();
    std::cerr << "BENEFIT_ERROR " << e.what() << '\n';
    return 1;
  }
}

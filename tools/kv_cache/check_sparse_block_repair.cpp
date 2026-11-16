// SPDX-License-Identifier: Apache-2.0
/** @file check_sparse_block_repair.cpp
 * @brief Public API smoke: bake one superset, RoPE-only and block repair.
 * Usage: sparse_block_repair_check MODEL_DIR OUTPUT_JSON
 */
#include <causal_lm.h>
#include <factory.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <model_registry.h>

using json = causallm::json;
static json read(const std::filesystem::path &p) {
  std::ifstream in(p);
  json result;
  in >> result;
  return result;
}
int main(int argc, char **argv) {
  if (argc != 3)
    return 2;
  json report = {{"passed", false}};
  try {
    const auto dir = std::filesystem::absolute(argv[1]);
    auto cfg = read(dir / "config.json"),
         gen = read(dir / "generation_config.json"),
         nc = read(dir / "nntr_config.json");
    nc["init_seq_len"] = 512;
    nc["max_seq_len"] = 1024;
    nc["num_to_generate"] = 8;
    nc["skip_prefill"] = true;
    nc["repetition_penalty"] = 1.0;
    for (const char *key : {"model_file_name", "tokenizer_file"}) {
      std::filesystem::path p = nc.at(key).get<std::string>();
      if (p.is_relative())
        nc[key] = (dir / p).string();
    }
    causallm::registerAllModels();
    auto m = causallm::Factory::Instance().create(
      cfg.at("architectures").at(0).get<std::string>(), cfg, gen, nc);
    if (!m)
      throw std::runtime_error("Model architecture not registered");
    m->initialize();
    auto *c = dynamic_cast<causallm::CausalLM *>(m.get());
    if (!c)
      throw std::runtime_error("CausalLM required");
    m->load_weight(nc.at("model_file_name"));
    std::string header = "You are a helpful assistant.\n";
    std::string removed, kept;
    for (unsigned i = 0; i < 24; ++i) {
      if (i < 8)
        removed += "Unused tool: convert red circles to blue squares.\n";
      kept += "The access code is ORCHID-842. Keep this fact.\n";
    }
    const std::string tail = "Return only the access code.\n";
    const auto source = header + removed + kept + tail,
               subset = header + kept + tail;
    auto ids = m->getTokenizer()->Encode(source, true);
    report["source_tokens"] = ids.size();
    c->run(source, false, "", "", false);
    auto asset =
      c->captureSparseKVPrefix(std::vector<uint32_t>(ids.begin(), ids.end()));
    auto run = [&](bool repair) {
      c->setSessionTokenPosition(0);
      c->setSparseKVPrefix(asset);
      c->setSparseKVRecompute({repair, 32, .15});
      c->run(subset, false, "", "", false);
      const auto &s = c->getSparseKVRecomputeStats();
      return json{{"text", c->getOutput(0)},
                  {"ids", c->getLastGeneratedTokenIds()},
                  {"prefix_tokens", s.prefix_tokens},
                  {"first_layer_tokens", s.full_first_layer_tokens},
                  {"probe_tokens", s.probe_key_tokens},
                  {"budget_tokens", s.budget_tokens},
                  {"selected_tokens", s.selected_tokens},
                  {"replay_calls", s.replay_calls},
                  {"probe_ms", s.probe_ms},
                  {"replay_ms", s.replay_ms}};
    };
    report["rope_only"] = run(false);
    report["block_repair"] = run(true);
    const auto &s = c->getSparseKVRecomputeStats();
    if (!s.full_first_layer_tokens || s.selected_tokens > s.budget_tokens)
      throw std::runtime_error("Sparse repair was not exercised within budget");
    c->setSparseKVPrefix(nullptr);
    c->setSessionTokenPosition(0);
    c->run(subset, false, "", "", false);
    report["cold_subset"] = {{"text", c->getOutput(0)},
                             {"ids", c->getLastGeneratedTokenIds()}};
    report["passed"] = true;
  } catch (const std::exception &e) {
    report["error"] = e.what();
  }
  std::ofstream out(argv[2]);
  out << report.dump(2) << '\n';
  std::cout << report.dump(2) << '\n';
  return report.at("passed").get<bool>() ? 0 : 1;
}

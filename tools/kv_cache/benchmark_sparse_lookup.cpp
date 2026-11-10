// SPDX-License-Identifier: Apache-2.0
/** @file benchmark_sparse_lookup.cpp
 * @brief Standalone exact radix lookup overhead versus optional sparse search.
 * Build with -DSPARSE_LOOKUP_API=1 for the sparse-capable header.
 */
#include <chrono>
#include <device_kv_manager.h>
#include <iomanip>
#include <iostream>
#include <string>

int main(int argc, char **argv) {
  using Manager = causallm::DeviceKVManager;
  const bool enable = argc > 1 && std::string(argv[1]) == "on";
  uint32_t next = 1;
  Manager manager([&](unsigned, unsigned) { return next++; },
                  [](uint32_t, unsigned) {}, [](uint32_t) {}, [] {}, 2, 8);
  manager.setPrefixLengths([](uint32_t) {
    return std::vector<unsigned>{512, 4096};
  });
#ifdef SPARSE_LOOKUP_API
  if (enable)
    manager.setSparseRestore(
      [](uint32_t, const std::vector<nntrainer::kv::SparsePrefixRun> &,
         size_t) {
        throw Manager::PrefixUnavailable("No shifted rows in this benchmark");
      });
#else
  if (enable)
    return 2;
#endif
  Manager::Tokens source(4096);
  for (unsigned i = 0; i < 8; ++i) {
    for (unsigned j = 0; j < source.size(); ++j)
      source[j] = 100 + j % 1000;
    source.front() = 10000 + i;
    manager.publish(source, source.size(), 0);
  }
  auto full = source;
  full.push_back(999999);
  Manager::Tokens partial(source.begin(), source.begin() + 512);
  partial.insert(partial.end(), 128, 999998);
  Manager::Tokens miss(256, 999997);
  std::cout << "{\"sparse\":" << (enable ? "true" : "false") << ",\"cases\":{";
  bool comma = false;
  for (const auto &entry : {std::make_pair("exact_full", full),
                            std::make_pair("exact_partial", partial),
                            std::make_pair("miss", miss)}) {
    const size_t expected = std::string(entry.first) == "exact_full"      ? 4096
                            : std::string(entry.first) == "exact_partial" ? 512
                                                                          : 0;
    for (unsigned i = 0; i < 100; ++i)
      if (manager.reuse(entry.second) != expected)
        return 1;
    const auto start = std::chrono::steady_clock::now();
    constexpr unsigned iterations = 200000;
    for (unsigned i = 0; i < iterations; ++i)
      if (manager.reuse(entry.second) != expected)
        return 1;
    const double us = std::chrono::duration<double, std::micro>(
                        std::chrono::steady_clock::now() - start)
                        .count() /
                      iterations;
    if (comma)
      std::cout << ',';
    comma = true;
    std::cout << '\"' << entry.first << "\":" << std::setprecision(9) << us;
  }
  std::cout << "}}\n";
}

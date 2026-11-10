// SPDX-License-Identifier: Apache-2.0
/** @file prefill_prefix_check.cpp
 * @brief Immutable prefill pages, window layout and transactional validation.
 * Link with libcausallm and libnntrainer; no DSP/model weights required.
 */
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fp16.h>
#include <iostream>
#include <kv_cache_manager.h>
#include <map>
#include <stdexcept>

static void require(bool ok) {
  if (!ok)
    throw std::runtime_error("Prefill prefix check failed");
}
static void checkSparseGather() {
  using Manager = causallm::KVCacheManager;
  using nntrainer::kv::SparseRope;
  for (const bool fp32 : {false, true}) {
    Manager cache;
    cache.allocate(2, 1, 512, std::vector<unsigned>{8, 4},
                   fp32 ? ml::train::TensorDim::DataType::FP32
                        : ml::train::TensorDim::DataType::UINT16,
                   ml::train::TensorDim::Format::NCHW, {192, 512}, {64, 0});
    const std::vector<SparseRope> rope{{{0.03f, 0.001f}, false},
                                       {{0.07f, 0.0f}, false}};
    auto store = [&](unsigned l, unsigned p, unsigned d, float k, float v) {
      const size_t offset =
        size_t(p - cache.getBase(l)) * cache.getKeyCache(l).width() + d;
      if (fp32) {
        cache.getKeyCache(l).getData<float>()[offset] = k;
        cache.getValueCache(l).getData<float>()[offset] = v;
      } else {
        cache.getKeyCache(l).getData<uint16_t>()[offset] =
          nntrainer::compute_fp32_to_fp16(k);
        cache.getValueCache(l).getData<uint16_t>()[offset] =
          nntrainer::compute_fp32_to_fp16(v);
      }
    };
    auto fill = [&] {
      for (unsigned l = 0; l < 2; ++l)
        for (unsigned p = cache.getBase(l); p < 192; ++p)
          for (unsigned h = 0; h < cache.getKeyCache(l).width() / 4; ++h) {
            float key[]{float(p + 1) * 0.01f, 0.2f, float(h + 1), 0.4f};
            rope[l].reposition(key, 0, p);
            for (unsigned d = 0; d < 4; ++d)
              store(l, p, h * 4 + d, key[d], float(p) * 0.01f + d);
          }
    };
    fill();
    cache.setPosition(128);
    const auto a = cache.capturePrefix(128, nullptr, true);
    require(cache.compact(192));
    fill();
    cache.setPosition(192);
    const auto b = cache.capturePrefix(192, nullptr, true);
    const std::vector<nntrainer::kv::SparsePrefixRun> runs{{0, 0, 65},
                                                           {80, 65, 79}};
    require(!Manager::coversSparsePrefix(*a, runs));
    require(!Manager::coversSparsePrefix(*b, runs));
    require(Manager::coversSparsePrefix(*b, {{128, 0, 64}}));
    require(!Manager::coversSparsePrefix(*b, {{128, 1, 64}}));
    auto metadata = *b;
    for (auto &layer : metadata.layers)
      layer.pages.clear();
    require(Manager::coversSparsePrefix(metadata, {{128, 0, 64}}));
    auto gathered = Manager::gatherSparsePrefix({a, b}, runs, {4, 4}, rope);
    require(gathered->position == 144 && gathered->layers[0].base == 16);
    cache.restorePrefix(*gathered);
    auto load = [&](unsigned l, unsigned row, unsigned d, bool key) {
      auto &tensor = key ? cache.getKeyCache(l) : cache.getValueCache(l);
      const size_t offset = size_t(row) * tensor.width() + d;
      return fp32 ? tensor.getData<float>()[offset]
                  : nntrainer::compute_fp16_to_fp32(
                      tensor.getData<uint16_t>()[offset]);
    };
    for (unsigned l = 0; l < 2; ++l)
      for (unsigned dst = gathered->layers[l].base; dst < 144; ++dst) {
        const unsigned src = dst < 65 ? dst : dst + 15;
        for (unsigned h = 0; h < cache.getKeyCache(l).width() / 4; ++h) {
          float expected[]{float(src + 1) * 0.01f, 0.2f, float(h + 1), 0.4f};
          rope[l].reposition(expected, 0, dst);
          for (unsigned d = 0; d < 4; ++d) {
            require(std::fabs(load(l, dst - cache.getBase(l), h * 4 + d, true) -
                              expected[d]) < (fp32 ? 2e-6f : 0.003f));
            const auto &original = src < 128 ? a : b;
            const auto &original_layer = original->layers[l];
            const size_t original_row = src - original_layer.base;
            const auto *original_v =
              original_layer.pages[original_row / 64]->valueData() +
              ((original_row % 64) * original_layer.width + h * 4 + d) *
                original->element_bytes;
            float expected_v;
            if (fp32)
              std::memcpy(&expected_v, original_v, 4);
            else {
              uint16_t bits;
              std::memcpy(&bits, original_v, 2);
              expected_v = nntrainer::compute_fp16_to_fp32(bits);
            }
            require(load(l, dst - cache.getBase(l), h * 4 + d, false) ==
                    expected_v);
          }
        }
      }
    bool rejected = false;
    try {
      Manager::gatherSparsePrefix({b}, runs, {4, 4}, rope);
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    require(rejected);
    rejected = false;
    try {
      Manager::gatherSparsePrefix({a, b}, {{80, 1, 3}}, {4, 4}, rope);
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    require(rejected);
    // Source snapshots remain intact and can still restore exact prefixes.
    cache.restorePrefix(*a);
    require(cache.getPosition() == 128 && cache.getBase(0) == 0);
  }
}

int main() {
  checkSparseGather();
  using Manager = causallm::KVCacheManager;
  Manager cache;
  cache.allocate(2, 1, 512, std::vector<unsigned>{8, 4},
                 ml::train::TensorDim::DataType::UINT16,
                 ml::train::TensorDim::Format::NCHW, {192, 512}, {64, 0});
  auto fill = [&] {
    for (unsigned l = 0; l < 2; ++l) {
      auto &k = cache.getKeyCache(l), &v = cache.getValueCache(l);
      for (unsigned r = 0; r < k.height(); ++r)
        for (unsigned d = 0; d < k.width(); ++d) {
          const auto i = size_t(r) * k.width() + d;
          k.getData<uint16_t>()[i] = uint16_t((r + cache.getBase(l)) * 13 + d);
          v.getData<uint16_t>()[i] = uint16_t((r + cache.getBase(l)) * 17 + d);
        }
    }
  };
  fill();
  cache.setPosition(192);
  const auto a = cache.capturePrefix(64);
  const auto b = cache.capturePrefix(128, a.get());
  require(a->layers[0].pages[0] == b->layers[0].pages[0]);
  require(b->layers[0].pages[0] != b->layers[0].pages[1]);
  std::fill_n(cache.getKeyCache(0).getData<uint16_t>(), 128 * 8, 0xffff);
  cache.restorePrefix(*a);
  require(cache.getPosition() == 64);
  require(cache.getKeyCache(0).getData<uint16_t>()[63 * 8] == 63 * 13);
  cache.restorePrefix(*b);
  require(cache.getKeyCache(0).getData<uint16_t>()[127 * 8] == 127 * 13);
  require(cache.compact(128) && cache.getBase(0) == 64);
  fill();
  cache.setPosition(192);
  const auto partial = cache.capturePrefix(159, b.get());
  const auto c = cache.capturePrefix(192, partial.get());
  require(partial->layers[0].pages.back()->key.size() == 31u * 8u * 2u);
  require(c->layers[0].pages.back() != partial->layers[0].pages.back());
  cache.restorePrefix(*partial);
  require(cache.getPosition() == 159);
  require(cache.getKeyCache(0).getData<uint16_t>()[94 * 8] == 158 * 13);
  cache.restorePrefix(*c);
  require(c->layers[0].base == 64 && c->layers[0].pages.size() == 2);
  require(c->layers[0].pages[0] == b->layers[0].pages[1]);
  require(c->layers[1].pages[0] == b->layers[1].pages[0]);
  cache.restorePrefix(*a);
  cache.restorePrefix(*c);
  require(cache.getBase(0) == 64 && cache.getPosition() == 192);
  require(cache.getKeyCache(0).getData<uint16_t>()[0] == 64 * 13);
  require(cache.getKeyCache(0).getData<uint16_t>()[127 * 8] == 191 * 13);
  auto invalid = *c;
  invalid.layers[1].pages.clear();
  try {
    cache.restorePrefix(invalid);
    require(false);
  } catch (const std::invalid_argument &) {
    require(cache.getPosition() == 192 && cache.getBase(0) == 64);
  }
  // A read-only final view has exactly the same live KV and dropped-row maxima
  // as the original compaction, without modifying any prefill payload.
  Manager moved, viewed;
  for (auto *m : {&moved, &viewed}) {
    m->allocate(1, 1, 256, std::vector<unsigned>{8},
                ml::train::TensorDim::DataType::UINT16,
                ml::train::TensorDim::Format::NCHW, {192}, {64});
    for (unsigned i = 0; i < 192u * 8u; ++i) {
      m->getKeyCache(0).getData<uint16_t>()[i] = uint16_t(i);
      m->getValueCache(0).getData<uint16_t>()[i] = uint16_t(i + 3);
    }
    m->setPosition(159);
  }
  require(moved.compact(159));
  viewed.prepareDecodeView(159);
  require(viewed.getDecodeOffset(0) == 95);
  require(viewed.getKeyCache(0).getData<uint16_t>()[0] == 0);
  require(viewed.droppedAmaxK(0) == moved.droppedAmaxK(0));
  require(viewed.droppedAmaxV(0) == moved.droppedAmaxV(0));
  const auto mv = moved.capturePrefix(159), vw = viewed.capturePrefix(159);
  require(mv->layers[0].base == vw->layers[0].base);
  for (size_t i = 0; i < mv->layers[0].pages.size(); ++i) {
    const auto &x = mv->layers[0].pages[i], &y = vw->layers[0].pages[i];
    require(x->key == y->key && x->value == y->value);
  }
  try {
    viewed.compact(159);
    require(false);
  } catch (const std::logic_error &) {
  }
  viewed.restorePrefix(*vw);
  require(viewed.getDecodeOffset(0) == 0);
  require(viewed.getKeyCache(0).getData<uint16_t>()[0] == 95u * 8u);

  // Borrowed RPC slabs survive owner release. Sliding compaction freezes only
  // pages in the mutated layer before its original memmove.
  std::shared_ptr<const Manager::PrefixSnapshot> pinned, still_borrowed;
  std::map<void *, std::shared_ptr<void>> slabs;
  {
    Manager borrowed;
    borrowed.setSharedAllocator(
      [&](size_t bytes) -> void * {
        void *p = std::malloc(bytes);
        if (p)
          slabs.emplace(p, std::shared_ptr<void>(p, std::free));
        return p;
      },
      [&](void *p) { slabs.erase(p); });
    borrowed.setSharedRetain([&](void *p) { return slabs.at(p); });
    borrowed.allocate(1, 1, 256, std::vector<unsigned>{8},
                      ml::train::TensorDim::DataType::UINT16,
                      ml::train::TensorDim::Format::NCHW, {192}, {64});
    auto *k = borrowed.getKeyCache(0).getData<uint16_t>();
    auto *v = borrowed.getValueCache(0).getData<uint16_t>();
    for (unsigned i = 0; i < 192u * 8u; ++i)
      k[i] = v[i] = uint16_t(i);
    borrowed.setPosition(159);
    pinned = borrowed.capturePrefix(159);
    require(pinned->layers[0].pages[0]->key_view ==
            reinterpret_cast<unsigned char *>(k));
    const auto staged = borrowed.capturePrefix(159, pinned.get(), true);
    require(borrowed.prefixPayloadBytes(159) == 159u * 8u * 2u * 2u);
    for (const auto &page : staged->layers[0].pages)
      require(!page->key_view && !page->value_view && !page->key_owner);
    require(borrowed.compact(159)); // unaligned sliding base is preserved
    require(reinterpret_cast<const uint16_t *>(
              staged->layers[0].pages[0]->keyData())[10] == 10);
    require(!pinned->layers[0].pages[0]->key_view);
    const auto live = borrowed.capturePrefix(159);
    require(live->layers[0].base == 95);
    borrowed.restorePrefix(*live);
    require(borrowed.getPosition() == 159);
    still_borrowed = borrowed.capturePrefix(159);
    require(still_borrowed->layers[0].pages[0]->key_view != nullptr);
  }
  require(slabs.empty());
  require(reinterpret_cast<const uint16_t *>(
            still_borrowed->layers[0].pages[0]->keyData())[0] == 95u * 8u);
  still_borrowed.reset();
  require(reinterpret_cast<const uint16_t *>(
            pinned->layers[0].pages[0]->keyData())[10] == 10);
  std::cout << "Prefill shared pages, window layout, invalid restore: PASS\n";
}

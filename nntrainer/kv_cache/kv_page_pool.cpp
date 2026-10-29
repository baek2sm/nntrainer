// SPDX-License-Identifier: Apache-2.0
/** @file kv_page_pool.cpp
 * @brief Budgeted page reuse with allocation ownership held by every view.
 * @author Codex (OpenAI)
 */
#include <algorithm>
#include <cstring>
#include <kv_cache/kv_cache_spec.h>
#include <kv_cache/kv_page_pool.h>
#include <utility>

namespace nntrainer::kv {

KvPagePool::KvPagePool(size_t budget_bytes,
                       std::shared_ptr<MemAllocator> allocator) :
  allocator_(std::move(allocator)), budget_bytes_(budget_bytes) {
  if (!allocator_ || !allocator_->isHostAddressable() ||
      allocator_->isDeviceVisible() || allocator_->isSVM() ||
      allocator_->needsRegister())
    throw std::invalid_argument("KV page pool requires a plain host allocator");
}

std::shared_ptr<KvStorage> KvPagePool::acquire(size_t bytes) {
  if (!bytes)
    throw std::invalid_argument("KV page size must be positive");
  for (auto &page : pages_)
    if (page.use_count() == 1 && page->bytes() == bytes) {
      std::memset(page->data(), 0, bytes);
      return page;
    }
  const size_t aligned = checkedAdd(bytes, 63) & ~size_t{63};
  const size_t next = checkedAdd(reserved_bytes_, aligned);
  if (budget_bytes_ && next > budget_bytes_)
    throw std::bad_alloc();
  auto page = KvStorage::create(bytes, allocator_);
  pages_.push_back(page);
  reserved_bytes_ = next;
  return page;
}

void KvPagePool::trim() {
  pages_.erase(std::remove_if(pages_.begin(), pages_.end(),
                              [&](const std::shared_ptr<KvStorage> &page) {
                                if (page.use_count() != 1)
                                  return false;
                                reserved_bytes_ -=
                                  (page->bytes() + 63) & ~size_t{63};
                                return true;
                              }),
               pages_.end());
}

} // namespace nntrainer::kv

// SPDX-License-Identifier: Apache-2.0
/** @file kv_page_pool.h
 * @brief Thread-confined host page pool shared by synchronous KV sessions.
 * @author Codex (OpenAI)
 */
#ifndef NNTRAINER_KV_PAGE_POOL_H
#define NNTRAINER_KV_PAGE_POOL_H

#include <cstddef>
#include <kv_cache/kv_storage.h>
#include <memory>
#include <vector>

namespace nntrainer::kv {

/** The pool holds one reference to each allocation. A page is reusable only
 * when no session or exported view retains it. No device completion is implied.
 * The optional budget includes idle pages and 64-byte allocation rounding.
 */
class KvPagePool {
public:
  explicit KvPagePool(
    size_t budget_bytes = 0,
    std::shared_ptr<MemAllocator> allocator = std::make_shared<MemAllocator>());
  std::shared_ptr<KvStorage> acquire(size_t bytes);
  size_t reservedBytes() const { return reserved_bytes_; }
  /** Free unused pages, for example before admitting a different geometry. */
  void trim();

private:
  std::shared_ptr<MemAllocator> allocator_;
  std::vector<std::shared_ptr<KvStorage>> pages_;
  size_t budget_bytes_;
  size_t reserved_bytes_ = 0;
};

} // namespace nntrainer::kv
#endif

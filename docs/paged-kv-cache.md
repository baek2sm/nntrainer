# Paged CPU KV cache and attention

The internal `nntrainer::kv::KvCacheManager` supports a paged CPU backend.
Set `CacheSpec::page_tokens` to a positive token count (for example 16).
Zero keeps the existing contiguous backend. Each owner layer and batch has
its own logical block table; layer aliases resolve to the owner's table.
Only blocks needed for an admitted step are allocated. Physical blocks hold
token-major KV heads, without expanding grouped query heads.

`KvPagePool` can be shared between managers, allowing independent sessions to
reuse released physical storage. Its optional byte budget includes 64-byte
allocation rounding and idle blocks. `trim()` frees idle blocks, including
blocks with sizes no longer needed. Pool and session operations are synchronous
and thread confined; they must all run on the same thread or be externally
serialized.

```cpp
using namespace nntrainer::kv;
auto pool = std::make_shared<KvPagePool>(256 * 1024 * 1024);
CacheSpec spec;
LayerSpec layer;
layer.num_kv_heads = 2;
layer.head_dim = 64;
layer.capacity = layer.max_tokens = 4096;
layer.encoding = Encoding::Float32;
layer.storage_type = StorageType::Float32;
spec.layers = {layer};
spec.page_tokens = 16;
KvCacheManager cache;
cache.initialize(spec, std::make_shared<nntrainer::MemAllocator>(), pool);

// k/v: [new_tokens, kv_heads, head_dim], in the cache encoding.
// q: FP32 [new_tokens, query_heads, head_dim], with RoPE already applied.
auto step = cache.beginStep(new_tokens);
cache.append(step, 0, k.data(), v.data(), k.size() * sizeof(float));
cache.markReady(step, 0); // After producing all batches of this owner layer.
AttentionOptions options;
options.query_heads = 8;
auto output = cache.attention(step, 0, q, options);
cache.commit(std::move(step));
```

`attention()` reads the checked KV segments directly. Its online softmax keeps
one head-sized accumulator and does not gather KV history or allocate the full
attention score matrix. It supports causal prefill, decode, GQA/MQA/MHA,
noncausal attention, a trailing window and optional logit softcapping. Queries
and results are FP32; cache data can be FP32 or IEEE binary16 (including UINT16
backing without native half support). Scale defaults to `1/sqrt(head_dim)`.
The window ends at each query's causal endpoint, or at the step endpoint for
noncausal attention. This scalar CPU implementation prioritizes portability
and correctness; it does not claim the throughput of device attention kernels.

Producers can also write through `writeKey()` / `writeValue()` segments instead
of `append()`. A page-crossing write returns multiple segments in logical order.
Call `markReady()` before reading or attending, and commit only after every
owner producer is ready. Layer aliases cannot produce their own KV.

Failed step admission leaves the session unchanged; successfully allocated
blocks may remain idle in the pool. Aborting a step returns newly allocated
whole blocks beyond the committed position. Reset returns all session blocks;
backward `setPosition()` returns whole blocks beyond the new position. A
partial tail remains private to that session. Forward seek is rejected because
it would publish unwritten KV. There is no cross-session prefix sharing.

Every exported view pins its physical block. Invalidating a view prevents API
access but does not recycle its allocation until the view itself is released.
A raw pointer must not escape its synchronous operation. `CacheStats` reports
session-owned bytes; `KvPagePool::reservedBytes()` also includes idle and
view-pinned blocks. A bounded pool can reject a step while stale views pin
blocks; release the views before retrying.

The existing CausalLM Tensor-placeholder/MHA graph adapter remains contiguous:
`keyMemory()`, `valueMemory()` and `bindStep()` reject paged storage. Paged callers
use `append()` and `attention()` directly rather than passing page storage to a
dense Tensor consumer. Ring caches, device execution, attention sinks, automatic
prefix caching and asynchronous completion are outside this CPU backend.

Numerical tests in `unittest_kv_cache_core.cpp` compare paged attention against
an independent dense softmax reference across page-crossing prefill and decode,
multiple batches, GQA, aliases, windows and softcapping. Additional cases cover
binary16 decoding, pinned pages, shared pool admission, reuse and allocation
failure. These tests were added without building or running them on macOS.

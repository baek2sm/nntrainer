# Sparse KV prefix reuse

For a concise review of the implementation, measured benefits and portable
Quick.KV experiment reproduction, start with
[Sparse KV analysis and reproduction](sparse-kv-quality-todo.md).

Branch: `kv/sparse-prefix-reuse-20261110`, based on
`9034300447b30915963b14b32d74d4e0da78c77a` (TurboQuant experiments).

This ports Quick.KV's ordered-subset gathering and RoPE correction into the
NNTrainer KV managers. The source reference is the local Quick.KV checkout at
`d8f8626197c4734e67d8fbac959ac572302ed1fa`, particularly
`quickkv/matching/sparse.py`, `quickkv/transfer/rope.py`, and `docs/sparse.md`.
The implementation is C++ and uses NNTrainer storage and backend interfaces.

For a source token sequence `A B C D` and a request `A C D E`, reuse `A C D`
and compute `E`. Each symbol can represent one token or a contiguous run.
The destination is contiguous; source positions can have gaps. Keys move by
`R(destination - source)` and values are copied without modification.

**This is approximate reuse.** A retained token's KV still contains information
from deleted earlier tokens. RoPE fixes positions; it cannot undo that attention
history. Outputs can differ from recomputing the subset. The feature is off by
default and tracks approximate history separately from exact cache entries.

## Matching and storage

* The core manager traverses compressed radix edges allowing source token
  deletions. Every result comes from one root-to-terminal history; sibling
  branches are never mixed. The HTP manager ranks deletion matches from its
  bounded collection of published histories alongside its exact radix hits.
* Token IDs must match, in order. Greedy earliest occurrences maximize the
  reusable request prefix. Duplicate tokens are supported. Unlike Quick.KV's
  segment-based template matching, this implementation permits individual
  token deletions. It stops at the first request token absent from the remaining
  source; the rest of the request is evaluated normally.
* Runs merge when source and destination are both contiguous. They may cross
  pages and finish inside a page. Matching never substitutes text or assumes
  that tokenization is unchanged after editing text.
* HTP keeps the existing exact lookup. Its automatic policy requires at least
  64 additional reusable tokens over the best exact hit before gathering a
  sparse prefix. Full exact hits and requests too short to gain 64 tokens
  bypass sparse search. This prevents a one-token gain from triggering a full
  prefix copy and rotation. `DeviceKVManager::setMinimumSparseGain()` can change
  this policy explicitly; the direct CPU core API remains caller-controlled.
  Sparse gathering allocates destination pages, leaves source pages immutable,
  and corrects K using each layer's head geometry and model frequencies.
* HTP builds sorted unique token IDs for retained histories only when sparse
  lookup is enabled. An absent first token rejects a source before querying
  backing boundaries; an absent token after a common header stops the matcher
  without scanning the remaining source. This index only rejects impossible
  matches; IDs that are present still use the ordered match and backed-row
  checks. Exact-only publication does not build this index.
* Approximate sessions and continuations are not published into the exact
  cache. Forking and session switching preserve this provenance. The CPU core
  can truncate to an exact leading region and then resume exact publication.
* The host core supports FP32, FP16 and packed TurboQuant keys. Moved packed
  keys are decoded, transformed back from the Hadamard basis, position-corrected
  and encoded again. Packed V bytes are copied directly. Re-encoding K adds
  quantization error. Aliased layers use the owner's pages and RoPE contract.
* Flash-backed core pages use the existing checked loader; unavailable source
  pages shorten a core hit. HTP restores host snapshots from DRAM, Flash or
  baked frames. A source lacking necessary sliding rows becomes a cache miss.
  It never fills a missing row with zeros or an unrelated branch's KV.

## HTP configuration

Add this to the model's existing original Q8 prefix configuration:

```json
{
  "skip_prefill": true,
  "htp": {
    "kv_session": true,
    "kv_paged": true,
    "kv_prefix_cache": true,
    "kv_sparse_prefix_cache": true,
    "kv_seed_q": false,
    "kv_turboquant": false,
    "attn_q2": true,
    "oproj_u16": true
  }
}
```

The existing Gemma4 requirements still apply: batch one, Q8 attention,
64-row-aligned prefill plane and an enabled device KV manager. The application
installs the sparse restorer when `enableDeviceKVManager()` runs. Existing
runtime validation rejects unsupported combinations. Device TurboQuant remains
incompatible with device KV sessions; this change does not enable combining
those modes. Packed CPU sparse reuse is supported independently.

Gemma4's fused QKV producer applies RoPE before attention, whose `use_rope` flag
is consequently false. Sparse restoration explicitly requests the upstream
producer frequencies, including the partial global RoPE. Treating that flag as
meaning unrotated cache keys is incorrect. The device check verifies this for
all 30 attention layers (D256 sliding and D512 global).

After gathering, original Q8 calibration is discarded. The first suffix step
calibrates the restored rows and new rows again. This includes invalidating
stale process-local entries loaded from a calibration file. Existing sliding
compaction and the Q8-to-decode handoff continue from the new positions.

The host core API is explicit:

```cpp
// One model-derived SparseRope per layer. count excludes the final prompt
// token when that token must still be evaluated to produce logits.
auto result = manager.reuseSparsePrefix(ids.data(), count, layer_rope);
// result.tokens is the reusable prefix length; result.approximate() reports
// whether a deletion changed its causal history.
```

## Validation

The checked-in device report is
[validation/sparse-prefix-s26u-20261110.json](validation/sparse-prefix-s26u-20261110.json).
Host checks cover 76 KV core cases, including seven sparse cases, 100 randomized
comparisons against an independent dynamic-programming subsequence oracle,
FP16/FP32/packed keys, aliases, partial pages, source immutability, allocation
failure, Flash corruption, provenance and session forks. The core and snapshot
checks also pass ASan/UBSan (LeakSanitizer disabled in the ptrace environment).
Device-manager header tests pass ASan/UBSan. The real Android build passes.

S26U SM-S948U, Gemma4 26B A4B, NPU attention and MoE, Q8, prefill plane 512,
maximum context 2048, deterministic generation budget 24:

| Request | Reused KV rows | Fresh prefill rows | Warm recompute TTFT | Sparse TTFT |
| --- | ---: | ---: | ---: | ---: |
| Delete 174 tokens from a 936-token source | 761 | 0 | 2968 ms | 514 ms |
| Same deletions plus a changed question suffix | 739 | 24 | 3015 ms | 1856 ms |

The first request still evaluates its final prompt token through the single-row
generation path; zero fresh *prefill* rows does not mean logits were cached.
Numbers above are one sequential device comparison, including lookup, gathering,
RoPE and cache handoff in TTFT. They are not repeated, thermally controlled
throughput benchmarks. Retaining source snapshots costs memory: this is a
prefill-reuse feature, not a claim of lower total memory than a cold request.

Both subset requests retained the requested fact `ORCHID-842`. The subset-only
request's generated tokens differed from cold; the changed-question request's
tokens matched cold exactly in this fixture. Both cold and sparse can emit
repeated thought/turn markers with this model configuration, so fact retention
is not evidence of generally clean formatting or broad model quality. Original
source replay was an exact hit and reproduced all original generated IDs. No
approximate prefix was added to the exact index.

Reproduce Android model validation with the optional device-check build target:

```sh
htp_sparse_prefix_check MODEL_DIR OUTPUT_JSON
```

It creates a model-local test configuration in memory, keeps the last prompt
token for fresh inference, checks the actual producer RoPE geometry, exercises
both full subset reuse and a newly evaluated suffix, checks fact retention,
and verifies exact source replay and absence of cache contamination.

Existing-path performance comparisons and the small-gain regression fix are
recorded in [sparse-kv-prefix-regression.md](sparse-kv-prefix-regression.md).

A repeated Q8-only cold/exact/sparse benefit matrix at 4K–24K on S26U is
recorded in [sparse-kv-prefix-benefit.md](sparse-kv-prefix-benefit.md), with raw
cohorts and CSV. TurboQuant is disabled in all three variants.

Absent-token miss optimizations and expanded device measurements are recorded
in [sparse-kv-prefix-miss.md](sparse-kv-prefix-miss.md).

Quality improvements, Quick.KV experiment evidence and reproduction commands
are tracked in [sparse-kv-quality-todo.md](sparse-kv-quality-todo.md).

The 4096-row prefill comparison against `5e97254` and the owned sparse-source
snapshot correction are recorded in
[sparse-kv-prefix-4096-validation.md](sparse-kv-prefix-4096-validation.md).

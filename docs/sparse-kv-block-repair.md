# Sparse KV assembly and block repair

The existing position-only implementation is described in
[sparse-kv-prefix-cache.md](sparse-kv-prefix-cache.md). It gathers retained
superset rows, applies `R(destination - source)` to K, and copies V unchanged.
Those CPU-core and HTP restoration paths already existed before this change.

This change adds an explicit CausalLM source API and a synchronous reference
executor for contiguous block repair. It stores one immutable superset KV
source, not one cache or delta per subset combination.

## Execution

With repair disabled, assembly ends after gather and K position correction.
The final prompt token remains fresh so inference can produce logits.

With repair enabled:

1. Gather the source into private working KV storage and align K positions.
2. Recompute the **entire first decoder block**, including attention and FFN.
3. Compute the second decoder block's input norm and K projection / K norm for
   every retained token. The probe follows only K's graph dependencies; it does
   not compute the second block's full-prompt Q/V, attention or FFN.
4. Apply the producer's actual RoPE to the fresh probe K and compare it with the
   loaded, already position-aligned K. Compare in the cache storage precision.
5. Sum squared K differences over channels, heads and tokens in each fixed
   contiguous block. Rank blocks by this sum. Leading exact-prefix rows do not
   contribute to the score. Ties use the earlier block.
6. Select whole blocks without exceeding `floor(prefix_tokens * ratio)` rows.
   The final short block is indivisible too. This can undershoot the budget;
   it never rounds up to exceed 15%. A block straddling the exact prefix counts
   all its rows against the budget.
7. Replay only selected blocks from the second decoder block onward. Preserve
   each block's absolute subset positions, causal mask and access to the full
   retained KV history. Adjacent selected blocks are merged up to the activation
   plane size; disjoint spans need separate calls. Unselected later-layer KV
   rows remain byte-identical to the assembled baseline.

The first block's boundary activations are temporary per-request staging. They
are released when repair returns and are not added to the persistent cache or
stored by combination. Second-layer projections are recomputed for selected
blocks when replay starts. This small duplicated projection work is included
in runtime; the implementation does not claim it is free.

The selection layer is **index 1, the actual second decoder block**. The first
block's K/V are projected from token-local embeddings before its attention, so
their mathematical values do not encode a changed preceding context. Computing
the first block is what makes that context change observable in the second
block's K. LMCache's documented example likewise sets
`LMCACHE_BLEND_CHECK_LAYERS=1` (zero based):
<https://docs.lmcache.ai/kv_cache_optimizations/blending.html>.
This implementation selects blocks once; it does not reproduce the paper's
gradual filtering across multiple layers.

## Configuration and public API

In `nntr_config.json`, repair is off by default:

```json
{
  "sparse_kv_recompute": {
    "enabled": true,
    "block_tokens": 32,
    "ratio": 0.15
  }
}
```

For the synchronous CPU path, capture one freshly computed superset and arm it
for subsequent independent requests. Token IDs must correspond exactly to the
captured KV rows. Never independently tokenize fragments and assume their
concatenation matches tokenizing the complete prompt.

```cpp
model.run(superset_text, false, "", "", false);
auto ids = model.getTokenizer()->Encode(superset_text, true);
auto source = model.captureSparseKVPrefix(
  std::vector<uint32_t>(ids.begin(), ids.end()));

model.setSessionTokenPosition(0); // independent request
model.setSparseKVPrefix(source);
model.setSparseKVRecompute({false, 32, 0.15}); // RoPE-only
model.run(subset_text, false, "", "", false);

model.setSessionTokenPosition(0);
model.setSparseKVRecompute({true, 32, 0.15});  // block repair
model.run(subset_text, false, "", "", false);
const auto &stats = model.getSparseKVRecomputeStats();

model.setSparseKVPrefix(nullptr); // disarm explicit source
```

The source object belongs to its capturing model instance and owns a copied
snapshot; resetting the live session
does not overwrite it. Capturing an approximate or repaired live history as a
fresh source is rejected. The explicit source API cannot be combined with the
device manager's existing automatic source lookup. The device manager remains
responsible for its own approximate-history provenance.

`getSparseKVRecomputeStats()` reports retained rows, the integer token budget,
selected rows, block scores / positions, first-block and K-probe row counts,
replay call count and separate probe/replay times. Those times exclude gather
and source loading; they are not end-to-end TTFT measurements.

## Execution scope and costs

The new repair executor currently supports the **synchronous CPU engine**,
batch one, externally supplied FP16/FP32 KV, standard numbered decoder blocks,
and complete retained history in every layer. The new explicit CausalLM source
API also requires `repetition_penalty=1`. Packed attention-owned history, FSU,
GPU execution and resident HTP execution are rejected for this new executor.
Existing HTP RoPE-only sparse reuse continues through its existing manager.

A compacted sliding snapshot only retains the final window. Such a snapshot
cannot support arbitrary early-block queries, even if later global layers
retain all rows. Repair therefore rejects missing history before executing the
graph. Supporting it requires loading each selected block's causal sliding
window, or retaining that history; silently using the final window is wrong.
Resident NPU repair needs its own materialization and quantized-mirror handoff.
No NPU quality, latency or load/compute-overlap claim is made by this reference.

The 15% cap applies to token positions replayed in later decoder blocks. It
does **not** include the mandatory full first block, full-prompt second-layer
K probe, small repeated selected projections, or the fresh suffix. Model-wide
work must be reported separately from this token selection ratio. The reference
performs assembly before repair and does not overlap storage loads with compute.

Persistent storage is one superset asset plus small configuration. Request
memory includes working KV, first-block boundary staging proportional to prompt
length and hidden width, and small score/index arrays. There is no learned
estimator, calibration dataset, scale, or stored subset-specific KV delta.

## Reproduction

Build with the existing CPU transformer/test configuration, then:

```sh
ninja -C BUILD Applications/CausalLM/nntr_quantize \
  Applications/CausalLM/unittest_causallm_models \
  Applications/CausalLM/unittest_kv_cache_manager \
  test/unittest/unittest_kv_cache_core \
  Applications/CausalLM/sparse_block_repair_check

OPENBLAS_NUM_THREADS=1 meson test -C BUILD --no-rebuild \
  unittest_kv_cache_core unittest_kv_cache_manager unittest_causallm_models \
  --print-errorlogs

NNTR_ENGINE=cpu OPENBLAS_NUM_THREADS=4 \
  BUILD/Applications/CausalLM/sparse_block_repair_check \
  /path/to/nntrainer-qwen3-model smoke.json
```

Tests check budget/tail/invalid input handling, V/source preservation,
unselected later-layer byte identity, actual selected-row changes, and recovery
of the cold subset KV with a full repair budget. The executable exercises the
public run/capture/arm APIs on a real model. Its short synthetic fixture is an
integration smoke check, not evidence of general quality improvement.


### Recorded checks (2026-11-16)

- CPU KV core: 76 tests, zero failures.
- Application KV manager: 37 tests, zero failures.
- Model suite: 135 cases, 134 passed, one existing QINT4 case skipped, zero failures.
- New sparse-specific checks: eight, all passed, including a forced seven-row
  probe/replay schedule and cross-model source rejection.
- The public API smoke exercised a 476-token superset, 395 reused subset rows,
  a 59-row cap and 43 actual selected rows (one full 32-row block plus the
  11-row tail), through two replay calls. First-block and K-probe counts were
  both 395. The earlier 220-token / 139-row smoke retained an 11-row tail under
  its 20-row cap. Neither execution rounded a whole block above the budget.
- Both local 0.6B checkpoints produced repeated tokens in the cold control as
  well as both reuse modes. These are plumbing checks, **not generation-quality
  evidence**. No generation-quality improvement or speedup is claimed.

[Raw outputs, timings, test counts and source hashes](validation/sparse-block-repair-cpu-20261116.json)
are preserved for reproduction. The earlier short Q4_0 run is retained alongside
its replacement FP32 integration case rather than silently discarded.

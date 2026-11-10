# Sparse prefix reuse benefit on Galaxy S26 Ultra

Measured on 2026-11-10. All variants use Q8 KV, with **TurboQuant off**.
The primary comparison is full request-KV recomputation versus existing exact
prefix caching versus sparse prefix caching. Fan cooling is enabled for every
sample in the final table. Two samples per cell; latency is external TTFT.

| Request | Exact reused | Sparse reused | Cold (s) | Exact (s) | Sparse (s) | Sparse vs cold | Sparse vs exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 24K unchanged (exact control) | 24576 | 24576 | 90.44 | 2.49 | 2.32 | 38.96x | 1.07x |
| 4K retained; delete 20K near 0.5K | 0 | 4096 | 14.37 | 12.80 | 2.94 | 4.88x | 4.35x |
| 8K retained; delete 16K near 4K | 4096 | 8192 | 27.80 | 13.94 | 3.20 | 8.68x | 4.35x |
| 20K retained; delete 4K near 4K | 4096 | 20480 | 73.45 | 56.37 | 4.31 | 17.04x | 13.08x |
| 20K retained; delete 4K near 12K | 12288 | 20480 | 74.12 | 31.32 | 3.84 | 19.31x | 8.16x |
| 20K retained; four 1K holes | 4096 | 20480 | 74.57 | 56.92 | 4.26 | 17.51x | 13.37x |
| 24K minus one token near 4K | 4096 | 24575 | 90.78 | 73.33 | 4.66 | 19.48x | 15.74x |
| 4K complete cache miss | 0 | 0 | 14.19 | 13.39 | 13.66 | 1.04x | 0.98x |

For the requested 20K/4K case, exact reuses 4096 rows and computes 16401 new
prefill rows. Sparse reuses all 20480 context rows and computes only 17 new
question rows, plus the final prompt token needed for logits. Observed TTFT
falls from 73.45s cold / 56.37s exact to 4.31s sparse.

An unchanged input remains an exact hit in both cache modes (zero sparse
gathers); its timing difference does not demonstrate an additional sparse gain.
The complete-miss control has no reuse and is about 2% slower with sparse
enabled than exact in these two samples. This experiment does not establish
zero overhead for cache misses or a universal speedup.

## Observed ranges

Minimum–maximum external TTFT in seconds; two samples do not establish a
confidence interval.

| Request | Cold range | Exact range | Sparse range |
| --- | ---: | ---: | ---: |
| 24K unchanged (exact control) | 89.755–91.128 | 2.405–2.580 | 2.316–2.326 |
| 4K retained; delete 20K near 0.5K | 14.286–14.458 | 12.798–12.799 | 2.936–2.949 |
| 8K retained; delete 16K near 4K | 27.774–27.827 | 13.905–13.975 | 3.157–3.245 |
| 20K retained; delete 4K near 4K | 73.127–73.771 | 55.637–57.097 | 4.307–4.313 |
| 20K retained; delete 4K near 12K | 74.116–74.132 | 30.940–31.695 | 3.821–3.858 |
| 20K retained; four 1K holes | 74.334–74.813 | 56.898–56.938 | 4.202–4.314 |
| 24K minus one token near 4K | 90.371–91.184 | 73.211–73.439 | 4.579–4.739 |
| 4K complete cache miss | 14.111–14.272 | 13.229–13.549 | 13.653–13.665 |

## Setup and output checks

The 4K/12K/24K producer requests take 81.76s total wall time in this run.
Model load/setup takes 1.44s in the cached process and
1.42s in the cold process. These are recorded separately
and excluded from per-request TTFT. Producers are built incrementally using
available exact checkpoints; this is an observed preparation cost, not the
incremental overhead of capturing KV versus ordinary inference.

Every cached sample retains **2880 MiB (2.8125 GiB)** of source prefix-host
accounted backing, including retained shared slabs; every cold sample retains
**0 bytes**. This excludes model weights,
live destination KV and temporary gather/restore buffers. Prefix reuse trades
source memory/preparation for repeated-request latency; this is not a claim of
lower total memory than cold inference.

All 16 exact samples match the corresponding cold generated token IDs. All
cold repetitions are stable. All 7 positive fact-retrieval cases keep
`ORCHID-842` with sparse. Sparse IDs differ from cold in `4k_at_0k`,
`8k_at_4k` and `20k_multi_holes`; the other five cases match, including the
negative complete-miss control. The negative control has removed the secret
from its input, so absence of the code there is expected. The model can emit
repeated thought/turn markers in all modes; these checks do not establish
general output quality.

Host validation: 76 KV-core tests and 37 CausalLM KV tests pass. Core/snapshot
and device-manager checks pass ASan/UBSan (LeakSanitizer disabled under ptrace).
The Android build and device snapshot checks pass. Changed C/C++ uses
clang-format 14. No cross-platform CI matrix is claimed by this local run.

## Measurement

- Device: SM-S948U (Galaxy S26 Ultra), serial R3CY70LV96T, DSP v81, 12 GB RAM.
- Model: local `longgen-q8`, TL2 rotated weights, 30 attention layers: 25 sliding
  (head dim 256, 8 KV heads, window 1024), 5 global (head dim 512, 2 KV heads,
  partial RoPE 0.25). Attention/GQA and MoE execute on the NPU/HTP path.
- Q8 describes device attention KV. Host prefixes retain the existing native
  prefill snapshot representation and can pin whole shared-memory slabs.
- Prefill plane: 1024; max sequence capacity: 28672; greedy generation budget:
  16 tokens; `kv_seed_q=false`, `kv_request_reserve=true`, `page_cache=0`,
  `moe.cache_grow=0`; 8 host threads. K in case names means 1024 tokens.
- Primary metric: **external TTFT**, measured with a steady clock from calling
  `run` until its first nonempty streamer callback. This includes tokenization,
  lookup, gather/re-RoPE, restore, newly computed suffix and first visible token.
  Internal prefill-only time omits important reuse overhead and is not the
  speedup denominator.
- **Cold**: model weights/workspaces warmed, request KV fully recomputed. All
  four cache/session flags disabled; no source prefix memory retained. Every
  measured cold sample checks zero reused rows and zero prefix-host bytes.
  Model loading/disk-cold inference is outside this comparison.
- **Exact**: existing exact-prefix path with sparse callback removed.
- **Sparse**: same prefix sources, sparse enabled, minimum gain 64 tokens.
  Unchanged exact hits continue using the exact path.
- Separate cold and cached processes. Cached cohort prepares native 4K/12K/24K
  checkpoints, then alternates exact/sparse order per repetition. The final cold
  cohort follows the cached cohort. An external fan is enabled throughout both.
  Two repeats per case/mode: 8 cases, 48 final measured requests. The original
  cold cohort was started before the fan was enabled and is excluded from the
  final table. No CPU/NPU clock or thermal lock, so ratios are observed device
  results, not universal guarantees.
- Source production/model initialization excluded from request TTFT. Producers
  and warmup are retained in the raw reports so preparation cost is visible.
  Requests never publish new prefixes during measurement. No Flash prefix hits.
- Context begins with `The secret code is ORCHID-842.` followed by distinct
  numbered records. Each request appends the same 18-token question. The last
  input token is still freshly evaluated for logits; a full *context* hit leaves
  17 fresh prefill rows plus that last token. This is a synthetic deletion
  workload, not a general language-quality benchmark.
- Deletions operate on token IDs. Decode/encode roundtrip equality is checked.
  Some deletion boundaries shift by up to 63 tokens for stable tokenization;
  actual positions are in `deleted_source_ranges` in the raw reports. Exact
  hit lengths are checked against the native checkpoints, rather than inferred
  from where the text was edited.

## Implementation changes used for this experiment

- Match only source tokens whose KV rows have actually been published. The
  final one-row inference token can have a known ID without host prefill KV.
- Load the ending source checkpoint first. If it covers every required global
  and retained sliding row, avoid loading earlier checkpoints. Otherwise keep
  the existing multi-checkpoint fallback; missing rows remain a miss.
- Compute RoPE sine/cosine once per layer and contiguous gathered run. All
  tokens/heads in that run share the same source-to-destination position delta.
  This preserves the rotation and skips sliding rows no longer needed.

The 4096-row prefill-plane pilot failed in the device fused QKV operation
(`0x80000583`). Its data are excluded. Every result here uses the stable 1024-row
plane equally across all three modes.

## Scope

Sparse reuse is approximate: positional correction cannot remove attention
history from deleted tokens. Generated IDs and secret-code retention are
reported for each mode; fact retention in this fixture does not establish
broad model quality. Cache hits require source KV already prepared and retained;
source memory is an explicit cost, not a memory-saving claim versus cold.
A cache miss still computes the full request. The unchanged-input and miss
controls bound what these samples say about overhead.

## Reproduction

Build the optional `htp_sparse_benefit` Android target in
`Applications/CausalLM/app_build/meson.build`, deploy it with the matching
NNTrainer shared libraries, and run sequentially on the same device:

```sh
export LD_LIBRARY_PATH="$PWD"
export ADSP_LIBRARY_PATH="$PWD;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp"
export NNTR_NUM_THREADS=8
export NNTR_HTP_KV_MANAGER=1
./htp_prefill_check
./htp_sparse_benefit MODEL_DIR benefit-cold.json 2 all cold
./htp_prefill_check
./htp_sparse_benefit MODEL_DIR benefit-cache.json 2 all cache
```

Copy the two JSON reports to the host, then:

```sh
python3 tools/kv_cache/summarize_sparse_benefit.py \
  benefit-cold.json benefit-cache.json benefit-summary
```

The analyzer checks matching model/case settings, balanced repeats, source-row
counts, absent Flash hits, zero retained cold prefixes and exact/cold output
IDs. It exports a CSV, summary JSON and Markdown table. The C++ target also
supports one-case filtering and `all` mode (cold/exact/sparse sharing retained
source memory); `all` is a different baseline and is not used for this table.


## Raw results

[Raw cohorts, metadata and summary](validation/sparse-prefix-benefit-s26u-20261110.json)
and [CSV table](validation/sparse-prefix-benefit-s26u-20261110.csv).
The earlier pre-fan/mixed-cooling cold cohort and failed 4096-plane pilot are
excluded. The checked-in raw report contains only the 48 final samples.

To re-run the analyzer from the combined artifact:

```sh
python3 - <<'PYRAW'
import json, pathlib
j = json.loads(pathlib.Path('docs/validation/sparse-prefix-benefit-s26u-20261110.json').read_text())
for mode in ('cold', 'cache'):
    pathlib.Path('/tmp/benefit-' + mode + '.json').write_text(json.dumps(j[mode]))
PYRAW
python3 tools/kv_cache/summarize_sparse_benefit.py \
  /tmp/benefit-cold.json /tmp/benefit-cache.json /tmp/benefit-summary
```

A subsequent absent-token lookup optimization and expanded miss-only comparison
are recorded in [sparse-kv-prefix-miss.md](sparse-kv-prefix-miss.md). The table
above preserves the original measurements.

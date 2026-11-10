# Sparse prefix miss overhead follow-up

The earlier two-sample benefit matrix showed 13.39s exact versus 13.66s sparse
for a 4K complete miss. We separated lookup cost from model execution and
removed the source-length-dependent scan for absent token IDs. This is a
Q8-only comparison; TurboQuant stays off.

## Implementation

Each immutable published source builds sorted unique token IDs only when
sparse lookup is enabled. A request whose first ID is absent skips that source
before allocating runs or querying backend boundaries. At later token
mismatches, the ordered matcher checks membership before scanning the remaining
source. Absent IDs stop the plan immediately. Present IDs still use the original
ordered scan: membership alone never authorizes a hit. Extra IDs corresponding
to unbacked final rows are conservative; the backed-row limit still applies.

Enabling sparse after publication builds missing indexes; toggling it off and
on reuses indexes. LRU eviction/clear removes them with their source. Ordinary
exact-only publication does not build the index. The memory cost is one 32-bit
entry per distinct source ID, plus vector metadata, per retained history.

## S26U lookup measurements

Eight 24K source histories; 200000 lookups per cell, microseconds per lookup.
The before binary uses the parent manager header; both binaries use the same
benchmark/compiler settings. No NPU inference or KV copying in this table.

| Lookup | Before exact-only | Before sparse enabled | After exact-only | After sparse enabled |
| --- | ---: | ---: | ---: | ---: |
| Exact full hit | 19.786540 | 19.032052 | 20.008206 | 20.025568 |
| Exact partial hit / sparse miss | 0.093653 | 48.480095 | 0.100133 | 1.073405 |
| Complete miss | 0.004061 | 47.507218 | 0.004062 | 0.069308 |
| Miss after 128 common tokens | 0.004161 | 47.771536 | 0.004170 | 0.329617 |

Complete-miss extra lookup cost versus exact-only is
0.065246µs;
miss after a shared header adds
0.325448µs.
The earlier 270ms full-request difference cannot be attributed to an original
lookup taking about 48µs. These are separate measurements and workloads.

## Full model miss measurements

Gemma4 26B A4B, NPU attention/MoE, Q8, 1024-row prefill plane, 4K input context
plus an 18-token question, 16-token generation budget. Same three DRAM sources
at 4K/12K/24K (2880 MiB retained backing) for both modes. First six repeats per
mode, then ten additional repeats because variability exceeded the observed
difference; **all 16 repeats per mode are included**. Each cohort alternates
exact/sparse then sparse/exact order. Same deployed binary in both cohorts.
No cold-baseline rerun is needed to compare these two miss paths. Model and
producer preparation are excluded; lookup, restore and all inference until the
first nonempty streamer callback are included in external TTFT.

| Mode | Samples | Median TTFT | Mean TTFT | Range |
| --- | ---: | ---: | ---: | ---: |
| exact | 16 | 12.6010s | 12.5061s | 12.1283–12.8582s |
| sparse | 16 | 12.6158s | 12.5139s | 11.9371–12.8593s |

Sparse median difference: +0.118%; mean difference:
+0.063%. Paired mean difference is +7.82ms;
an exploratory paired t interval is [-56.98, +72.62]ms at 95%.
Clocks/thermals are not locked, and rounds within a cohort can be correlated;
the interval is descriptive, not proof of universal zero overhead. The code
removes linear absent-ID scanning; full-device timing must still be interpreted
with its observed variation.

All 32 requests reuse zero KV, compute 4113 prefill rows plus the final prompt
token and produce identical generated IDs. No Flash or sparse restore is called.
Host tests: 76 core and 37 CausalLM KV tests; ASan/UBSan core, snapshots and
manager checks; Android build and device snapshot checks all pass. Indexed and
unindexed run plans agree in 1600 randomized comparisons, alongside existing
duplicates, positive deletion hits, unbacked-tail and publication tests.

## Reproduction

Build `benchmark_sparse_lookup.cpp` with `-O3 -std=c++17 -DSPARSE_LOOKUP_API=1`
and the relevant manager header; run `on 24576` and `off 24576`. Build the
optional Android `htp_sparse_benefit` target and use the same runtime environment
as [the benefit matrix](sparse-kv-prefix-benefit.md):

```sh
./htp_prefill_check
./htp_sparse_benefit MODEL_DIR miss.json 6 4k_cache_miss cache 1024
./htp_sparse_benefit MODEL_DIR miss-10.json 10 4k_cache_miss cache 1024
```

[Raw lookup and full-model samples](validation/sparse-prefix-miss-s26u-20261110.json).
The prior benefit matrix remains unchanged as a historical measurement.
Current sparse-enabled publication uses owned source pages; exact replay of
the historical retained-source memory requires commit `9bf6f1fad` and its
original 1024-row driver (without the new final row-count argument).

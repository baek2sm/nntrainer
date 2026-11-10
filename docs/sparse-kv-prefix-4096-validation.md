# Sparse prefix validation with 4096 row prefill

The performance reference is `5e97254e764aa30db4223c33d0dc3b0fded091b8`:
20,480 common-prefix tokens took 46.829 s to recompute, 2.752 s for a DRAM hit
and 5.325 s for a Flash hit. Those are application TTFT values, not isolated
prefill kernel timings. The original report used the fastest of four
recompute samples and the mean of two hit samples on SM-S948N.

The earlier [sparse benefit matrix](sparse-kv-prefix-benefit.md) used a
1024-row prefill plane on SM-S948U. Its roughly 73 s 20K recompute time and
speedup ratios must not be substituted for the original 4096-row reference.

## Implementation correction

The sparse benchmark now defaults to 4096 prefill rows and accepts an explicit
row count for diagnostic comparisons. It records that setting in its config;
the 1024-row historical experiment remains available by passing `1024`.

An earlier 4096 pilot returned `0x80000583`, which the SDK defines as
`AEE_EBUSY`. A fresh run with the current deployed libraries completed 24K
prefill in 51.052 s internally, with external TTFT 51.697 s. This does not
establish the cause of the earlier busy response. That pilot also lacked the
explicit `kv_request_reserve=true`, `page_cache=0` and `moe.cache_grow=0`
settings used by the final benchmark, so it was not a controlled test of
prefill row count alone.

The subsequent three-history 4096 stress run ended during the 24K producer's
decode transition, and the device rebooted. Its last reported available memory
was 803.7 MiB. The post-reboot boot reason was `reboot`; kernel crash evidence
was unavailable. The interrupted JSON contained only NUL bytes and is excluded
from timing comparisons.

Inspection found that borrowed DRAM snapshots kept entire shared KV slabs
alive, including unused global capacity and sliding prefill workspace. When
`htp.kv_sparse_prefix_cache` is enabled, publication now uses existing owned
snapshot pages, retaining only the captured rows. Owned pages from an earlier
checkpoint can still be shared. Exact-only publication keeps its original
borrowed-page path. Gather, RoPE, Q8 calibration and approximate provenance
remain as before; selective recomputation remains a research TODO.

The corrected 4096 run retains 280 / 640 / 1080 MiB after preparing the
4K / 12K / 24K sources. The earlier 1024 run retained 2880 MiB after all three.
These are accounted source-prefix bytes, excluding weights, active KV and
temporary buffers. Source preparation now includes copying the owned pages.

A later cold-only cohort lost its ADB connection after 12 completed requests.
Those requests retained zero source-prefix bytes and reused zero tokens;
reported process peaks were approximately 5492–5494 MiB. The disconnect
requires separate investigation and prevents declaring full device stability.

## Validation method

The external fan is broken for this follow-up (user confirmation). Current
runs have no fan, and neither clocks nor thermal state are locked. Compare
the freshly built baseline and current version on the same device; differences
from older cooled measurements alone are not evidence of a code regression.

Completed checks and limits:

- 4096-row exact/sparse cohort: **32/32 requests passed**, two per case/mode.
- Cold cohort: **12/16 requests completed**, covering six cases. Every completed
  cold output matches the corresponding exact-cache output. The cold report
  remains `passed=false`; the missing cases are `24k_one_token_at_4k` and
  `4k_cache_miss`. Neither is counted as a successful cold comparison.
- Rebuilt `5e97254` Android engine/CausalLM versus current, with the **same
  current DSP/provider**: 4K and 20K DRAM/Flash comparisons passed. All input
  IDs, output IDs and final positions match across versions. This isolates
  host implementation changes; it is not a full original-DSP comparison.
- Host: 76 KV-core and 37 CausalLM-manager tests passed. ASan/UBSan core and
  snapshot checks passed, including owned snapshot lifetime, exact payload
  size and data survival after producer destruction. LeakSanitizer is disabled
  in this ptrace environment. Android snapshot and baked-artifact checks passed.
- The user ended device testing after the second disconnect because the device
  was no longer available. Original-DSP/SDK 20K comparison and real-model
  4K/8K pre-baked generation/restart checks were prepared but **not executed**.
  No full long-run device stability claim is made. Changed C++ is formatted
  with clang-format 14.

### Same device host regression comparison

Seconds, baseline/current. Recompute uses four samples per length; DRAM and
Flash each use two. These are medians, unlike the historical table's fastest
recompute and mean hit aggregation. Recompute/hit order alternates. No material
host-path slowdown was observed in these sampled comparisons.

| Common prefix | Recompute baseline/current | DRAM baseline/current | Flash baseline/current |
| --- | ---: | ---: | ---: |
| 4,096 | 11.824 / 11.449 | 2.342 / 2.323 | 3.807 / 3.711 |
| 20,480 | 59.451 / 59.475 | 2.742 / 2.726 | 5.770 / 5.728 |

### Sparse reuse with 4096 row prefill

Seconds, median of two samples per available cell. Cold runs occurred later
than the alternating exact/sparse cohort, without fan cooling. Thermal drift
limits cross-cohort cold speedup claims; the exact/sparse pairs are the more
direct comparison. A dash denotes an uncompleted cold comparison.

| Request | Cold | Exact | Sparse | Sparse vs exact |
| --- | ---: | ---: | ---: | ---: |
| `24k_exact_control` | 76.207 | 2.302 | 2.286 | 1.01x |
| `4k_at_0k` | 11.563 | 8.670 | 2.906 | 2.98x |
| `8k_at_4k` | 22.563 | 10.141 | 3.202 | 3.17x |
| `20k_at_4k` | 61.275 | 38.926 | 4.265 | 9.13x |
| `20k_at_12k` | 61.680 | 22.951 | 4.174 | 5.50x |
| `20k_multi_holes` | 61.850 | 42.226 | 4.619 | 9.14x |
| `24k_one_token_at_4k` | — | 56.088 | 5.087 | 11.03x |
| `4k_cache_miss` | — | 10.549 | 10.474 | 1.01x |

The unchanged-input and complete-miss controls perform no sparse gather;
their small timing differences are not an additional sparse reuse gain.

All 14 positive sparse outputs retain `ORCHID-842`; the complete-miss control
does not contain it. The completed `4k_at_0k` sparse outputs differ from cold
token IDs; the other five completed sparse/cold cases match. Fact retention
in this synthetic fixture is not a general quality evaluation.

[Raw scalar measurements, configurations, coverage and binary/source hashes](validation/sparse-prefix-4096-s26u-20261110.json) preserve the incomplete cold
status and both disconnects. Repeated grid input arrays are stored as SHA256
fingerprints after checking exact input equality; the full input generator is
`tools/kv_cache/benchmark_htp_prefix_grid.cpp`.

## Reproduction

Run device programs sequentially with matching libraries, eight host threads,
`NNTR_HTP_KV_MANAGER=1` and the model's original Q8 configuration. TurboQuant
is disabled. The filtered grid keeps the original producer/checkpoint logic
but skips the separate 20K backup-overhead experiment.

```sh
htp_prefill_check
htp_prefix_grid_bench MODEL_DIR UNIQUE_FLASH_DIR grid20k.json 20480
htp_prefix_grid_bench MODEL_DIR UNIQUE_FLASH_DIR_2 grid4k.json 4096
htp_sparse_benefit MODEL_DIR sparse-cold.json 2 all cold 4096
htp_sparse_benefit MODEL_DIR sparse-cache.json 2 all cache 4096
python3 tools/kv_cache/summarize_sparse_benefit.py \
  sparse-cold.json sparse-cache.json sparse-summary
```

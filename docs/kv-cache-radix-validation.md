# Radix prefix cache validation

The prefix cache now indexes token sequences with a compressed radix tree.
Edges split at differing tokens, including inside a page. Complete-page
terminals keep stable block IDs for sessions, LRU and Flash files. Eviction
prunes empty branches and merges token-only unary nodes. KV storage remains
paged; cross-session sharing and Flash restoration use complete pages.

`NNTR_KV_PAGED=1` selects paged CPU storage; adding
`NNTR_KV_PREFIX_CACHE=1` enables the radix index. With paging disabled, the
core manager uses contiguous buffers and supports a single KV session.
Multi-session KV and radix prefix caching currently require paging.

## Functional checks

| Suite | Ubuntu x86-64 | Android arm64 |
| --- | --- | --- |
| KV core | 63 passed | 63 passed |
| CausalLM KV adapter | 26 passed | 26 passed |
| Serving, including Qwen3 0.6B FP32 HTTP | 19 passed | 19 passed |

The five added radix tests cover compressed-edge splitting, page-boundary
matching, eviction and merging with live session references, comparison with
a linear prefix oracle, exact branched Flash restoration, and corruption
isolation between sibling branches. Ubuntu KV core also passed a targeted
AddressSanitizer build with leak detection.

Android validation used a Galaxy S23 Ultra (SM-S918N), Android 14, NDK
27.2.12479018, android-29 and the CPU backend. No tests were skipped.
A separate Qwen3 0.6B FP32 run used two sessions, 8-token pages, 64 MiB DRAM,
128 MiB Flash and a 62 MiB restore reserve. A/B/A/B requests preserved each
session's responses and reused 24 tokens per repeated request. The final
lookup reported 12 Flash loads and zero I/O errors. SSE output matched the
ordinary response; session listing remained responsive during generation.

Local XML, logs, source/binary hashes and the HTTP driver are under
`build-kv-android/validation/radix-prefix-cache/`. These generated artifacts
and model weights are not committed.

## Android timing comparison

This is a preliminary comparison with the existing buffer path, selected
using `NNTR_CPU_KV_LEGACY=1`. That option bypasses the new CPU core manager;
the CausalLM buffer facade remains. It does not represent removal of every
KV management operation.

The completed workload used Qwen3 0.6B FP32 weights, FP16 KV, 128 prompt
tokens, 16 decode tokens, a 512-token capacity, 128-token prefill chunks,
and four threads pinned to CPUs 3-6. Each process had one warmup and three
measured runs. Two process orders, forward and reversed, supplied six
measured samples per mode. The table reports medians across those samples.
The first prompt token changed on each iteration to prevent prefix hits.
Model loading, HTTP and Flash I/O are excluded.

| Mode | Prefill ms | Change | Decode ms/token | Change |
| --- | ---: | ---: | ---: | ---: |
| Legacy buffer path | 2223.5 | baseline | 116.6 | baseline |
| Core, contiguous | 2231.5 | +0.4% | 117.1 | +0.4% |
| Paged | 2222.0 | -0.1% | 140.5 | +20.5% |
| Paged with radix prefix cache, cold | 2226.5 | +0.1% | 127.0 | +8.9% |

Paged decode varied substantially between processes: their medians were
124.1 and 156.1 ms/token. Clocks were not locked, and Android thermal status
0 does not exclude frequency or thermal effects. These results measure
whole execution paths, including attention and allocation differences;
they do not isolate the cost of radix lookup or management instructions.
The contiguous core showed a small difference in this workload, while the
paged paths need further investigation before assuming performance parity.

The run was stopped at the user's request. The 384-token comparison is
incomplete and excluded. Raw 128-token results, commands, thermal samples,
the C++ runner source and the host orchestration script are under
`build-kv-android/validation/kv-manager-ab-radix/`.

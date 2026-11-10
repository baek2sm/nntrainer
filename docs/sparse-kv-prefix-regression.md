# Existing-path regression checks for sparse prefix reuse

The comparator is commit `9034300447b30915963b14b32d74d4e0da78c77a`, before
sparse prefix support. The candidate includes `2b6bcb342` and the fast-path and
minimum-gain fixes described below. The Android host engine and CausalLM library
were built separately for each implementation with the same compiler/options.
The DSP/provider libraries and model files were held constant.

## Regressions found and fixed

1. With sparse enabled, a full exact hit still scanned all published histories
   for sparse matches. That cannot improve a hit already covering every
   eligible prompt token. Skip that search. The host lookup benchmark fell
   from 8.05 microseconds to 0.51 microseconds for this case.
2. A partial exact hit of 512 rows was displaced by a sparse hit of **513** rows:
   one extra newline token could be matched at a later source position. This
   triggered gathering and correcting the entire prefix for a one-token gain.
   The device fixture measured 1591 ms TTFT on that path. Automatic device
   matching now requires **64 additional reusable tokens** over the best exact
   candidate. Both the potential gain and the actual match are checked. The
   fixture again uses the original 512-row exact prefix and evaluates its
   suffix normally. Gains of 1 and 63 are rejected in host tests; a gain of 64
   is accepted. The direct CPU sparse API remains explicitly caller-controlled.
3. Prefix publication checks the existing disabled/empty conditions before
   looking up approximate-session metadata.

The gain threshold is a conservative dispatch rule, not a complete cost model.
It does not guarantee a speed win for every sparse hit, model, or context size.

## S26U comparison

SM-S948U, Gemma4 26B A4B, NPU attention and MoE, maximum context 2048,
prefill plane 512, `skip_prefill=true`, deterministic generation budget 32.
Q8 emitted 32 tokens (31 steady decode steps); TQ emitted 15 tokens (14 steady
steps, stopping at EOS), identically in baseline and candidate. Q8 and TQ
are compared to their own corresponding baseline, not to each other.

The benchmark excludes a per-process source warmup and measures 60 requests.
Ordinary Q8 and existing TurboQuant each have six samples per implementation,
using a current/baseline/baseline/current block order. Exact full hit, partial
hit and miss each have four samples per variant. Their block order is baseline
sparse-off, current off, current on, current on, current off, baseline off.
The producer prefix is frozen after warmup: measured branches are not published
and cannot turn a later partial-hit measurement into a full hit.

The ordinary request has 936 prompt tokens. The exact partial case reuses 512
rows and prefills 20 new rows; the full hit reuses 935 rows and still evaluates
the final prompt token. The miss is a separate short prompt and reuses zero
rows. All output token IDs, decode lengths, active KV payload bytes and retained
prefix payload bytes match across each corresponding baseline/candidate case.

Median changes relative to the pre-sparse baseline (negative means faster):

| Mode | Case | Prefill | Decode ms/token | TTFT |
| --- | --- | ---: | ---: | ---: |
| Sparse off, ordinary Q8 | Prefill + decode | -2.82% | -2.65% | -1.28% |
| Sparse off, existing TQ | Prefill + decode | -0.84% | -1.03% | -1.16% |
| Sparse off, exact cache | Full hit | — | -1.47% | +0.36% |
| Sparse off, exact cache | Partial hit | +1.19% | +1.51% | +0.13% |
| Sparse off, exact cache | Miss | +0.25% | -0.89% | +0.05% |
| Sparse on, exact cache | Full hit | — | -2.34% | -0.16% |
| Sparse on, exact cache | Partial hit | +1.40% | +0.58% | +2.15% |
| Sparse on, exact cache | Miss | +1.17% | -2.33% | +0.84% |

These runs show no material slowdown in the measured existing paths after the
fixes. They do **not** establish zero overhead or a statistically proven speedup:
frequency and temperature were not locked. An earlier comparison containing a
build gap showed a 2–3% Q8 prefill slowdown that did not persist in consecutive
cross-ordered runs. Differences within a few percent must be interpreted with
that variability. The report retains individual measurements and ranges.

The payload sizes match; total process memory is not claimed identical.
File-backed weight residency affects observed process peaks, and the new
management metadata has a small fixed cost. For example, on the checked x86-64
libstdc++ ABI, `sizeof(DeviceKVManager)` changes from 552 to 600 bytes and
`sizeof(Session)` from 32 to 40 bytes. This is not an Android ABI size claim.
The CPU core also has one additional provenance position per session. No new
KV tensor mirror is allocated when sparse is disabled.

## Lookup cost and remaining limits

`benchmark_sparse_lookup.cpp` uses eight 4096-token histories, 200,000 lookups
per case, five repetitions and one pinned host CPU. Median microseconds:

| Lookup | Baseline | Initial sparse implementation, off | Fixed, off | Fixed, on |
| --- | ---: | ---: | ---: | ---: |
| Full exact hit | 0.508 | 0.508 | 0.506 | 0.507 |
| Partial hit, enough possible gain to search | 0.058 | 0.058 | 0.057 | 6.389 |
| Long miss, enough possible gain to search | 0.003 | 0.003 | 0.003 | 6.124 |

Enabled sparse search still has a cost on long partial hits and misses. It
scans candidate histories and can allocate match plans. The full-hit guard
and minimum-gain bound eliminate searches that cannot help; they do not make
all searches free. Candidate count/context length can increase that cost.
Sparse reuse itself also copies KV, applies RoPE and may recalibrate Q8.

These comparisons cover this device, checkpoint, context and generation budget.
They do not validate other SoCs, long contexts beyond 2048, all RoPE variants,
Flash performance, or the entire platform build matrix. Existing host ownership,
snapshot and ring tests still pass: 76 core cases, 37 CausalLM KV manager cases,
and the device-manager ASan/UBSan check including the gain policy.

Raw measurements, equality checks, code and binary hashes:
[validation/sparse-prefix-regression-s26u-20261110.json](validation/sparse-prefix-regression-s26u-20261110.json).

Reproduction tools:

```sh
htp_sparse_regression MODEL_DIR plain-q8 3 OUTPUT_JSON
htp_sparse_regression MODEL_DIR plain-tq 3 OUTPUT_JSON
htp_sparse_regression MODEL_DIR exact-off 2 OUTPUT_JSON
htp_sparse_regression MODEL_DIR exact-on 2 OUTPUT_JSON
```

The baseline supports the first three modes. Build the same benchmark driver
against the corresponding baseline headers and libraries; do not mix the
runtime-config or device-manager layouts between builds. The final TQ driver
sets `attention_kv_dtype=turboquant_4bit_nc`; its Q8/exact settings remain Q8.
The only driver adjustment between Q8/exact measurements and TQ measurements
was that startup dtype selection, outside the measured inference calls.

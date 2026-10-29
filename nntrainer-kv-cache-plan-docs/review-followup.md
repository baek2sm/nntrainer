# CPU KV review follow-up (2026-10-29)

The external review reproduced Linux correctness at the final reviewed HEAD.
Its CI, runtime enforcement, default-performance and coverage findings are
being addressed before declaring the CPU work ready for an upstream PR.
`AGENTS.md`, vendored source, existing binary snapshots and model files remain
unchanged by this follow-up. Local execution artifacts remain uncommitted.

## CI checks

The CI changed-line clang-format-14 command reproduced two MHA formatting
suggestions even though formatting the whole file had passed. Level-1 Doxygen
reproduced missing leading-file tags in eight new C++ files. Both checks pass
after the targeted whitespace/header corrections. GNU indent is not executed
on the working tree because its checker mutates tracked files in place.
Evidence is in the ignored `build-kv-android/validation/ci-followup/` directory.

## Work order

1. Correct CI metadata and restore conservative Android packing defaults.
2. Connect checked core views and producer readiness to actual MHA operations;
   retain compatibility storage coverage and reject unsupported CPU ring use.
3. Measure load, prefill and decode with a dedicated no-dump benchmark, with
   identical arithmetic, weights, requests and thermal conditions per comparison.
4. Extend end-to-end capacity, repeated-session and failure-path verification.
5. Optimize only measured bottlenecks, then rerun correctness/performance gates.
6. Close the supported CPU scope and inspect available OpenCL/QNN capabilities
   before selecting the next backend. CPU/GPU physical sharing remains unproven.

## Attribution and branch policy

Human author/committer identity, DCO sign-off and the requested
`Co-authored-by: Codex (OpenAI)` trailer are preserved. GitHub may not display
an email-free trailer as a linked coauthor; no email address is invented.
Existing development history is retained. Independent bug-fix branches are
prepared without publishing PRs or rewriting the user's branch.

Prior documents are local development records. Portable instructions for an
upstream reviewer use configurable model/build paths and an explicit device
serial; local artifact paths alone are not presented as reproducibility proof.

## Checked execution bridge

The facade no longer marks all owners ready when a graph returns. A synchronous
`KvStepScope` binds the current lease to MHA across the application/layer DSOs.
The real producer checks its K/V write Tensor addresses and sizes against
`writeKey`/`writeValue` views before touching storage, publishes readiness after
both writes, and checks `readKey`/`readValue` before attention. Missing producer
completion now fails commit. A retained access rejects an expired graph scope.
Views still do not guard arbitrary escaped raw Tensor pointers.

CPU facade ring allocation and managed OHWI/ring/INT8 operations are rejected
explicitly. Experimental core ring views are not CPU attention support. Hybrid
models declare convolution blocks as non-producers; LFM2 prefill/decode now use
the same wrapper and return its pooled outputs through the matching release API.
Non-KV encoder execution is not claimed as a lease-backed KV path.

Tiny CausalLM adapters now execute the real wrapper rather than bypassing it.
Seven real-graph tests cover missing producers after actual MHA execution,
foreign K/V before writes, capacity/interval rejection, repeated reset, hybrid
decode and pooled output reuse. Linux normal Meson passes core22, facade25,
legacy20, ring22, graph16, policy23 and models106 plus one existing QINT4 skip.
ASan/UBSan/LSan passes all22 core cases. Performance is evaluated separately.

The `cache_index = _from` behavior predated this work. Earlier storage ownership
refactoring is not described as a new arithmetic optimization or full runtime
view enforcement. The current adapter adds actual scoped checks and producer
completion; it does not provide rollback of arbitrary writes or async fences.

## Independent review branches

- `fix/graph-refinalize-mode`: `21f415b01`, four product/test files, no journal.
- `fix/qwen3-layergraph-weight-order`: `b78b52f6b`, converter plus eight synthetic
  CLI cases (failing before the order fix and passing afterwards).
- `fix/gemma4-skip-prefill-lmhead`: existing `21d03c477`, related fix/test/document.

All three are local, independent single-commit branches based on main. No PRs
were published and the user's development history was not rewritten. The
separate Gemma document remains with its dedicated review branch; the combined
development branch retains history and is not the proposed upstream PR payload.

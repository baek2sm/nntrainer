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

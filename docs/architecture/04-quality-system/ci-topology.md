# CI Topology & Enhancement Plan

> **The cross-platform proof engine.** Most targets cannot be built locally, so CI
> *is* the evidence that a change is portable. This document maps the current
> GitHub Actions surface to what each gate proves, then proposes enhancements so
> the matrix is the reliable middle of the trust pyramid.

---

## 1. Current CI surface (`.github/workflows/`)

| Workflow | Proves | Locally runnable? |
|---|---|---|
| `windows.yml` | MSVC x64 meson build + test | **Yes** (helper script) |
| `daily_build_windows_x86_64.yml` | Windows x64 daily | Partly (build) |
| `windows_arm_clang.yml`, `daily_build_windows_arm.yml` | Windows ARM / clang | No — reason + CI |
| `windows_cross_compile.yml` | Windows cross-compile | No |
| `ubuntu_clean_meson_build.yml` (+`_clang`) | Clean Linux build (gcc/clang) | No |
| `android.yml`, `daily_build_android.yml`, `causallm_android.yml` | Android NDK build | No |
| `daily_build_tizen_arm.yml`, `daily_build_tizen_x86_64.yml`, `gbs_build.yml` | Tizen GBS packaging | No |
| `yocto.yml` | Yocto build | No |
| `pdebuild.yml` | Debian package build | No |
| `cpp_linter.yml` | clang-format 14 on **changed lines** | Partly |
| `static.check.yml` (+ `static.check.scripts/`) | doxygen tags, cppcheck, spelling, prohibited words, exec bits, newline, shellcheck/pylint | Partly |
| `check_count.yml` | commit count / body length (`nobody.sh`) | Yes |
| `coverage.yml` | test coverage | No |
| `codeql.yml` | security static analysis | No |
| `ubuntu_benchmarks.yml` | perf benchmarks | No |
| `pr-desc.yml` | AI PR description (GitHub Models) | n/a — the bot pattern we reuse |
| `labeler.yml`, `close_stale_pr.yml`, `pr-desc.yml` | repo automation | n/a |

CI triggers on `pull_request` to `main`.

---

## 2. What CI is the *only* evidence for

Because we can only execute Windows x64 MSVC locally, these are trusted **only**
via CI and must be treated as the real signal:

- Linux gcc/clang portability (`ubuntu_*`).
- Android NDK / JNI ABI (`android.yml`, `causallm_android.yml`).
- Tizen / Yocto / Debian packaging.
- Windows ARM / clang.
- Coverage and CodeQL.

A green local Windows build is **necessary, not sufficient**. The cross-platform
hazards in [cross-platform.md](../03-crosscutting/cross-platform.md) are exactly
what these gates catch.

---

## 3. Enhancement plan (draft)

Goal: make CI a dependable middle tier — fast feedback, clear required set,
strong cross-platform coverage, and tight integration with the AI bot and the
doc-sync contract.

### 3.1 Define a required-checks set
Designate the gates that **must** be green to merge (e.g. `windows.yml`,
`ubuntu_clean_meson_build*.yml`, `cpp_linter.yml`, `static.check.yml`,
`check_count.yml`, the new doc-sync check). Branch protection enforces them so a
red required gate blocks merge regardless of human approval.

### 3.2 Fast-feedback ordering
Run cheap gates first (format, static check, commit hygiene, Windows build) so
authors get signal in minutes; reserve long matrix builds (Tizen/Yocto/Android)
for after the cheap gates pass or on a label.

### 3.3 Doc-sync gate (the contract's teeth)
A new check: if a PR touches architecture-significant paths
(`nntrainer/engine.*`, `context*`, `*_context.*`, subsystem dirs, public API
headers) but **not** `docs/architecture/`, fail with a message pointing at the
[doc-maintenance contract](doc-maintenance-contract.md). Allow an explicit
`skip-arch-doc` label (with justification) for genuine no-op cases.

### 3.4 Coverage delta
Surface coverage change per PR (from `coverage.yml`); flag PRs that add code
without adding tests. Start advisory, make it required once stable.

### 3.5 AI-review gate
Add `ai-review.yml` ([ai-review-bot](ai-review-bot.md)) running on PRs, posting a
structured review comment. Start advisory (comment only), never auto-blocking on
the model's verdict — humans and the deterministic gates remain the authority.

### 3.6 Full-matrix-on-label
A `full-ci` label (or scheduled daily, which already exists) to run the complete
cross-platform matrix on demand, so heavy jobs don't gate every push but are one
click away before merging risky changes.

### 3.7 Required-check honesty in reports
Any automated/agent report must label each gate **executed** vs **reasoned**.

---

## 4. Mapping to the local CI-equivalent checks

The orchestrator's `architecture-and-ci.md` §6 and the `review-for-nntr` skill
already encode local equivalents (clang-format changed-lines, doxygen, commit
hygiene, cross-platform hazards). These remain the pre-push oracle; CI is the
authoritative confirmation. Keep the two in sync — when a CI gate changes, update
§6 and this doc together.

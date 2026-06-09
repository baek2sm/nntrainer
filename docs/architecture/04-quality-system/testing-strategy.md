# Testing Strategy (TDD) & Local Unit-Test Enhancement

> **Base of the trust pyramid.** Low-level correctness is established by tests
> written *before or with* the change. If the test suite is trusted, humans can
> stop re-deriving low-level correctness by eye.

---

## 1. Principle: test-first

For any feature or bugfix:

1. Write a failing test that encodes the expected behavior (cite the relevant
   `INV-<AREA>-N` invariant where one applies).
2. Implement until it passes — and the rest of the suite stays green.
3. Refactor under green.

A bug fix starts with a test that **reproduces** the bug (red), then the fix
turns it green. This is the `superpowers:test-driven-development` discipline; it
is the default, not the exception.

---

## 2. Test taxonomy (where things go)

| Kind | Location | Proves |
|---|---|---|
| **Unit** | `test/unittest/<module>/` and `test/unittest/unittest_*.cpp` | One type/op in isolation; fast, deterministic. |
| **Dispatch contract** | `unittest_compute_ops_dispatch.cpp`, `unittest_nntrainer_fallback.cpp` | The backend backbone invariants (INV-BCK/TEN). |
| **Layer** | `test/unittest/layers/` (66+ files) | Forward **and** backward, properties, golden values. |
| **Model / integration** | `test/unittest/models/`, `integration_tests/`, `test_models/` | End-to-end build/train/infer, save/load round-trip. |
| **API-level** | `test/ccapi/`, `test/tizen_capi/`, `test/jni/`, `test/nnstreamer/` | Public surfaces (INV-CTX-3). |

Built with GoogleTest via Meson; runs in `meson test`.

---

## 3. What "tested" means per architecture layer

| Layer | Minimum bar before merge |
|---|---|
| Tensor op | Unit test for the op (fp32; + fp16/quant if implemented) **and** a dispatch test if it adds a `ComputeOps` method. |
| Layer | Forward + backward numeric test with known/golden values; property validation test. |
| Optimizer | Update math vs a known trajectory; scheduler determinism. |
| Graph/compiler | Execution-order / realizer-order test; negative test for bad input. |
| Model | At least one train/infer test; save/load round-trip if serialization touched. |
| Backend | Dispatch test + `supports_*()` fallback test. |

Negative tests (invalid config, unsupported op, malformed model) are first-class
— most "fail loud" invariants (INV-ERR, INV-LAY-3, INV-CMP-3) need one.

---

## 4. Local execution (what we can run here)

Windows x64 MSVC is the only locally-runnable target:

```powershell
powershell -ExecutionPolicy Bypass -File .claude/nntr/nntr-build.ps1 -RepoRoot <wt> -Stage all      # compile + test
powershell -ExecutionPolicy Bypass -File .claude/nntr/nntr-build.ps1 -RepoRoot <wt> -Stage compile  # build only
```

This reproduces `windows.yml` (vcvars64 → meson 1.7.2 → setup/compile/test).
Everything else is established by reasoning + CI ([ci-topology](ci-topology.md)).
A report must state which tests **ran** vs were **reasoned about**.

---

## 5. Quality bars for tests themselves

- **Deterministic.** No reliance on wall-clock, unseeded RNG, network, or
  hardware presence. Seed all randomness (INV-DAT-2). A flaky test is a bug to
  fix, never a retry to paper over.
- **Focused & fast.** Prefer narrow unit tests over broad integration tests that
  are hard to debug when red.
- **Low duplication.** Reuse `nntrainer_test_util.*` and shared fixtures.
- **Portable.** No env-specific paths or host-only assumptions (they break the
  cross-platform CI the test is supposed to protect).

---

## 6. Local unit-test enhancement plan (draft backlog)

Concrete, independently-shippable improvements that raise the floor:

1. **Coverage map.** Use `test/unittest/unittestcoverage.py` / `coverage.yml` to
   publish a per-module coverage number; identify the lowest-covered invariants.
2. **Invariant-citing tests.** Backfill tests that explicitly assert the numbered
   invariants (e.g. a test named for INV-TEN-3 cross-vendor throw).
3. **Negative-path coverage.** Add missing "fail loud" tests at finalize/compile.
4. **Determinism audit.** Find and fix tests depending on ordering/timing.
5. **Golden-value fixtures.** Standardize numeric layer tests against stored
   references to make backward-pass regressions obvious.
6. **Fast subset target.** A `meson test` suite tag for a <2 min pre-push smoke
   set, distinct from the full suite.

Each item is small enough to be its own PR with its own test — and a good fit for
the nntr-orchestrator pipeline.

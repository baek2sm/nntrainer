# L3 — Dataset (`nntrainer/dataset/`)

> **Layer 3 component.** Feeds training/inference batches to the model loop.

## 1. Responsibility

Provide data producers and loaders behind a uniform producer interface so the
model loop can pull batches without knowing the source (file, generator,
in-memory), with optional preprocessing.

## 2. Internal structure

- A `DataProducer`/loader interface + concrete producers (file-based,
  function/generator-based, etc.).
- Batch assembly, shuffling, and iteration bookkeeping.

## 3. Contracts & invariants

- **INV-DAT-1 — Uniform producer interface.** The model loop depends only on the
  producer abstraction, not concrete sources.
- **INV-DAT-2 — Deterministic with a seed.** Shuffling/iteration is reproducible
  given a seed (needed for stable tests and INV-CTX-4).
- **INV-DAT-3 — No hardware/env assumptions in core.** Paths and sources are
  configured, not hardcoded (cross-platform rule).

## 4. Dependencies

- **Uses:** `tensor/` (batch buffers), `utils/`.
- **Used by:** `models/` (train/infer loop).

## 5. Change checklist

- [ ] New producer implements the full interface + iteration semantics.
- [ ] Determinism under a fixed seed tested.
- [ ] No env-specific paths; portable across targets.

## 6. Review focus & common pitfalls

- Off-by-one at epoch/batch boundaries; last partial batch handling.
- Flaky tests from nondeterministic ordering.
- Threaded producers racing on shared buffers (see
  [concurrency](../03-crosscutting/concurrency.md)).

## 7. Tests

- `test/unittest/datasets/` (10 files).

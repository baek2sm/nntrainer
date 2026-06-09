# L3 — Optimizers (`nntrainer/optimizers/`)

> **Layer 3 component.** Weight-update algorithms and learning-rate scheduling.

## 1. Responsibility

Implement optimizers (SGD, Adam, …) that consume gradients and update weights,
plus learning-rate schedulers that drive the LR over training.

## 2. Internal structure

- Optimizer interface + concrete impls; per-weight optimizer state (e.g. Adam
  moments) allocated through the tensor layer.
- Learning-rate schedulers (constant, step, exponential, …) behind a common API.

## 3. Contracts & invariants

- **INV-OPT-1 — Stateless step contract.** `apply(gradients, weights)` is the only
  way state advances; no hidden global mutation.
- **INV-OPT-2 — Optimizer state lifecycle.** Per-weight state is sized at
  finalize and reused; not reallocated per step (INV-CTX-1).
- **INV-OPT-3 — Schedulers are pure functions of step/epoch.** Deterministic and
  unit-testable.

## 4. Dependencies

- **Uses:** `tensor/` (state + math), `utils/` (properties).
- **Used by:** `models/` (train loop), `layers/` (weights).

## 5. Change checklist

- [ ] New optimizer: math unit-tested against known trajectories.
- [ ] State sizing happens at finalize; no per-step allocation.
- [ ] Scheduler determinism tested across step/epoch boundaries.

## 6. Review focus & common pitfalls

- Numeric correctness (epsilon, bias correction) — golden-value tests.
- State not reset between runs.
- fp16/quantized weight update precision.

## 7. Tests

- `test/unittest/optimizers/`, `unittest_nntrainer_lr_scheduler.cpp`.

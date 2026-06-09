# L3 — Models (`nntrainer/models/`)

> **Layer 3 component.** The top of the dependency stack: orchestrates the whole
> build → compile → train/infer lifecycle.

## 1. Responsibility

Provide `NeuralNetwork` (and model-level config/wrappers): assemble layers into a
graph, compile/finalize, run the training loop (forward → loss → backward →
optimizer step), run inference, and save/load via the schema.

## 2. Internal structure

```
NeuralNetwork
  ├─ setProperty / addLayer / setOptimizer / setDataset
  ├─ compile()  → builds NetworkGraph (via compiler/graph)
  ├─ initialize()/finalize() → finalizeContext() stamps tensors
  ├─ train()    → epoch loop over dataset: forward/backward/apply
  ├─ inference()
  └─ save()/load() → schema serialization
```

## 3. Contracts & invariants

- **INV-MOD-1 — Lifecycle order.** `compile → finalize → train/infer`. Calling out
  of order is an error surfaced to the caller, not undefined behavior.
- **INV-MOD-2 — Orchestration only.** `models/` coordinates; it does not
  reimplement layer math, graph ordering, or tensor ops.
- **INV-MOD-3 — Round-trip save/load.** A model saved then loaded reproduces
  results within tolerance, across targets (INV-CTX-4).

## 4. Dependencies

- **Uses:** `compiler/`, `graph/`, `layers/`, `optimizers/`, `dataset/`,
  `tensor/`, `schema/`. Top of the stack — nothing in core depends on it.

## 5. Change checklist

- [ ] Lifecycle transitions guarded and tested.
- [ ] Train/infer numeric behavior covered by a model-level test.
- [ ] Save/load round-trip tested if serialization touched.
- [ ] Public C++/CAPI surface changes flagged (INV-CTX-3) → human review.

## 6. Review focus & common pitfalls

- State leaking between epochs/runs; non-deterministic training.
- Public-API breakage (wide blast radius).
- Save/load schema drift.

## 7. Tests

- `test/unittest/models/` (14 files), `unittest_nntrainer_models.cpp`,
  `unittest_nntrainer_modelfile.cpp`, `unittest_nntrainer_save_with_dtype.cpp`,
  and integration tests in `test/unittest/integration_tests/`.

# L3 — Layers (`nntrainer/layers/`)

> **Layer 3 component.** Where most feature work lands: the catalog of layer
> types and the `LayerNode` that wraps them for the graph.

## 1. Responsibility

Implement layers (forward/backward math, weight/gradient management, property
handling) behind a uniform interface so the graph can treat them uniformly.

## 2. Internal structure

- A `Layer` interface + `LayerNode` wrapper (graph-facing adapter: name,
  connections, properties, weights).
- ~60+ concrete layers (conv, pooling, fully_connected, normalization,
  activation, attention, rnn/lstm/gru families, embedding, reshape, etc. — see
  `docs/components.md` for the user-facing catalog).
- `LayerImpl`/common helpers for shared property and weight plumbing.

## 3. Contracts & invariants

- **INV-LAY-1 — Uniform lifecycle.** Every layer implements the same
  finalize → forward → calcDerivative/calcGradient lifecycle the graph calls.
- **INV-LAY-2 — Tensor ops only via Tensor API.** Layers compute through tensor
  ops; they do not select backends or call kernels directly (INV-CONT-1).
- **INV-LAY-3 — Properties validated at finalize.** Invalid configs throw at
  finalize, not silently mid-forward.
- **INV-LAY-4 — Registration.** New layers register through the app context
  factory so they're reachable by name from INI/compiler.

## 4. Dependencies

- **Uses:** `tensor/`, `utils/` (properties), `optimizers/` indirectly (weights).
- **Used by:** `graph/` (as `LayerNode`), `compiler/` (instantiates by name).

## 5. Change checklist

- [ ] Forward **and** backward implemented and unit-tested (golden values).
- [ ] Properties documented (doxygen) and validated at finalize.
- [ ] Registered in the factory; reachable by INI keyword if user-facing.
- [ ] fp16 / quantized behavior considered.
- [ ] If user-facing, update `docs/components.md` catalog too.

## 6. Review focus & common pitfalls

- Backward/gradient correctness (the classic bug site) — needs numeric tests.
- Shape inference at finalize; broadcasting edge cases.
- Hidden per-step allocations hurting on-device budgets (INV-CTX-1).

## 7. Tests

- `test/unittest/layers/` (66+ files) — per-layer forward/backward & property
  tests. New layers **must** add a test here (TDD).

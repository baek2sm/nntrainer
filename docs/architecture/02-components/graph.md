# L3 — Graph (`nntrainer/graph/`)

> **Layer 3 component.** The compute-graph representation that sits between the
> compiler (which builds it) and the model loop (which executes it).

## 1. Responsibility

Represent the network as an ordered graph of `LayerNode`s; compute execution
order; and **finalize context** — allocate tensors and stamp each with the right
`ContextData` so the backbone can dispatch (see [backends](backends.md)).

## 2. Internal structure

- `network_graph.*` — the central `NetworkGraph`: node insertion, topological
  execution ordering, `finalizeContext()`, in/out connection wiring.
- Graph-level utilities for traversal and validation.

```
LayerNodes ──topo-sort──▶ execution order
                              └─ finalizeContext():
                                   allocate tensors via tensor pool/planner
                                   stamp each tensor's ct_data_ (ContextData)
```

## 3. Contracts & invariants

- **INV-GRA-1 — Single stamping site.** ContextData is attached to tensors in
  `network_graph.cpp::finalizeContext()` and nowhere else. (Mirrors INV-CONT-2.)
- **INV-GRA-2 — Deterministic execution order.** The computed order is stable for
  a given model; ordering changes are behavior changes and need tests.
- **INV-GRA-3 — Allocation through the pool.** Tensors are allocated via the
  tensor pool/planner, not ad hoc, so memory reuse holds (see
  [memory-management](../03-crosscutting/memory-management.md)).

## 4. Dependencies

- **Uses:** `layers/` (`LayerNode`), `tensor/` (pools, ContextData), the backbone.
- **Used by:** `models/` (drives forward/backward over the order), `compiler/`
  (produces the graph).

## 5. Change checklist

- [ ] Any change to execution ordering has a graph unit test.
- [ ] `finalizeContext()` still the only stamping site.
- [ ] Allocation still flows through the pool/planner.
- [ ] Memory-reuse / lifetime implications reviewed.

## 6. Review focus & common pitfalls

- Off-by-one / cycle issues in ordering; multi-output (`MultiOut`) wiring.
- Tensor lifetime vs pool reuse — premature reuse causes silent corruption.
- Context stamping missed for a newly introduced tensor.

## 7. Tests

- `test/unittest/unittest_nntrainer_graph.cpp`,
  `unittest_nntrainer_exe_order.cpp`, and `test/unittest/graph/`.

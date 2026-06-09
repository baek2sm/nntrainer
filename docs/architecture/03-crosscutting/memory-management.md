# L4 — Memory Management

> **Cross-cutting.** On-device budgets (INV-CTX-1) make memory an architectural
> concern, not a micro-optimization. Deep reference: `docs/memory-management.md`.

## The model

Tensors are allocated and reused through a **pool + planner**, not via ad-hoc
`new`/`malloc` per op:

```
TensorPool  ──asks──▶  Planner (basic_planner / others)
                          └─ assigns offsets so non-overlapping-lifetime
                             tensors share the same backing memory
Cache layers (cache_*) manage spill/reload when memory is tight.
```

Allocation is wired during `graph::finalizeContext()` (INV-GRA-3), so by train
time the memory map is fixed.

## Invariants

- **INV-MEM-1 — Allocate via the pool.** New tensors in the graph go through the
  pool/planner; no ad-hoc allocation that escapes lifetime planning.
- **INV-MEM-2 — Lifetime correctness.** A planner reuses memory only across
  tensors whose lifetimes don't overlap. A wrong lifetime → silent corruption.
- **INV-MEM-3 — No per-step growth.** Steady-state training must not allocate per
  step (optimizer state, layer scratch sized at finalize).
- **INV-MEM-4 — ContextData survives reallocation.** `CREATE_IF_EMPTY_DIMS` may
  reallocate an output mid-kernel; ct_data is propagated **after** the kernel
  (INV-TEN-2).

## Review focus & pitfalls

- Hidden copies on tensor construction/slicing/broadcast.
- A new tensor that bypasses the pool.
- Lifetime overlap bugs surfacing only under specific graphs/batch sizes.
- Growth over epochs (leak) — watch optimizer/cache state.

## Tests

`test/unittest/memory/` (6 files), `unittest_nntrainer_tensor_pool*.cpp`.

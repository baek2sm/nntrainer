# L4 — Concurrency & Initialization Order

> **Cross-cutting.** Thread-safety of shared state, init ordering, and
> shared-tensor hazards.

## Init ordering

- **INV-CON-1 — Guarded singleton init.** `ensureComputeOps()` is
  `std::call_once`-guarded and is the only path that binds global CPU
  `g_compute_ops` (INV-BCK-2). Never call `init_backend()` directly.
- **INV-CON-2 — Engine before Contexts.** Engine binds compute ops once, then
  registers Contexts (INV-BCK-3). Don't introduce code that touches a Context
  before the Engine has initialized.

## Shared-tensor hazards

- **INV-CON-3 — Tensors aren't implicitly thread-safe.** A tensor shared across
  threads needs external synchronization; ops don't lock internally.
- **INV-CON-4 — ContextData is shared_ptr.** Copying a tensor shares the
  `ContextData`; lifetime is refcounted, but the underlying device queue/handle
  may not be thread-safe — don't submit from two threads without coordination.

## Data producers

- Threaded `dataset/` producers must not race on shared batch buffers (INV-DAT
  determinism); guard handoff between producer and consumer.

## Review focus & pitfalls

- New global/static mutable state → must be init-order-safe and documented here.
- Parallel sections sharing tensors without synchronization.
- OpenCL/QNN queue submitted from multiple threads.
- Non-deterministic test failures = likely a race; treat as a real bug
  (see [systematic-debugging] discipline), not flakiness to retry away.

## Tests

`test/unittest/unittest_nntrainer_task.cpp` and threading helpers in `utils/`.

# L3 — Dispatch Backbone (`engine` / `context` / `*_context`)

> **Layer 3 component.** The machinery that routes a tensor op to CPU, GPU, or
> NPU. This is the contract every new backend must follow. The full rationale
> lives in `docs/backend_guide/ARCHITECTURE.md`; the cross-cutting view is
> [`../03-crosscutting/dispatch-and-backends.md`](../03-crosscutting/dispatch-and-backends.md).
> This doc is the component-level reference.

## 1. Responsibility

Own the dispatch chain and per-vendor entry points so tensor ops reach the right
backend without conditional compilation at the call site.

## 2. Internal structure

| File | Role |
|---|---|
| `engine.{h,cpp}` | Process-wide singleton; registry of `Context` by name; `ensureComputeOps()`. |
| `context.h` | Abstract `Context` — user-facing per-vendor entry point. |
| `context_data.h` | `ContextData` — per-vendor state blob (compute_ops ptr, allocator, queue…). |
| `app_context.*` | CPU `Context` impl. |
| `cl_context.*` | OpenCL `Context` impl (+ `cl_svm_allocator`, buffer mgmt). |
| `qnn_context.*` | QNN/NPU `Context` impl. |
| `mem_allocator.*` | Allocation policy used by ContextData. |

```
Engine  ──registers──▶  Context ("cpu"|"gpu"|"qnn")
                          └─ owns shared_ptr<ContextData>
                                └─ compute_ops : ComputeOps*  (~80 virtual kernels)
```

## 3. Contracts & invariants

- **INV-BCK-1 — Default-throw, not pure-virtual.** Unimplemented `ComputeOps`
  methods throw `std::runtime_error("<op> not implemented by this backend")`.
  Backends override only what they support. **Never** convert these to `= 0`.
- **INV-BCK-2 — Single init path.** `ensureComputeOps()` binds the global CPU
  `g_compute_ops` and is `std::call_once`-guarded. **Never** call `init_backend()`
  directly.
- **INV-BCK-3 — Startup order.** `Engine` calls `ensureComputeOps()` once, then
  registers each available `Context`.
- **INV-BCK-4 — supports_*() predicates.** Accelerator-only ops default their
  predicate to `false`; call sites must branch + fall back.
- **INV-BCK-5 — Vendor gating in one place.** A backend is compiled in via an
  `enable-<x>` meson option and wired in `Engine::add_default_object` under
  `#if ENABLE_X`. The `#if` lives **here**, not at tensor call sites.

## 4. Adding a backend (checklist)

1. Headers/kernels under `nntrainer/tensor/<x>_operations/`, gated by `enable-<x>`.
2. A `ContextData` subclass (only if it needs state).
3. A `ComputeOps` subclass overriding supported ops + `supports_*()`.
4. A `Context` subclass mirroring `ClContext`/`QNNContext`.
5. Wire into `Engine::add_default_object` under `#if ENABLE_X`.
6. Add a dispatch unit test.
7. **Update this doc + `01-container-view.md`** and add an
   [ADR](../adr/) — a new backend changes the architecture shape.

Full checklist: `docs/backend_guide/ARCHITECTURE.md` §7.

## 5. Dependencies

- **Uses:** `tensor/` types, `mem_allocator`, OpenCL/QNN device plumbing.
- **Used by:** `graph/` (`finalizeContext()` stamps ContextData), `tensor/`
  (`getOps()`).

## 6. Review focus & common pitfalls

- A new op added as pure-virtual (breaks every backend) — must be default-throw.
- Backend selection leaking to a call site as `#ifdef`.
- Forgetting the `supports_*()` fallback → runtime throw on unsupported HW.
- Thread-safety of init; see [concurrency](../03-crosscutting/concurrency.md).

## 7. Tests

- `test/unittest/unittest_compute_ops_dispatch.cpp` — the dispatch contract.
- `test/unittest/unittest_nntrainer_fallback.cpp` — unattached → global CPU path.
- `test/unittest/unittest_nntrainer_appcontext.cpp` — Context registration.
- OpenCL kernel tests under `unittest_opencl_*`, `unittest_attention_kernels_cl.cpp`.

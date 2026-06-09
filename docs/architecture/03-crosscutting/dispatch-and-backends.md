# L4 — Dispatch & Backends

> **Cross-cutting.** The contract that lets tensor ops reach CPU/GPU/NPU without
> `#ifdef` at the call site. Component-level detail: [backends](../02-components/backends.md).
> Full rationale: `docs/backend_guide/ARCHITECTURE.md` (authoritative).

## The chain

```
Engine ──▶ Context ("cpu"|"gpu"|"qnn") ──▶ ContextData ──▶ ComputeOps* ──▶ kernels
                                              (on TensorBase::ct_data_)
```

## The rules (consolidated invariants)

These restate the per-component invariants from the cross-cutting angle, so a
reviewer can check a backend-touching PR from one page:

1. **Default-throw, not pure-virtual** (INV-BCK-1). New ops are virtual methods
   with a throwing default body. Backends override only what they support.
2. **Single init path** (INV-BCK-2/3). `ensureComputeOps()` (call_once) binds CPU
   `g_compute_ops`; never call `init_backend()` directly.
3. **ContextData on TensorBase** (INV-TEN-1), stamped at compile time in
   `finalizeContext()` (INV-GRA-1), propagated to op results **after** the kernel
   (INV-TEN-2).
4. **Cross-vendor op mismatch throws** `std::invalid_argument` (INV-TEN-3);
   unattached tensors fall back to global CPU `g_compute_ops`.
5. **Accelerator ops are opt-in** via `supports_*()` (default false) + call-site
   fallback loop (INV-TEN-5). No preprocessor branches at call sites.
6. **Vendor gating in one place** — `#if ENABLE_X` only in
   `Engine::add_default_object` and under `enable-<x>` meson options (INV-BCK-5).

## Why it's shaped this way

- **No `#ifdef` sprawl:** backends are added by subclassing, not by editing every
  call site. New hardware = new files + one wiring point.
- **Fail loud, not silent:** default-throw means a missing kernel surfaces as a
  clear runtime error instead of wrong numbers.
- **Backward compatible:** legacy tensors with no ContextData still run on CPU.

## Review focus for backend-touching PRs

- Did a new op land as pure-virtual? → reject (breaks all backends).
- Is backend selection leaking to a call site? → reject.
- Missing `supports_*()` fallback? → unsupported-HW crash risk.
- New backend without an ADR + doc updates? → architecture change, needs both.

## Tests

`unittest_compute_ops_dispatch.cpp`, `unittest_nntrainer_fallback.cpp`,
`unittest_nntrainer_appcontext.cpp`, OpenCL kernel tests.

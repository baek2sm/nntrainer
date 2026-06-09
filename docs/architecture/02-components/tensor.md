# L3 — Tensor (`nntrainer/tensor/`)

> **Layer 3 component.** The foundational data type. Almost every other
> subsystem depends on it, so its contracts are the most widely-felt in the
> codebase.

## 1. Responsibility

Provide the tensor abstraction and core tensor operations: shape/stride/layout,
storage and lifetime (pools/planners/cache), data-type handling (fp32, fp16,
quantized int4/q4_0/char), and the per-op dispatch into the backend backbone.

## 2. Internal structure

```
Tensor            public Pimpl handle — thin, forwards to TensorBase
  └─ TensorBase   storage metadata: dim, strides, data, ct_data_ (ContextData)
        └─ getOps()  resolves the ComputeOps* for this tensor's backend
              └─ ComputeOps virtual call → kernel (NEON/AVX/OpenCL/QNN)
```

- **Core types:** `tensor.*`, `tensor_dim.*`, `tensor_base.*`, `float_tensor.*`,
  `half_tensor.*`, `char_tensor.*`, `var_grad.*`.
- **Pools & planning:** `tensor_pool.*`, `basic_planner.*`, `cache_*` — memory
  reuse across the graph; see [memory-management](../03-crosscutting/memory-management.md).
- **Quantization:** quantizer + quantized tensor types and the `supports_*()`
  accelerator-op predicates (q4_0 batch GEMM, int4 GEMV/GEMM).
- **Backends:** CPU math (`cpu_backend/*`, OpenBLAS/NEON/AVX), OpenCL
  (`cl_operations/*`, `.cl` kernels). Dispatch detail: [backends](backends.md).

## 3. Contracts & invariants

- **INV-TEN-1 — ContextData on TensorBase.** `std::shared_ptr<ContextData> ct_data_`
  is a member of `TensorBase`, **not** the `Tensor` wrapper. Do not move it.
- **INV-TEN-2 — Post-kernel ct_data propagation.** Binary/unary ops propagate the
  receiver's ct_data to the result **after** the kernel runs, because
  `CREATE_IF_EMPTY_DIMS` may reallocate the output mid-kernel.
- **INV-TEN-3 — Cross-vendor mismatch throws.** `a.dot(b)` with two *different*
  non-null ContextData throws `std::invalid_argument` (pointing at `Tensor::to()`).
  If either side is unattached, it falls back to global `g_compute_ops`.
- **INV-TEN-4 — No hidden backend selection.** A tensor op never picks a backend
  with `#ifdef`; it resolves via `getOps()` on its own ct_data.
- **INV-TEN-5 — Accelerator ops are opt-in.** Accelerator-only kernels are guarded
  by a `supports_*()` predicate (default `false`); call sites branch on it and
  fall back to a single-op loop. No preprocessor branches.

## 4. Dependencies

- **Uses:** `utils/` (properties, fp16, threading), the dispatch backbone, BLAS.
- **Used by:** `layers/`, `models/`, `graph/`, `optimizers/`, `dataset/` — almost
  everything. Per INV-CONT-3, `tensor/` must **not** depend upward on those.

## 5. Change checklist

- [ ] New op added to `ComputeOps` as a **default-throw** virtual (never `= 0`).
- [ ] CPU implementation provided; other backends override only if supported.
- [ ] Accelerator-only op paired with a `supports_*()` predicate + fallback loop.
- [ ] Shape/stride/broadcast math checked at boundaries; no hidden copies.
- [ ] fp16 / quantized paths considered (or explicitly N/A).
- [ ] Dispatch unit test added/updated (`unittest_compute_ops_dispatch.cpp`).
- [ ] Cross-platform types audited (integer width, pointer size) — see
      [cross-platform](../03-crosscutting/cross-platform.md).

## 6. Review focus & common pitfalls

- **ABI/API reach:** Tensor is foundational; signature changes ripple everywhere.
- **Hidden copies** on construction/slicing; in-place vs out-of-place correctness.
- **CPU vs OpenCL divergence** for the same op — a top regression source.
- **Concurrency:** tensors shared across threads; see
  [concurrency](../03-crosscutting/concurrency.md).

## 7. Tests

- `test/unittest/unittest_nntrainer_tensor*.cpp` (fp32/fp16/nhwc/advanced/pool/types/utils)
- `test/unittest/unittest_compute_ops_dispatch.cpp` (the dispatch contract)
- `test/unittest/unittest_nntrainer_cpu_backend*.cpp`, NEON kernels, OpenCL BLAS.
- New tensor behavior **must** ship with a unit test (TDD) — see
  [testing-strategy](../04-quality-system/testing-strategy.md).

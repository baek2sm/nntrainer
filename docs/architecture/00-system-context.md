# L1 — System Context

> **Layer 1 (Context).** The outermost view: what NNTrainer is, who and what
> interacts with it, what it depends on, and which platforms it must run on.
> This is the most stable document in the tree — changes here are rare and
> always warrant human architecture review.

---

## Responsibility

NNTrainer is an **on-device neural-network training and inference library**
written in C++ (Samsung; upstream `nntrainer/nntrainer`). It trains and runs
models *on the target device itself* — phones, TVs, wearables, embedded boards —
rather than in a datacenter, under tight memory and compute budgets.

The build system is **Meson**. The library exposes C and C++ APIs, plus
language/framework bindings, and integrates with the NNStreamer media pipeline.

---

## Context diagram

```
                         ┌────────────────────────────────────────┐
                         │            Application code              │
                         │  (C++ API, C API/CAPI, JNI, NNStreamer)  │
                         └───────────────────┬──────────────────────┘
                                             │ build/train/infer
                                             ▼
   ┌──────────────┐        ┌────────────────────────────────────────────┐
   │  Model files │◀──────▶│                NNTrainer                     │
   │ INI / ONNX / │ load/  │  Engine · Graph · Layers · Tensor · Compiler │
   │ TFLite /     │ save   │  Optimizers · Dataset · Backends             │
   │ safetensors  │        └───────┬───────────────┬──────────────┬───────┘
   └──────────────┘                │               │              │
                                   ▼               ▼              ▼
                          ┌────────────┐   ┌────────────┐  ┌──────────────┐
                          │ CPU math   │   │  OpenCL    │  │  QNN / NPU   │
                          │ (NEON/AVX, │   │   (GPU)    │  │  accelerator │
                          │  OpenBLAS) │   │            │  │              │
                          └────────────┘   └────────────┘  └──────────────┘
                                   │               │              │
                                   ▼               ▼              ▼
                          ┌───────────────────────────────────────────────┐
                          │   Target OS / HW: Ubuntu · Android NDK ·        │
                          │   Windows/MSVC · Tizen · ARM · x86_64           │
                          └───────────────────────────────────────────────┘
```

---

## Actors and external systems

| Actor / system | Interaction |
|---|---|
| **Application developers** | Build models via INI config, C++/C API, or JNI; run training/inference loops. |
| **Model files** | Import/export: INI model configs, ONNX, TFLite, FlatBuffer schema snapshots, safetensors. |
| **NNStreamer** | NNTrainer layers can be embedded in NNStreamer media pipelines (and vice-versa). |
| **Hardware backends** | CPU (NEON on ARM, AVX on x86, via OpenBLAS), GPU via OpenCL, NPU via QNN. |
| **Packaging / distros** | Debian (`debian/`), Tizen GBS (`packaging/`), Yocto — each is a delivery target with its own CI. |

---

## Supported targets (all first-class)

Ubuntu, Android NDK, Windows/MSVC, Tizen, ARM, x86_64. **No target is
secondary.** A change that compiles and passes tests on one platform but breaks
another is a regression, not a partial success. This single fact drives most of
the cross-platform rules in [`03-crosscutting/cross-platform.md`](03-crosscutting/cross-platform.md)
and the CI matrix in [`04-quality-system/ci-topology.md`](04-quality-system/ci-topology.md).

---

## Key external dependencies

| Dependency | Used for | Notes |
|---|---|---|
| **Meson** (1.7.2) | Build orchestration | Pinned; 1.8.0 has a Windows `/WX` regression. |
| **OpenBLAS** | CPU BLAS kernels | Vendored/sub-projected per platform. |
| **OpenCL** | GPU backend | Optional (`enable-opencl`); off on Windows native build. |
| **QNN SDK** | NPU backend | Optional; `fake-qnn-sdk/` provides a stub for builds without the real SDK. |
| **FlatBuffers** | Model serialization (`schema/`) | Persist/restore models. |
| **GoogleTest** | Unit tests | All of `test/unittest/`. |

---

## System-level invariants

- **INV-CTX-1 — On-device budget.** Memory and compute are constrained. Hidden
  allocations, copies, and unbounded buffers are architecturally significant, not
  micro-optimizations.
- **INV-CTX-2 — Multi-target parity.** Behavior must be equivalent across
  supported targets unless a difference is intrinsic to the hardware and
  documented.
- **INV-CTX-3 — Stable public surface.** C/C++/CAPI/JNI are consumed externally;
  breaking them is a high-impact, human-review-required event.
- **INV-CTX-4 — Reproducible serialization.** Models saved by one version/target
  must load correctly on others; schema changes are versioned.

---

## What changes at this layer (and triggers human architecture review)

- Adding/removing a supported target or a top-level external dependency.
- Adding a new public API surface (a new binding, a new file format).
- A new hardware backend family (beyond CPU/OpenCL/QNN).

Anything at this layer requires an [ADR](adr/) and high-level human sign-off.

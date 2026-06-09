# L4 — Cross-Platform Correctness

> **Cross-cutting, #1 regression risk.** All of Ubuntu, Android NDK,
> Windows/MSVC, Tizen, ARM, and x86_64 are first-class (INV-CTX-2). Most CI build
> jobs run on platforms we **cannot** execute locally, so cross-platform
> correctness is established mostly by code-level reasoning + the CI matrix.

## The rule

A change that compiles and passes on one platform but breaks another is a
**regression**, not a partial success. Prefer general fixes; use platform/compiler
branches only for genuinely necessary differences, and keep them narrow.

## What to watch (review + AI-bot checklist)

| Hazard | Why it bites | Guard |
|---|---|---|
| Integer width / signedness | `long` differs (LLP64 on Windows vs LP64 on Linux) | Use fixed-width / `size_t`; no assumptions. |
| Pointer size | 32-bit Android vs 64-bit | No pointer↔int casts assuming width. |
| File handles / descriptors | `HANDLE` vs `int fd` | Abstract behind utils; no raw assumptions. |
| Filesystem paths | `\` vs `/`, drive letters | No hardcoded separators or absolute paths. |
| Line endings | CRLF vs LF | Repo newline rules; no CRLF in sources. |
| Compiler extensions | MSVC vs GCC/Clang vs NDK | Stick to C++20 standard; narrow `#if`. |
| Host-only commands | shell assumptions in scripts | Portable scripts; shellcheck/pylint clean. |
| BLAS/NEON/AVX paths | arch-specific kernels | Keep behind `supports_*()` / arch guards. |
| JNI/NDK assumptions | Android-only ABI | Verify in `android.yml` / `causallm_android.yml`. |

## Local vs reasoned coverage

| Platform | Can we run it here? |
|---|---|
| Windows x64 MSVC | **Yes** — `.claude/nntr/nntr-build.ps1` (meson 1.7.2). |
| Windows arm / clang | No — reason + CI (`windows_arm_clang.yml`, `windows_cross_compile.yml`). |
| Ubuntu (gcc/clang) | No — reason + CI (`ubuntu_clean_meson_build*.yml`). |
| Android NDK | No — reason + CI (`android.yml`, `causallm_android.yml`). |
| Tizen / Yocto | No — reason + CI (`daily_build_tizen_*`, `gbs_build.yml`, `yocto.yml`). |

**Always state in a report which gates were executed locally vs reasoned about.**
This honesty is part of the trust model — see
[testing-strategy](../04-quality-system/testing-strategy.md).

## Meson notes

- Pin **meson 1.7.2** (1.8.0 has a Windows `/WX` regression).
- Native files: `windows-native.ini` (MSVC: `enable-opencl=false`,
  `default_library=static`, c++20, `vsenv=true`), `windows-native-clang.ini` (clang).
- New build options must default safely on every target.

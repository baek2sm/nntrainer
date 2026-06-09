# L4 — Error Handling & Logging

> **Cross-cutting.** How failures surface and how the library reports state.
> Sources: `nntrainer_error.h`, `nntrainer_log.h`, `nntrainer_logger.*`.

## Error model

- **Exceptions for programmer/contract errors.** Invalid config, out-of-order
  lifecycle, unsupported op → throw a `std::` exception with a clear message:
  - unimplemented backend op → `std::runtime_error("<op> not implemented…")` (INV-BCK-1)
  - cross-vendor op mismatch → `std::invalid_argument` pointing at `Tensor::to()` (INV-TEN-3)
- **Fail loud, fail early.** Validate at `finalize`/`compile` (INV-LAY-3, INV-CMP-3,
  INV-MOD-1), not silently mid-forward.
- **Error codes** at C-API boundaries map exceptions to status codes for CAPI/JNI
  consumers (public surface — INV-CTX-3).

## Invariants

- **INV-ERR-1 — Messages are actionable.** Include the op/layer/context name and
  what was expected. The dispatch errors above are the model to follow.
- **INV-ERR-2 — No silent fallback to wrong results.** Missing capability throws
  or falls back to a *correct* path (CPU), never to garbage.
- **INV-ERR-3 — Consistent logging.** Use the `nntrainer_log` macros/levels;
  don't `printf`/`std::cout` in library code.

## Logging

- Levels via `nntrainer_logger`; respect them, no debug spew at info level.
- Logging must be cheap on the hot path (on-device budgets, INV-CTX-1).

## Review focus

- New error path: is the message actionable and tested (negative test)?
- Any `cout`/`printf` slipping into library code → reject.
- Exception safety around tensor allocation (no leaks on throw).

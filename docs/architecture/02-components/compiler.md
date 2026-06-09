# L3 — Compiler (`nntrainer/compiler/`)

> **Layer 3 component.** Turns a model *description* into an executable graph.

## 1. Responsibility

Parse model descriptions (INI, ONNX, TFLite) via interpreters and apply
**realization passes** that transform/normalize the graph (e.g. flatten,
activation realization, multi-out, recurrent unrolling) before execution.

## 2. Internal structure

- **Interpreters:** INI interpreter, ONNX interpreter, TFLite interpreter — each
  produces a graph representation from its format.
- **Realizers:** ordered passes that rewrite the graph into a canonical,
  executable form.

```
description (INI/ONNX/TFLite) ──interpreter──▶ raw graph ──realizers (ordered)──▶ NetworkGraph
```

## 3. Contracts & invariants

- **INV-CMP-1 — Realizer order matters.** Passes run in a defined order; reordering
  is a behavior change requiring tests.
- **INV-CMP-2 — Format in, graph out.** Interpreters only translate; they do not
  execute or allocate device tensors.
- **INV-CMP-3 — Layer names resolve via factory.** Unknown keywords fail at
  compile with a clear error, not silently.

## 4. Dependencies

- **Uses:** `layers/` (instantiate by name), `graph/` (emit), `utils/` (INI), `schema/`.
- **Used by:** `models/` (`compile()`).

## 5. Change checklist

- [ ] New realizer placed correctly in pass order; documented.
- [ ] New format/op has interpreter coverage + a round-trip test.
- [ ] Error messages for malformed input are clear and tested (negative tests).

## 6. Review focus & common pitfalls

- Pass-ordering bugs producing subtly wrong graphs.
- Incomplete format coverage silently dropping ops.
- Divergence between INI and ONNX/TFLite paths for equivalent models.

## 7. Tests

- `test/unittest/compiler/` (7 files) — interpreter and realizer tests.

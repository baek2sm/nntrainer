# L3 — Component View (index)

> **Layer 3 (Components).** One document per subsystem. Each zooms into a box
> from [`../01-container-view.md`](../01-container-view.md) and describes its
> internal structure, key types, **contracts & invariants**, dependencies, and
> the review focus for changes in that area. These change with ordinary feature
> work — and must be updated in the same PR (see the
> [doc-maintenance contract](../04-quality-system/doc-maintenance-contract.md)).

## Documents

| Doc | Subsystem | Path |
|---|---|---|
| [tensor.md](tensor.md) | Tensor abstraction, ops, pools, quantization | `nntrainer/tensor/` |
| [backends.md](backends.md) | Dispatch backbone: Engine/Context/ContextData/ComputeOps | `nntrainer/{engine,context*,app_context,cl_context,qnn_context}.*` |
| [graph.md](graph.md) | NetworkGraph, execution order, context stamping | `nntrainer/graph/` |
| [layers.md](layers.md) | Layer implementations + LayerNode | `nntrainer/layers/` |
| [models.md](models.md) | NeuralNetwork lifecycle, train/infer loop | `nntrainer/models/` |
| [compiler.md](compiler.md) | Interpreters + realization passes | `nntrainer/compiler/` |
| [optimizers.md](optimizers.md) | Optimizers + LR schedulers | `nntrainer/optimizers/` |
| [dataset.md](dataset.md) | Data producers/loaders | `nntrainer/dataset/` |

## Per-document template

Every L3 doc follows the same skeleton so reviewers and the AI bot know where to
look:

1. **Responsibility** — one paragraph.
2. **Internal structure** — key files/types and how they relate.
3. **Contracts & invariants** — numbered `INV-<AREA>-N`, citable by tests/reviews.
4. **Dependencies** — what it uses, what uses it (direction matters — see INV-CONT-3).
5. **Change checklist** — what a contributor must do/verify when touching it.
6. **Review focus & common pitfalls** — where bugs and regressions cluster.
7. **Tests** — where the unit tests live and how this area should be tested.

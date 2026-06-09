# L4 — Cross-Cutting Concerns (index)

> **Layer 4.** Concerns that don't belong to a single subsystem but constrain all
> of them. A change touching one of these areas should be reviewed against the
> relevant document here, regardless of which subsystem the diff lives in.

| Doc | Concern |
|---|---|
| [dispatch-and-backends.md](dispatch-and-backends.md) | The Engine/Context/ComputeOps backbone and backend rules. |
| [cross-platform.md](cross-platform.md) | Multi-target correctness (Ubuntu/Android/Windows/Tizen/ARM/x86_64). |
| [memory-management.md](memory-management.md) | Tensor pools, planners, lifetime, on-device budgets. |
| [error-and-logging.md](error-and-logging.md) | Error model, exceptions, logging. |
| [concurrency.md](concurrency.md) | Threading, init ordering, shared-tensor safety. |

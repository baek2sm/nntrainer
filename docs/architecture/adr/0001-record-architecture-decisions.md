# 0001. Record architecture decisions

- **Status:** Accepted
- **Date:** 2026-06-02
- **Deciders:** NNTrainer maintainers
- **Affects:** all of `docs/architecture/`

## Context

NNTrainer is entering a phase of high-velocity, increasingly AI-assisted change
from both internal and external contributors. Decisions that shape the
architecture (the dispatch backbone, invariants, supported targets) are currently
recorded only implicitly in code and scattered docs. Future contributors — human
and AI — repeatedly re-litigate settled decisions because the rationale isn't
written down.

## Decision

We will record architecturally significant decisions as ADRs in
`docs/architecture/adr/`, using a lightweight Markdown template, numbered
sequentially and append-only. An ADR is required for the triggers listed in the
[ADR README](README.md).

## Consequences

- Positive: settled decisions have a citable rationale; reviews reference ADRs
  instead of re-arguing; the architecture docs gain a "why" layer.
- Cost: a small authoring step for significant changes (intentionally scoped to
  significant ones only).
- Follow-up: link ADRs from the architecture docs they justify.

## Alternatives considered

- **Rationale inline in docs only** — gets overwritten; loses history of *why a
  thing changed*. ADRs are append-only by design.
- **Wiki / external tracker** — drifts from the repo; not reviewable in PRs.

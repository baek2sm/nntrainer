# Architecture Decision Records (ADRs)

> An ADR captures **one architecturally significant decision**: the context, the
> decision, and its consequences. ADRs are the *why* behind the structure
> documented in the rest of this tree. They are append-only history — supersede,
> don't rewrite.

## When an ADR is required

- Any change to a numbered invariant (`INV-*`).
- Any L1/L2 change (system context, subsystem map, dispatch backbone).
- A new backend, external dependency, public API surface, or build option.
- Any decision a future reader would otherwise ask "why is it like this?" about.

These are exactly the [human-review escalation triggers](../04-quality-system/human-review-playbook.md#4-escalation-triggers).

## Process

1. Copy [`template.md`](template.md) to `NNNN-short-title.md` (next number).
2. Status starts `Proposed`; set `Accepted` when merged, `Superseded by NNNN`
   when replaced.
3. Reference the ADR from the architecture doc it affects.

## Index

| # | Title | Status |
|---|---|---|
| [0001](0001-record-architecture-decisions.md) | Record architecture decisions | Accepted |
| [0002](0002-layered-architecture-docs-and-trust-pyramid.md) | Layered architecture docs + trust pyramid | Proposed |

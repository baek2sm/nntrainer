# 0002. Layered architecture docs + trust pyramid

- **Status:** Proposed
- **Date:** 2026-06-02
- **Deciders:** NNTrainer maintainers (pending review)
- **Affects:** `docs/architecture/` (new tree), `04-quality-system/*`, CI, AI bots

## Context

Code units are getting larger and change is accelerating, much of it AI-assisted.
Line-by-line human review does not scale and is not where humans add the most
value. We need a way to keep low-level correctness trustworthy *without* a human
reading every line, while keeping humans firmly in control of architectural
direction.

The repo already has the ingredients: a clean dispatch architecture, extensive
GoogleTest unit tests, a broad CI matrix, and a working GitHub Models workflow
(`pr-desc`) with a per-module knowledge base.

## Decision

We will adopt a **trust pyramid**:

- **TDD + CI** establishes low-level correctness and cross-platform portability.
- **An AI review bot** (reusing the `pr-desc` machinery) checks conventions,
  scope, cross-platform rules, regression risk, and doc-sync on every PR.
- **Human reviewers** focus on architecture deltas, using a new layered
  `docs/architecture/` tree (L1 context → L2 containers → L3 components → L4
  cross-cutting) as the baseline.

We will enforce a **documentation maintenance contract**: architecture-altering
PRs update the matching doc in the same PR, enforced by a CI doc-sync gate, the AI
bot, and human reviewers. The architecture docs become the single source of truth
that also feeds the AI bots' context.

Invariants are numbered (`INV-<AREA>-N`) so tests and reviews can cite them.

## Consequences

- Positive: human attention concentrates on high-leverage architectural judgment;
  low-level correctness is trusted via tests + bot + gates; the map stays accurate
  because updating it is contractually part of changing the architecture; one doc
  edit improves both human and AI review.
- Cost: process additions (doc-sync gate, AI-review workflow, ADR habit) and the
  discipline to keep docs and code in sync; initial authoring of the doc tree.
- Follow-up: implement the roadmap in
  [`04-quality-system/README.md`](../04-quality-system/README.md) §"roadmap" as
  independent PRs; validate the doc-sync heuristic's false-positive rate before
  making it a hard gate.

## Alternatives considered

- **Status quo (heavier human review)** — does not scale with volume/PR size.
- **AI review only, no architecture docs** — bot lacks a stable baseline; humans
  lose the high-level map; no shared source of truth.
- **Architecture docs without the maintenance contract** — docs drift, become
  misleading, and erode trust faster than having none.

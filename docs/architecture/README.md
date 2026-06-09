# NNTrainer Architecture Documentation

> **Status: DRAFT for review.** This tree is a self-contained architecture map.
> It does not replace the existing implementation docs in `docs/backend_guide/`
> or `docs/components.md`. Those remain the deep references. This tree sits
> above them and is the review contract: the artifact human reviewers read first
> and the artifact every non-trivial PR must keep in sync.

---

## 1. Why this tree exists

NNTrainer changes quickly, and much of that change is AI-assisted. At that
rate, line-by-line human review is not where the highest value is. The useful
question is whether the shape of the system stayed correct.

So we split trust across three tiers:

```text
HUMAN REVIEW      architecture and intent
AI REVIEW BOT     conventions, scope, cross-platform, doc sync
TDD + CI          correctness, portability, build matrix
```

- **Base: TDD + CI.** Low-level correctness is established by tests and build
  checks.
- **Middle: AI review bot.** Checks project conventions, cross-platform rules,
  scope discipline, regression risk, and whether the architecture docs were
  updated.
- **Top: Human review.** Reviews architecture deltas: new dependencies, new
  layers, changed contracts, and changed invariants.

The pyramid only holds if the map stays accurate. That is enforced by the
documentation maintenance contract in
[`04-quality-system/doc-maintenance-contract.md`](04-quality-system/doc-maintenance-contract.md).

---

## 2. How the documents are layered

Read top-down. Each layer zooms in one level deeper.

| Layer | Document | Scope | Audience / change cadence |
|---|---|---|---|
| **L1 Context** | [`00-system-context.md`](00-system-context.md) | What NNTrainer is, who uses it, external dependencies, supported targets | Everyone. Changes rarely. |
| **L2 Containers** | [`01-container-view.md`](01-container-view.md) | The major subsystems, their responsibilities, and the dispatch backbone | Reviewers, new contributors. Changes on subsystem-level work. |
| **L3 Components** | [`02-components/`](02-components/) | Per-subsystem internal structure, key types, contracts, invariants | Feature authors in that area. Changes with feature work. |
| **L4 Cross-cutting** | [`03-crosscutting/`](03-crosscutting/) | Concerns that span subsystems: dispatch, cross-platform, memory, errors, concurrency | Anyone touching the relevant concern. |
| **L5 Repository map** | [`05-repository-map.md`](05-repository-map.md) | A project-wide guide to the root folders, application surfaces, and build entry points | New contributors, agents, reviewers. |
| **Quality system** | [`04-quality-system/`](04-quality-system/) | The TDD + CI + AI-review + human-review process that makes high-velocity change safe | Maintainers, CI owners. |
| **Decisions** | [`adr/`](adr/) | Architecture Decision Records for why the structure is the way it is | Everyone, when questioning a constraint. |

---

## 3. Relationship to existing docs

| This tree | Existing doc | Relationship |
|---|---|---|
| `01-container-view.md`, `03-crosscutting/dispatch-and-backends.md` | `docs/backend_guide/ARCHITECTURE.md` | We summarize the dispatch chain at L2/L4 and link down to the backend guide for the full contract. |
| `02-components/*.md` | `docs/components.md`, `.github/models/pr-desc/context/modules/*.md` | We give an architectural view: contracts, invariants, dependencies. `components.md` stays the feature catalog. |
| `05-repository-map.md` | repository root folders | We give a project-wide map that tells agents where to look first before opening code. |
| `04-quality-system/*` | `AGENTS.md`, `.github/workflows/`, `review-for-nntr` | We design the target system. The workflows and `AGENTS.md` are the current implementation; the quality-system docs describe both current state and the enhancement plan. |

---

## 4. How to use this tree

- New contributor: read L1 and L2, then the L3 doc for the area you will touch.
- Reviewer: read the PR's architecture-doc diff first, then use the human
  review playbook.
- Feature author: before coding, find the L3 or L4 doc you'll affect; after
  coding, update it in the same PR.
- AI review bot: these documents are part of its prompt context, so keeping
  them accurate directly improves review quality.

---

## 5. Document conventions

- Each L2, L3, and L4 doc starts with a Responsibility section, then Contracts
  and invariants, Dependencies, Change checklist, and Review focus.
- Invariants are numbered as `INV-<AREA>-N` so reviews and tests can cite them.
- Diagrams are ASCII so they diff cleanly in PRs.
- When code and a doc disagree, the code wins. The disagreement is a bug to fix
  in the doc.

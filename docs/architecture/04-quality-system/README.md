# Quality System — Overview

> **The system that makes high-velocity, AI-assisted change safe.** As department
> members and external developers land features quickly (much of it AI-assisted),
> we cannot rely on line-by-line human review. Instead we distribute trust across
> three tiers and keep an accurate architecture map as the human-review baseline.

## The trust pyramid (recap)

```
   HUMAN REVIEW    →  architecture & intent     (this docs tree is the map)
   AI REVIEW BOT   →  conventions, scope,        (deterministic, every PR)
                      cross-platform, doc-sync,
                      regression risk
   TDD + CI        →  correctness + portability  (tests + build matrix, every PR)
```

Each tier catches what the tier below can't economically catch:

- **TDD + CI** proves the code does what a test says, on every platform.
- **AI bot** catches convention/scope/cross-platform/regression issues that pass
  the build but a human shouldn't have to hunt for — and enforces doc-sync.
- **Human** judges whether the *shape* of the system changed correctly, using the
  architecture docs as the baseline. Low-level correctness is already trusted.

## The five documents

| Doc | What it defines |
|---|---|
| [testing-strategy.md](testing-strategy.md) | TDD conventions, the local unit-test enhancement plan, test taxonomy, what "tested" means per layer. |
| [ci-topology.md](ci-topology.md) | The CI matrix, what each gate proves, and the CI enhancement plan (cross-platform coverage + required checks). |
| [ai-review-bot.md](ai-review-bot.md) | Design of the AI review bot — reusing the existing GitHub Models `pr-desc` pattern — its prompt, context, checklist, and output. |
| [human-review-playbook.md](human-review-playbook.md) | What human reviewers do (architecture-delta review) and explicitly do **not** need to do. |
| [doc-maintenance-contract.md](doc-maintenance-contract.md) | The rule that keeps the map accurate: architecture changes update the matching doc in the same PR; bot + CI enforce it. |

## Implementation roadmap (draft — to refine with the team)

This tree is documentation only. Turning it into a running system is a sequence
of small, independent PRs (each a good candidate for the nntr-orchestrator
pipeline):

1. **Adopt the docs.** Land this `docs/architecture/` tree; link it from
   `CONTRIBUTING.md` and `AGENTS.md`.
2. **Doc-sync gate (cheap, high-leverage).** A CI check + AI-bot rule: PRs that
   touch flagged architecture paths must also touch `docs/architecture/`.
   See [doc-maintenance-contract](doc-maintenance-contract.md) §Enforcement.
3. **AI review bot.** Add `.github/workflows/ai-review.yml` + `.github/models/ai-review/`
   mirroring the existing `pr-desc` setup. See [ai-review-bot](ai-review-bot.md).
4. **Test-coverage expectations.** Document + (later) enforce "new behavior ships
   with a unit test" via coverage deltas. See [testing-strategy](testing-strategy.md).
5. **CI enhancement.** Make the cross-platform gates required, add fast-feedback
   ordering, and a label to run the full matrix. See [ci-topology](ci-topology.md).
6. **ADR habit.** Require an ADR for L1/L2 changes. See [adr/](../adr/).

Nothing here changes runtime code; it changes process + repo metadata. That keeps
each step low-risk and reviewable on its own.

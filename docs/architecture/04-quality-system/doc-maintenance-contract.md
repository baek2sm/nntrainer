# Documentation Maintenance Contract

> **The keystone.** The whole trust pyramid rests on the architecture map being
> accurate. A stale map makes human review *worse than nothing* — reviewers trust
> a description that no longer matches the code. This contract keeps the map true.

---

## 1. The rule

> **If a change alters the architecture, it updates the matching
> `docs/architecture/` document in the same PR.**

"Alters the architecture" means any of:

- A new/removed/moved subsystem or a new dependency edge (→ `01-container-view.md`,
  affected `02-components/*`).
- A change to a numbered invariant `INV-*` (→ the doc that owns it + likely an ADR).
- A new backend, dispatch tier, or change to where ContextData is stamped
  (→ `02-components/backends.md`, `03-crosscutting/dispatch-and-backends.md`).
- A new public API surface or format (→ `00-system-context.md`).
- A new cross-cutting concern or a change to one (→ `03-crosscutting/*`).
- A change to the quality process itself (→ `04-quality-system/*`).

Pure low-level changes (a kernel optimization that preserves all contracts, a new
test, a bug fix that restores documented behavior) do **not** need a doc update —
they're covered by tests, the bot, and gates.

---

## 2. Why "same PR" (not "later")

- A follow-up doc PR never happens at high velocity; drift accumulates silently.
- The doc diff *is* the high-level review artifact — humans read it first
  ([human-review-playbook](human-review-playbook.md) §3). No doc diff = no
  high-level summary to review.
- Keeping them together makes the architecture change reviewable as one unit:
  intent (doc) + implementation (code) side by side.

---

## 3. The single-source-of-truth loop

```
docs/architecture/02-components/*.md   ─┐
docs/architecture/03-crosscutting/*.md ─┤ canonical descriptions
                                         │
        (generate / sync-check)          ▼
.github/models/*/context/modules/*.md ──▶ AI bot context (pr-desc + ai-review)
                                         │
                                         ▼
            better PR descriptions + better AI reviews
                                         │
                                         ▼
            humans review the accurate doc delta
```

Updating an architecture doc therefore improves **both** human review (accurate
baseline) **and** AI review (better context) — one edit, both tiers benefit. The
sync between the canonical docs and the bot's `context/` specs is enforced by a
script or a CI drift-check (see [ai-review-bot](ai-review-bot.md) §5).

---

## 4. Enforcement

| Mechanism | What it does |
|---|---|
| **CI doc-sync gate** | PR touches architecture-significant paths but not `docs/architecture/` → fail with a pointer to this contract. Override: `skip-arch-doc` label + justification. ([ci-topology](ci-topology.md) §3.3) |
| **AI review bot** | Reports a doc-sync finding in every review ([ai-review-bot](ai-review-bot.md) §3.5). |
| **Human reviewer** | Reads the doc diff first; asks for it if missing ([human-review-playbook](human-review-playbook.md)). |
| **`code wins` rule** | When doc and code disagree, the code is right and the doc is a bug to fix — file it, don't trust the doc. |

The CI gate is intentionally a *path-based heuristic* (cheap, deterministic), not
a judgment of whether the doc content is correct — that judgment is the human's
and the bot's. The heuristic just guarantees the conversation happens.

---

## 5. Keeping invariants honest

- Invariants are numbered (`INV-<AREA>-N`) so tests and reviews can cite them.
- A test that asserts an invariant should reference its number in a comment, so
  changing the invariant surfaces the test that must change too.
- Removing/altering an invariant requires an [ADR](../adr/) recording why.

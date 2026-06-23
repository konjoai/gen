# Four-Gate Paper Screen: `{{CANDIDATE}}`

**Date:** {{DATE}}
**Author:** _(you)_
**Target constraint (charter §1):** _(which invariant slot does this attack, and why is it under-served?)_

> The screen is an analytic argument written *before any code*. Kill on paper if
> Gate 1 (Expressivity) or Gate 3 (Hardware) fails. A candidate that passes all
> four on paper earns a kill test (MQAR against the baselines). See
> `docs/CHARTER.md` §4.

---

## Graveyard pass (charter §4 step 2)

_Prior art for this primitive. Has it been tried? If it lost, to what and why?
Either fix that exact reason here or move on._

---

## Gate 1 — Expressivity

**Question:** Can it do exact retrieval / associative recall *in principle*? Could
it copy a token from far back?

**Verdict:** `pass` | `risk` | `fail`

**Argument:**

_(Analytic. e.g. state capacity vs. number of associations; an explicit
construction that performs recall, or an impossibility argument.)_

---

## Gate 2 — Trainability

**Question:** Do gradients survive at depth and length?

**Verdict:** `pass` | `risk` | `fail`

**Argument:**

_(e.g. subgradient behavior, vanishing/exploding analysis, dependence on
specialized init or normalization.)_

---

## Gate 3 — Hardware

**Question:** Does it reduce to dense matmul, a parallel scan, or an FFT at high
arithmetic intensity? (Scattered gather / dynamic control flow / per-element
branching = dead at scale.)

**Verdict:** `pass` | `risk` | `fail`

**Argument:**

_(Map the core operation to a known hardware-aligned primitive, or show why it
cannot. This is where most exotica die.)_

---

## Gate 4 — Scaling

**Question:** Does quality improve predictably as compute is added?

**Verdict:** `pass` | `risk` | `fail`

**Argument:**

_(Is there a clean knob converting compute → capability? Any reason to expect a
ceiling?)_

---

## Decision

**Paper verdict:** `paper-dead` (fails Gate 1 or Gate 3) | `earned-kill-test`

**If earned-kill-test — the kill test:** run `{{CANDIDATE}}` on MQAR across the K
range against the `attention` and `ssm` baselines (see `docs/gauntlet_baselines.md`).
Falsification criterion: _(state the bar it must clear, e.g. "match attention to
within 5% accuracy up to K=32, or it cannot be the sole mixer")_.

**One-line reason for the ledger:**

_(Copied into `decision_log.jsonl`.)_

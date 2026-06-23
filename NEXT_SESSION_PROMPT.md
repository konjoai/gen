# Next Session — Second Candidate Screen: crack the both-axes corner

**Type:** Candidate-screen sprint (the second; scaffold v0.1.0 + the tropical screen are done).
**Depends on:** `docs/gauntlet_baselines.md` (the two reference baselines) and
`docs/screens/tropical_max_plus.md` (the first screen, `killed-empirically`).

## Where we are (read first)

Three points now sit on the two-axis map (recall = MQAR-vs-K; state-tracking = parity / Dyck):

| mixer | recall (MQAR) | state-tracking (parity/Dyck) |
| --- | --- | --- |
| `attention` | aces to K=32 | ~chance |
| `ssm` (gated linear attn) | collapses by K=8 | aces |
| `tropical` (max,+) | aces to K=16, **collapses at K=32** | ~chance (like attention) |

**The corner is still empty.** Tropical was the recall-axis warm-up: it confirmed
hard (max,+) selection matches soft attention on recall up to moderate K (so the
softmax average is not load-bearing there), but its per-channel argmax loses
channel-consistency as keys compete, and the sharpness needed to fix that collides
with its Gate-2/Gate-3 numerical fragility. It bought no efficiency, so it was
`killed-empirically`. Nobody yet holds recall AND state-tracking together.

## The one-line kickoff

> Read `docs/CHARTER.md` (§3.1, §4), `docs/gauntlet_baselines.md`, and
> `docs/DECISION_LOG.md`. Paper-screen a **state-expansion recurrence**
> (delta-rule / fast-weight / test-time-regression flavored, charter §3.1) against
> the four gates. If it earns a kill test, implement it slow-but-correct and judge
> it on **both axes** against all three existing rows.

## Why state-expansion now (charter §3.1)

It attacks the corner directly: give the recurrence a larger, content-addressable,
**online-updated** state (a delta rule / fast-weight outer-product memory that
*writes corrections*, not just sums), so it can both (a) hold more associations
than the fixed-state `ssm` (move recall up at K>=8) and (b) keep the sequential
state-tracking the recurrence already has (parity / Dyck). The honest kill-gates
are **Gate 2** (does the online update train stably at depth) and **Gate 4** (does
the larger state actually lift the recall ceiling, or just shift it).

## Pre-registered falsification bar (write before running)

> Hold MQAR **materially above `ssm`** at K>=8 (e.g. >=0.6 where `ssm` is ~0.27)
> **while** keeping parity/Dyck **above the `attention` baseline** (>0.9, where
> attention is ~0.55). Clearing **both** is the corner and a publishable result.
> Clearing neither, or only one, is a clean negative — name which axis moved.

## Procedure (unchanged ritual)

1. Branch `claude/gen-screen-state-expansion` from `main`.
2. **Graveyard pass** (DeltaNet, Gated DeltaNet, TTT, Titans, fast-weight
   programmers — what did they hold/lose, and why) at the top of the screen.
3. `python scripts/paper_screen.py state_expansion`, fill the four gates, apply the
   kill-on-paper rule (Gate 1 or Gate 3 `fail` -> `paper-dead`, ledger + stop).
4. Only `earned-kill-test` proceeds: `src/gen/primitives/state_expansion.py`
   (`SequenceMixer`, slow/correct, real `step` recurrence this time, filled
   `gate_card`), register `"state_expansion"`, mirror `tests/test_interface.py`.
5. `python scripts/run_gauntlet.py state_expansion`; read **both axes**.
6. 30-run paired Wilcoxon (`gen.stats.wilcoxon`, p<0.05) for any "beats `ssm` on
   recall" or "matches `attention` on state-tracking" claim. Ledger
   (`killed-empirically` / `promoted`) honest against the pre-registered bar;
   update `CHANGELOG.md` and this file (to OT-as-core if the corner is still open).

## Guardrails carried over

- Slow-but-correct only; no fused kernel. Determinism + version header in every report.
- Fail loud; shape assertions at the mixer boundary. Watch for NaNs from the
  online update (the tropical sprint learned this the hard way: bound the state /
  normalize, do not let winner-take-all dynamics blow up the scores).
- Blocking CI stays blocking. One candidate only; do not start OT-as-core.

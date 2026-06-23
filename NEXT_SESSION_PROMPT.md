# Next Session — First Candidate Screen: the lead under-served constraint

**Type:** Candidate-screen sprint (the first real one; the scaffold is done).
**Depends on:** this repo's V1 baselines (`docs/gauntlet_baselines.md`) — the
reference rows your candidate is read against.

## The one-line kickoff

> Read `docs/CHARTER.md` (§3.2 negative space, §4 method) and
> `docs/gauntlet_baselines.md`. Paper-screen the lead candidate — **the tropical
> max-plus mixer** (or whichever under-served constraint I choose at kickoff) —
> against the four gates. If it earns a kill test, implement it in slow-but-correct
> form and run it on MQAR against the `attention` and `ssm` baselines.

## Why tropical first (charter §3.2)

Sparse selection is the dominant theme of the whole sub-quadratic field, and
max-plus algebra makes selection the *primitive* operation rather than a bolt-on.
The honest kill-gates are **Gate 2** (max is subgradient-only) and **Gate 3**
(max-reductions are bandwidth-bound but matmul-adjacent — feasible). OT-as-core
and NCA-as-causal-mixer are the next two in the queue.

## Procedure (do these in order)

1. **Graveyard pass.** Write the prior art for max-plus / tropical sequence
   mixing. Has the exact formulation been tried? If it lost, to what and why?
2. **Scaffold the screen:**
   ```bash
   python scripts/paper_screen.py tropical_max_plus
   ```
   Fill in `docs/screens/tropical_max_plus.md` — the analytic argument for each
   of the four gates. **Kill on paper if Gate 1 or Gate 3 fails.**
3. **Record the paper verdict** in the ledger:
   - `paper-dead` → append the entry with the reason; stop. (A clean negative is
     a deliverable — charter §4 step 8.)
   - `earned-kill-test` → continue.
4. **Implement the kill test.** Add `src/gen/primitives/tropical.py` implementing
   `SequenceMixer` (slow, correct, no kernel), fill its `gate_card`, register it
   in `REGISTRY`, then:
   ```bash
   python scripts/run_gauntlet.py tropical
   ```
5. **Decide against the baselines.** Compare the MQAR accuracy-vs-K curve to the
   baselines. Use the 30-run paired Wilcoxon gate (`gen.stats.wilcoxon`) for any
   "matches/beats attention" claim before it merges. Append the empirical verdict
   (`killed-empirically` or `promoted`) to the ledger.

## Falsification criterion to set up front

Before running: state the bar tropical must clear, e.g. *"match attention to
within 5% MQAR accuracy through K=32, while keeping a Gate-3-plausible reduction
to a parallel scan; otherwise it cannot be the sole mixer."* If it cannot, that
is the result — write the note and move to the next candidate.

## Guardrails carried over

- No language data, no kernels, no training framework. Slow-but-correct only.
- Determinism + version header in every report.
- Fail loud; shape assertions at the mixer boundary.
- Blocking CI stays blocking.

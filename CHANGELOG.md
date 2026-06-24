# Changelog

All notable changes to this project are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); this project uses semver.

## [0.3.0] — 2026-06-24

Second candidate screen: the delta-rule state-expansion recurrence. Verdict
`promoted` — the first mixer to occupy the recall-and-state-tracking corner.

### Added
- `docs/screens/state_expansion.md` — four-gate paper screen (graveyard pass +
  per-gate analysis; `earned-kill-test`) plus the empirical kill-test result and
  the 30-run Wilcoxon backing.
- `src/gen/primitives/state_expansion.py` — `DeltaNetMixer`: a DeltaNet-style
  delta-rule recurrence with an error-correcting outer-product write, unit-norm
  keys, and `beta = 2*sigmoid(.)` in `(0,2)` (the negative-eigenvalue lever for
  state-tracking). Real `step` recurrence; slow-but-correct sequential scan.
  Registered as `"state_expansion"`.
- `tests/test_state_expansion.py` — contract, causality, `step`==`forward`, a
  delta-reference correctness check, and the error-correcting overwrite property.
- `docs/gauntlet_state_expansion.md` — both-axes capability profile (seed 0).
- Decision-ledger row (`promoted`).

### Result — the corner
- **Recall:** matches `attention` through K=16 (1.000) and is far above `ssm`
  (0.27/0.19) — 30-run paired Wilcoxon at K=8 gives p=4.6e-5 vs `ssm`.
- **State-tracking:** parity and Dyck = 1.000, far above `attention` (0.57/0.54) —
  Wilcoxon at parity gives p=1.8e-6 vs `attention`. First non-specialist row.
- **Gate-4 scaling:** the K=32 recall ceiling at the default `d_key=32` (0.151) is a
  capacity knob, not a wall — `d_key=64` recovers K=32 to 0.991.
- Next sprint (`NEXT_SESSION_PROMPT.md`): the efficient chunked-parallel kernel
  (Gate-3 follow-through), Gated DeltaNet, and a width/length scaling study.

## [0.2.0] — 2026-06-23

First candidate screen: the tropical (max, +) mixer. Verdict `killed-empirically`.

### Added
- `docs/screens/tropical_max_plus.md` — four-gate paper screen (graveyard pass +
  per-gate analysis; paper verdict `earned-kill-test`) plus the empirical kill-test
  result and the 30-run Wilcoxon backing.
- `src/gen/primitives/tropical.py` — `TropicalMaxPlus`, a smoothed (max, +) mixer
  (QK-normalized compatibility, learnable per-head temperature, logsumexp
  relaxation in the matmul form). Registered as `"tropical"`. No recurrent form.
- `tests/test_tropical.py` — interface contract, causality, and two correctness
  checks (matmul form == logsumexp definition; behavior under sharper temperature).
- `docs/gauntlet_tropical.md` — both-axes capability profile (seed 0, env header).
- Decision-ledger row for `tropical` (gate card + both-axes summary + stats).

### Result
- **Recall axis:** tropical matches `attention` on MQAR through K=16 — at K=8 a
  30-run paired Wilcoxon finds no significant difference (p=0.86) while tropical
  significantly beats `ssm` (p=3e-6) — but recall **collapses to chance at K=32**
  ([0.151, 0.153, 0.149] across seeds vs attention's 1.000). The softmax average is
  not load-bearing for recall at low/moderate K; hard (max, +) selection suffices.
- **State-tracking axis:** unmoved (parity 0.64 / Dyck 0.55, ~attention's chance),
  as pre-registered — a stateless selector does not enter the corner.
- Fails the pre-registered through-K=32 bar and offers no efficiency win, so it is
  `killed-empirically`. The both-axes corner remains open; next candidate is a
  state-expansion recurrence (see `NEXT_SESSION_PROMPT.md`).

## [0.1.0] — 2026-06-23

Initial scaffold sprint: the reusable foundation for screening sequence-mixing
primitives against the Konjo Architecture Charter. No novel primitives.

### Added
- `docs/CHARTER.md` — the Konjo Architecture Charter (source of truth).
- `SequenceMixer` plug interface (`gen.primitives.base`) encoding the four-gate
  filter (Expressivity / Trainability / Hardware / Scaling) as a `GateCard`,
  with `forward` (causal), optional `step`, required `flops`, and boundary shape
  assertions.
- Reference baselines: `AttentionRef` (causal multi-head softmax attention, the
  known-capable control) and `SSMRef` (gated linear-attention recurrence, the
  known recall-limited control with a true `step` path).
- Tiny harness model `embed -> [mixer block × L] -> norm -> head`.
- Synthetic gauntlet: induction, MQAR (accuracy-vs-K curve), selective-copy, and
  formal languages (Dyck / parity / modular addition), with a deterministic
  train+eval harness and a report/verdict renderer.
- `gen.stats.wilcoxon` — self-contained 30-run paired Wilcoxon signed-rank gate
  (exact for small n, normal approximation with tie + continuity correction).
- `gen.ledger` — append-only JSONL four-gate decision log with a generated
  `DECISION_LOG.md` view.
- Scripts: `run_gauntlet.py`, `paper_screen.py`, `verify_deps.py`.
- Blocking CI (lint, format, type check, tests, fast gauntlet sanity) and
  pre-commit hooks (ruff, verify_deps). No `continue-on-error`.
- Tests for the interface, gauntlet scorers, the Wilcoxon gate, and the ledger.
- `docs/gauntlet_baselines.md` — V1 capability profiles for both baselines with
  the MQAR accuracy-vs-K curves and the environment/version header.

# Changelog

All notable changes to this project are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); this project uses semver.

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

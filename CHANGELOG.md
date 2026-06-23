# Changelog

All notable changes to this project are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); this project uses semver.

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

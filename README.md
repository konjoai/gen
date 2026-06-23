# gen (源)

> *gen* — 源, Japanese for **source / origin**. The package import name is `gen`;
> the PyPI distribution is `konjo-gen` (the name `gen` was already taken on PyPI).

A reusable scaffold for screening sequence-mixing primitives against the
**Konjo Architecture Charter** (`docs/CHARTER.md`). It is the cheap, local
gauntlet that every future "screen a new mixer" sprint drops a candidate into —
so each candidate is a drop-in, not a rebuild.

This repo implements **no novel primitive**. It provides the foundation:

- the **`SequenceMixer`** plug interface, which encodes the charter's four-gate
  filter as first-class structure (`src/gen/primitives/base.py`);
- two **reference baselines** — plain causal softmax attention (the known-capable
  control) and a minimal selective linear recurrence (the known recall-limited
  control);
- the **synthetic gauntlet** — induction, MQAR, selective-copy, and formal
  languages (Dyck / parity / modular addition) — that discriminates a capable
  mixer from an incapable one *in minutes on CPU/MPS*;
- the **30-run paired Wilcoxon** house gate (`src/gen/stats/wilcoxon.py`);
- the **append-only four-gate decision ledger** plus a paper-screen scaffolder.

## Why this exists (charter in one paragraph)

Anything nameable has been tried; what remains untried usually fails a hard
constraint. So re-derive a sequence model from the ten invariant constraints
(charter §1) and let novelty be the residue of rigor. Before any code, a
candidate is paper-screened on four gates — **Expressivity, Trainability,
Hardware, Scaling** — and most die at Gate 3 (hardware). A candidate that passes
on paper earns a **kill test**: MQAR against these baselines. *A primitive that
cannot pass MQAR will not model language.* Negative results are the deliverable.

## Quickstart

```bash
# install (CPU torch)
pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"

# checks
python scripts/verify_deps.py        # dependency provenance gate
pytest -q                            # interface / gauntlet / wilcoxon / ledger

# run the gauntlet on a baseline (prints a capability profile + verdict)
python scripts/run_gauntlet.py attention
python scripts/run_gauntlet.py ssm

# reproduce the V1 baseline report + seed the decision ledger
python scripts/run_gauntlet.py --baselines       # writes docs/gauntlet_baselines.md

# scaffold a four-gate paper screen for a future candidate
python scripts/paper_screen.py tropical_max_plus  # -> docs/screens/tropical_max_plus.md
```

The reference baselines establish the separation the harness must reproduce:
attention **aces MQAR across K**; the recurrence shows the **known recall gap**
(degrades as the key count K grows) while still passing induction and
selective-copy. See `docs/gauntlet_baselines.md`.

## Adding a candidate mixer (future sprints)

1. Implement `SequenceMixer` in `src/gen/primitives/<name>.py`: `forward`
   (causal, shape-asserted), optionally `step`, `flops`, and a filled-in
   `gate_card` (your paper-screen verdict on each of the four gates).
2. Register it in `src/gen/primitives/__init__.py:REGISTRY`.
3. `python scripts/run_gauntlet.py <name>` and read the MQAR curve against the
   baselines. Record the outcome in the ledger.

## Layout

```
src/gen/primitives/   SequenceMixer interface + attention/ssm baselines
src/gen/gauntlet/     harness, report, and the four task families
src/gen/stats/        30-run paired Wilcoxon gate
src/gen/ledger/       append-only JSONL decision log
scripts/              run_gauntlet, paper_screen, verify_deps
docs/                 CHARTER, four-gate template, decision log, baseline report
```

## Scope

Foundation only. No novel primitives, no training framework, no language data,
no fast kernels. The first real candidate (e.g. a tropical max-plus mixer) is a
separate sprint — see `NEXT_SESSION_PROMPT.md`.

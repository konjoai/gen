# Decision Log

Generated view of `decision_log.jsonl` (the append-only source of truth).
Do not edit by hand — regenerate with `scripts/run_gauntlet.py` / ledger tooling.

| candidate | date | verdict | gates (E/T/H/S) | reason | screen |
| --- | --- | --- | --- | --- | --- |
| `attention` | 2026-06-23 | promoted | P/P/~/P | reference baseline; gauntlet verdict: aces-mqar (recall holds across K) | — |
| `ssm` | 2026-06-23 | promoted | ~/P/P/~ | reference baseline; gauntlet verdict: recall-limited (MQAR degrades as K grows) | — |
| `tropical` | 2026-06-23 | killed-empirically | P/~/~/~ | matches attention on recall through K<=16 (K=8 Wilcoxon p=0.86, n.s.; significantly beats ssm p=3e-6) but recall collapses to chance at K=32 (per-channel argmax loses channel-consistency; sharper selection needs higher beta which hits the Gate-2/3 numerical fragility) and buys no efficiency; state-tracking unmoved (parity 0.64 / dyck 0.55). Fails the through-K=32 bar. | [doc](docs/screens/tropical_max_plus.md) |

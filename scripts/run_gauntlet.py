#!/usr/bin/env python3
"""Run a named mixer across the gauntlet and emit a report.

Examples:
    python scripts/run_gauntlet.py attention            # one mixer, print report
    python scripts/run_gauntlet.py --baselines          # both baselines -> V1 report + ledger
    python scripts/run_gauntlet.py ssm --quick          # fast sanity run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a plain script without installation.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gen import env_header  # noqa: E402
from gen.gauntlet.report import GauntletReport, render_markdown, run_gauntlet  # noqa: E402
from gen.ledger import append_entry, make_entry, regenerate_markdown_view  # noqa: E402
from gen.primitives import REGISTRY, get_mixer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LEDGER_JSONL = ROOT / "src" / "gen" / "ledger" / "decision_log.jsonl"
LEDGER_MD = ROOT / "docs" / "DECISION_LOG.md"
BASELINE_REPORT = ROOT / "docs" / "gauntlet_baselines.md"


def _gauntlet_summary(report: GauntletReport) -> dict[str, object]:
    return {
        "verdict": report.verdict(),
        "mqar_curve": [[k, round(a, 4)] for k, a in report.mqar_curve()],
        "capability": {k: round(v, 4) for k, v in report.capability_summary().items()},
    }


def _ledger_verdict(report: GauntletReport) -> str:
    # Baselines are references, not promotions: attention is the capable control,
    # the recurrence is the known recall-limited control. Both are recorded as
    # empirically characterized reference rows ("promoted" = adopted as baseline).
    return "promoted"


def seed_baselines(seed: int, quick: bool) -> None:
    reports: list[GauntletReport] = []
    for name in ("attention", "ssm"):
        print(f"[run_gauntlet] running baseline {name!r} ...", file=sys.stderr)
        reports.append(run_gauntlet(name, seed=seed, quick=quick))

    # V1 report
    lines = ["# Gauntlet baselines (V1 capability profiles)", ""]
    lines.append("Reference rows the harness must reproduce: attention aces MQAR across K; the")
    lines.append("recurrence shows the known recall gap (degrades as K grows) while still")
    lines.append("passing induction and selective-copy. If this separation is absent the harness")
    lines.append("is broken (charter V1).")
    lines.append("")
    for k, v in env_header(seed=seed).items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    for report in reports:
        lines.append(render_markdown(report))
        lines.append("")
    BASELINE_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"[run_gauntlet] wrote {BASELINE_REPORT}", file=sys.stderr)

    # Seed ledger (append-only) and regenerate the human view.
    for report in reports:
        mixer = get_mixer(report.mixer_name)(128)
        entry = make_entry(
            name=report.mixer_name,
            gate_card=mixer.gate_card.as_dict(),
            verdict=_ledger_verdict(report),
            reason=f"reference baseline; gauntlet verdict: {report.verdict()}",
            gauntlet_summary=_gauntlet_summary(report),
        )
        append_entry(LEDGER_JSONL, entry)
    regenerate_markdown_view(LEDGER_JSONL, LEDGER_MD)
    print(f"[run_gauntlet] seeded ledger -> {LEDGER_JSONL} and {LEDGER_MD}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mixer", nargs="?", choices=sorted(REGISTRY), help="registered mixer name")
    parser.add_argument(
        "--baselines", action="store_true", help="run both baselines, write V1 report + seed ledger"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true", help="fast reduced run")
    args = parser.parse_args(argv)

    if args.baselines:
        seed_baselines(args.seed, args.quick)
        return 0
    if not args.mixer:
        parser.error("provide a mixer name or --baselines")
    report = run_gauntlet(args.mixer, seed=args.seed, quick=args.quick)
    print(render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

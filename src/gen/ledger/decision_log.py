"""Append-only decision ledger (JSONL). One entry per screened candidate.

Charter §4: every candidate's fate is recorded — the four-gate self-scores, the
empirical gauntlet summary if it earned a kill test, the verdict, a one-line
reason, and a link to the screen doc. The JSONL file is the source of truth and
is *append-only*: entries are never rewritten. `DECISION_LOG.md` is a generated
human view produced from it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

# The four terminal fates of a candidate (charter method).
VERDICTS = ("paper-dead", "earned-kill-test", "killed-empirically", "promoted")
GATES = ("expressivity", "trainability", "hardware", "scaling")


@dataclass(frozen=True)
class LedgerEntry:
    name: str
    date: str
    verdict: str
    reason: str
    gate_card: dict[str, dict[str, str]]
    screen_doc: str | None = None
    gauntlet_summary: dict[str, object] | None = None

    def __post_init__(self) -> None:
        if self.verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {VERDICTS}, got {self.verdict!r}")
        missing = set(GATES) - set(self.gate_card)
        if missing:
            raise ValueError(f"gate_card missing gates: {sorted(missing)}")
        for gate, card in self.gate_card.items():
            if "status" not in card or "justification" not in card:
                raise ValueError(f"gate {gate!r} card needs status+justification, got {card!r}")
        if not self.reason.strip():
            raise ValueError("ledger entry needs a one-line reason")


def append_entry(path: str | Path, entry: LedgerEntry) -> None:
    """Append one entry as a JSON line. Never rewrites existing lines."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(entry), sort_keys=True) + "\n")


def read_entries(path: str | Path) -> list[LedgerEntry]:
    p = Path(path)
    if not p.exists():
        return []
    entries: list[LedgerEntry] = []
    with p.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"corrupt ledger at {p}:{lineno}: {exc}") from exc
            entries.append(LedgerEntry(**data))
    return entries


def make_entry(
    name: str,
    gate_card: dict[str, dict[str, str]],
    verdict: str,
    reason: str,
    *,
    screen_doc: str | None = None,
    gauntlet_summary: dict[str, object] | None = None,
    on: date | None = None,
) -> LedgerEntry:
    return LedgerEntry(
        name=name,
        date=(on or date.today()).isoformat(),
        verdict=verdict,
        reason=reason,
        gate_card=gate_card,
        screen_doc=screen_doc,
        gauntlet_summary=gauntlet_summary,
    )


def render_markdown(entries: list[LedgerEntry]) -> str:
    lines = [
        "# Decision Log",
        "",
        "Generated view of `decision_log.jsonl` (the append-only source of truth).",
        "Do not edit by hand — regenerate with `scripts/run_gauntlet.py` / ledger tooling.",
        "",
        "| candidate | date | verdict | gates (E/T/H/S) | reason | screen |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    short = {"pass": "P", "risk": "~", "fail": "F"}
    for e in entries:
        gates = "/".join(short.get(e.gate_card[g]["status"], "?") for g in GATES)
        screen = f"[doc]({e.screen_doc})" if e.screen_doc else "—"
        lines.append(f"| `{e.name}` | {e.date} | {e.verdict} | {gates} | {e.reason} | {screen} |")
    lines.append("")
    return "\n".join(lines)


def regenerate_markdown_view(jsonl_path: str | Path, md_path: str | Path) -> None:
    entries = read_entries(jsonl_path)
    Path(md_path).write_text(render_markdown(entries), encoding="utf-8")

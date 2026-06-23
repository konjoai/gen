#!/usr/bin/env python3
"""Scaffold a four-gate paper-screen doc for a candidate primitive.

This does NOT write the screen — the analytic argument is authored by a human
(charter §4 step 3). It stamps out `docs/screens/<candidate>.md` from
`docs/FOUR_GATE_TEMPLATE.md` with the candidate name and date filled in, and
prints the matching append-only ledger snippet to add once the screen is done.

    python scripts/paper_screen.py tropical_max_plus
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "docs" / "FOUR_GATE_TEMPLATE.md"
SCREENS_DIR = ROOT / "docs" / "screens"

SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", help="candidate slug, e.g. tropical_max_plus")
    parser.add_argument("--force", action="store_true", help="overwrite an existing screen doc")
    args = parser.parse_args(argv)

    candidate = args.candidate.strip().lower()
    if not SLUG_RE.match(candidate):
        parser.error(f"candidate must be a lowercase slug [a-z0-9_], got {args.candidate!r}")
    if not TEMPLATE.exists():
        print(f"FAIL: template not found at {TEMPLATE}", file=sys.stderr)
        return 1

    SCREENS_DIR.mkdir(parents=True, exist_ok=True)
    out = SCREENS_DIR / f"{candidate}.md"
    if out.exists() and not args.force:
        print(f"FAIL: {out} already exists (use --force to overwrite)", file=sys.stderr)
        return 1

    today = date.today().isoformat()
    body = TEMPLATE.read_text(encoding="utf-8")
    body = body.replace("{{CANDIDATE}}", candidate).replace("{{DATE}}", today)
    out.write_text(body, encoding="utf-8")
    print(f"scaffolded {out}")
    print()
    print("When the screen is filled in, append a ledger entry, e.g.:")
    print(
        '  python -c "from gen.ledger import append_entry, make_entry; '
        "append_entry('src/gen/ledger/decision_log.jsonl', make_entry("
        f"'{candidate}', gate_card={{...}}, verdict='earned-kill-test', "
        f"reason='...', screen_doc='docs/screens/{candidate}.md'))\""
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

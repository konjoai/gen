"""Console-script entry point for the gauntlet (`gen-gauntlet`).

Thin wrapper over `scripts/run_gauntlet.py` logic so the gauntlet is runnable
both as an installed command and as a script.
"""

from __future__ import annotations

import argparse

from gen.gauntlet.report import render_markdown, run_gauntlet
from gen.primitives import REGISTRY


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the synthetic gauntlet for a named mixer.")
    parser.add_argument("mixer", choices=sorted(REGISTRY), help="registered mixer name")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true", help="fast reduced run (CI sanity)")
    parser.add_argument("--device", default=None, help="torch device (default: auto)")
    args = parser.parse_args(argv)

    report = run_gauntlet(args.mixer, seed=args.seed, device=args.device, quick=args.quick)
    print(render_markdown(report))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

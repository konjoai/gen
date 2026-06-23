#!/usr/bin/env python3
"""Dependency provenance gate (charter §7 KCQF-lite).

Every dependency declared in pyproject.toml must be pinned to an exact version
(``name==X.Y.Z``) and that exact version must be the one installed. Inferred
package names, ranges, and version drift all fail loud with a non-zero exit.
Runtime deps are additionally import-checked. Wired into pre-commit and CI.
"""

from __future__ import annotations

import sys
import tomllib
from importlib import import_module
from importlib import metadata as importlib_metadata
from pathlib import Path

# Distribution name -> import module name, where they differ.
IMPORT_NAME = {"konjo-gen": "gen"}
# Dev tools that are not import modules (CLI binaries); version-check only.
NON_IMPORTABLE = {"ruff", "pre-commit", "mypy", "pytest"}


def _parse_pin(spec: str) -> tuple[str, str]:
    if "==" not in spec:
        raise ValueError(f"dependency not pinned with '==': {spec!r}")
    name, version = spec.split("==", 1)
    name = name.strip()
    version = version.strip()
    # reject compound specifiers like "x==1,>=0.9"
    if any(c in version for c in ",<>! "):
        raise ValueError(f"dependency has a non-exact specifier: {spec!r}")
    return name, version


def _collect_deps(pyproject: dict) -> list[str]:
    project = pyproject.get("project", {})
    deps = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        deps.extend(extra)
    return deps


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    pyproject_path = root / "pyproject.toml"
    if not pyproject_path.exists():
        print(f"FAIL: no pyproject.toml at {pyproject_path}", file=sys.stderr)
        return 1

    with pyproject_path.open("rb") as f:
        pyproject = tomllib.load(f)

    problems: list[str] = []
    checked = 0
    for spec in _collect_deps(pyproject):
        checked += 1
        try:
            name, want = _parse_pin(spec)
        except ValueError as exc:
            problems.append(str(exc))
            continue
        try:
            have = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            problems.append(f"{name}: pinned {want} but not installed")
            continue
        # `name==X` (no local segment) is satisfied by any local build `X+local`
        # per PEP 440 (e.g. torch 2.5.1 vs the installed 2.5.1+cpu wheel).
        have_release = have.split("+", 1)[0]
        if have != want and have_release != want:
            problems.append(f"{name}: pinned {want} but installed {have}")
            continue
        if name not in NON_IMPORTABLE:
            mod = IMPORT_NAME.get(name, name.replace("-", "_"))
            try:
                import_module(mod)
            except Exception as exc:  # noqa: BLE001 -- report any import failure verbatim
                problems.append(f"{name}: installed {have} but `import {mod}` failed: {exc!r}")

    if problems:
        print("verify_deps: FAIL", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    print(f"verify_deps: OK ({checked} dependencies pinned, installed, and importable)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

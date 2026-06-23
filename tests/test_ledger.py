"""Tests for the append-only decision ledger."""

from __future__ import annotations

from datetime import date

import pytest

from gen.ledger import (
    GATES,
    LedgerEntry,
    append_entry,
    make_entry,
    read_entries,
    regenerate_markdown_view,
    render_markdown,
)


def _card(status: str = "pass") -> dict[str, dict[str, str]]:
    return {g: {"status": status, "justification": f"{g} ok"} for g in GATES}


def test_append_then_read_roundtrip(tmp_path) -> None:
    path = tmp_path / "log.jsonl"
    e = make_entry("attention", _card(), "promoted", "reference baseline", on=date(2026, 1, 1))
    append_entry(path, e)
    got = read_entries(path)
    assert len(got) == 1
    assert got[0].name == "attention"
    assert got[0].verdict == "promoted"


def test_append_only_accumulates(tmp_path) -> None:
    path = tmp_path / "log.jsonl"
    append_entry(path, make_entry("a", _card(), "promoted", "r1"))
    append_entry(path, make_entry("b", _card(), "paper-dead", "r2"))
    entries = read_entries(path)
    assert [e.name for e in entries] == ["a", "b"]


def test_bad_verdict_rejected() -> None:
    with pytest.raises(ValueError):
        LedgerEntry(name="x", date="2026-01-01", verdict="nope", reason="r", gate_card=_card())


def test_missing_gate_rejected() -> None:
    partial = {"expressivity": {"status": "pass", "justification": "x"}}
    with pytest.raises(ValueError):
        LedgerEntry(name="x", date="2026-01-01", verdict="promoted", reason="r", gate_card=partial)


def test_empty_reason_rejected() -> None:
    with pytest.raises(ValueError):
        LedgerEntry(name="x", date="2026-01-01", verdict="promoted", reason="  ", gate_card=_card())


def test_markdown_view_generation(tmp_path) -> None:
    path = tmp_path / "log.jsonl"
    md = tmp_path / "DECISION_LOG.md"
    append_entry(path, make_entry("ssm", _card("risk"), "promoted", "recall-limited control"))
    regenerate_markdown_view(path, md)
    text = md.read_text()
    assert "ssm" in text
    assert "Decision Log" in text


def test_render_markdown_directly() -> None:
    e = make_entry("attention", _card(), "promoted", "ref", screen_doc="docs/screens/x.md")
    out = render_markdown([e])
    assert "attention" in out
    assert "[doc]" in out

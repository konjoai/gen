"""Contract tests for the SequenceMixer plug and its gate-card structure."""

from __future__ import annotations

import pytest
import torch

from gen.primitives import REGISTRY, AttentionRef, SSMRef
from gen.primitives.base import (
    NO_RECURRENT_FORM,
    GateCard,
    GateVerdict,
    SequenceMixer,
)

D_MODEL = 32


@pytest.fixture(params=sorted(REGISTRY))
def mixer(request) -> SequenceMixer:
    torch.manual_seed(0)
    return REGISTRY[request.param](D_MODEL)


def test_forward_preserves_shape_and_is_causal(mixer: SequenceMixer) -> None:
    x = torch.randn(4, 16, D_MODEL)
    y = mixer(x)
    assert y.shape == x.shape

    # causality: perturbing a future token must not change an earlier output
    x2 = x.clone()
    x2[:, 10:] += 5.0
    y2 = mixer(x2)
    torch.testing.assert_close(y[:, :10], y2[:, :10], rtol=1e-4, atol=1e-4)


def test_shape_assertions_fire(mixer: SequenceMixer) -> None:
    with pytest.raises(ValueError):
        mixer(torch.randn(4, D_MODEL))  # missing time dim
    with pytest.raises(ValueError):
        mixer(torch.randn(4, 16, D_MODEL + 1))  # wrong feature dim
    with pytest.raises(TypeError):
        mixer(torch.randint(0, 3, (4, 16, D_MODEL)))  # non-float


def test_flops_positive_and_monotone(mixer: SequenceMixer) -> None:
    assert mixer.flops(16) > 0
    assert mixer.flops(64) > mixer.flops(16)


def test_gate_card_has_four_gates(mixer: SequenceMixer) -> None:
    card = mixer.gate_card
    for gate in ("expressivity", "trainability", "hardware", "scaling"):
        verdict = getattr(card, gate)
        assert verdict.status in ("pass", "risk", "fail")
        assert verdict.justification.strip()


def test_attention_step_matches_forward() -> None:
    torch.manual_seed(0)
    m = AttentionRef(D_MODEL, n_heads=4)
    x = torch.randn(2, 8, D_MODEL)
    y_full = m(x)
    state = None
    for t in range(x.size(1)):
        y_t, state = m.step(x[:, t], state)
        torch.testing.assert_close(y_t, y_full[:, t], rtol=1e-4, atol=1e-4)


def test_ssm_step_matches_forward() -> None:
    torch.manual_seed(0)
    m = SSMRef(D_MODEL, n_heads=2, d_key=4)
    x = torch.randn(2, 8, D_MODEL)
    y_full = m(x)
    state = None
    for t in range(x.size(1)):
        y_t, state = m.step(x[:, t], state)
        torch.testing.assert_close(y_t, y_full[:, t], rtol=1e-4, atol=1e-4)


def test_gate_verdict_validation() -> None:
    with pytest.raises(ValueError):
        GateVerdict("maybe", "bad status")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        GateVerdict("pass", "   ")  # empty justification


def test_paper_verdict_logic() -> None:
    ok = GateVerdict("pass", "fine")
    fail = GateVerdict("fail", "no")
    card_dead = GateCard(expressivity=fail, trainability=ok, hardware=ok, scaling=ok)
    card_kill = GateCard(expressivity=ok, trainability=ok, hardware=ok, scaling=ok)
    assert card_dead.paper_verdict() == "paper-dead"
    assert card_kill.paper_verdict() == "earned-kill-test"


def test_default_step_is_sentinel() -> None:
    class NoRecur(SequenceMixer):
        def __init__(self) -> None:
            super().__init__()
            self.d_model = D_MODEL
            self.gate_card = GateCard(
                expressivity=GateVerdict("pass", "x"),
                trainability=GateVerdict("pass", "x"),
                hardware=GateVerdict("pass", "x"),
                scaling=GateVerdict("pass", "x"),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

        def flops(self, seq_len: int) -> int:
            return seq_len

    m = NoRecur()
    assert m.step(torch.zeros(1, D_MODEL), None) is NO_RECURRENT_FORM

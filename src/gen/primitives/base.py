"""The plug. Every candidate mixer implements `SequenceMixer`.

This file is the single most important contract in the repo: it encodes the
Konjo Charter's four-gate filter (docs/CHARTER.md, "The four-gate filter") as
first-class structure, so a candidate cannot enter the ledger without the
author having committed to a paper-screen verdict on each gate.

Keep this minimal. Resist adding methods the gauntlet does not call.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor, nn

# A mixer with no recurrent/inference form returns this sentinel from `step`;
# the harness notes it rather than treating absence as an error.
NO_RECURRENT_FORM = NotImplemented

GateName = Literal["expressivity", "trainability", "hardware", "scaling"]
GateStatus = Literal["pass", "risk", "fail"]


@dataclass(frozen=True)
class GateVerdict:
    """A self-assessed paper-screen verdict on one of the four charter gates."""

    status: GateStatus
    justification: str

    def __post_init__(self) -> None:
        if self.status not in ("pass", "risk", "fail"):
            raise ValueError(f"gate status must be pass|risk|fail, got {self.status!r}")
        if not self.justification or not self.justification.strip():
            raise ValueError("gate justification must be a non-empty one-liner")


@dataclass(frozen=True)
class GateCard:
    """The author's four-gate paper screen for a mixer.

    Gates, verbatim from the charter:
      1. Expressivity  — can it do exact retrieval / associative recall in principle?
      2. Trainability  — do gradients survive at depth and length?
      3. Hardware      — does it reduce to dense matmul / parallel scan / FFT at high AI?
      4. Scaling       — does quality improve predictably as compute is added?

    A primitive that fails Gate 1 (expressivity) or Gate 3 (hardware) on paper is
    filed in the graveyard, not coded. The decision ledger reads this card.
    """

    expressivity: GateVerdict
    trainability: GateVerdict
    hardware: GateVerdict
    scaling: GateVerdict

    def as_dict(self) -> dict[str, dict[str, str]]:
        return {
            "expressivity": {
                "status": self.expressivity.status,
                "justification": self.expressivity.justification,
            },
            "trainability": {
                "status": self.trainability.status,
                "justification": self.trainability.justification,
            },
            "hardware": {
                "status": self.hardware.status,
                "justification": self.hardware.justification,
            },
            "scaling": {"status": self.scaling.status, "justification": self.scaling.justification},
        }

    def paper_verdict(self) -> str:
        """`paper-dead` if Gate 1 or Gate 3 fails on paper, else `earned-kill-test`.

        This mirrors charter §4 step 3: no code for ideas that fail Gate 1 or Gate 3.
        Baselines self-report `pass` everywhere and so earn a kill test trivially.
        """
        if self.expressivity.status == "fail" or self.hardware.status == "fail":
            return "paper-dead"
        return "earned-kill-test"


class SequenceMixer(nn.Module, abc.ABC):
    """Causal sequence-mixing primitive: ``(B, T, D) -> (B, T, D)``.

    Channel mixing (the FFN), normalization, embedding and head live in
    `gen.model`; a mixer is *only* the sequence-mixing slot (charter §1.1).
    """

    #: Set by the concrete subclass in __init__.
    d_model: int

    #: Author-filled four-gate paper screen. The ledger reads this.
    gate_card: GateCard

    @abc.abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Parallel/training path over the full sequence. Causal.

        Args:
            x: ``(B, T, D)`` float tensor.
        Returns:
            ``(B, T, D)`` float tensor; position ``t`` depends only on ``<= t``.
        """
        raise NotImplementedError

    def step(self, x_t: Tensor, state: Any) -> tuple[Tensor, Any] | type(NotImplemented):  # type: ignore[valid-type]
        """Optional recurrent/inference path for a single token.

        Returns ``(y_t, new_state)`` for ``x_t`` of shape ``(B, D)``, or the
        sentinel ``NO_RECURRENT_FORM`` if the mixer has no recurrent form.
        """
        return NO_RECURRENT_FORM

    @abc.abstractmethod
    def flops(self, seq_len: int) -> int:
        """Analytic FLOP estimate for one forward pass at ``seq_len``.

        For later Pareto accounting (charter §4.6). Required, not optional:
        a mixer that cannot state its own cost cannot be screened on Gate 3/4.
        """
        raise NotImplementedError

    # --- shared guards -----------------------------------------------------

    @staticmethod
    def _assert_seq_shape(x: Tensor, d_model: int) -> None:
        """Shape assertion at the mixer boundary — no silent broadcasting."""
        if x.dim() != 3:
            raise ValueError(
                f"expected (B, T, D), got tensor with {x.dim()} dims: {tuple(x.shape)}"
            )
        if x.size(-1) != d_model:
            raise ValueError(
                f"expected last dim D={d_model}, got {x.size(-1)} (shape {tuple(x.shape)})"
            )
        if not torch.is_floating_point(x):
            raise TypeError(f"mixer input must be floating point, got {x.dtype}")

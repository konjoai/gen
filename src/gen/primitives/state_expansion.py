"""Candidate primitive: delta-rule state-expansion recurrence (DeltaNet-style).

Second screened candidate (see ``docs/screens/state_expansion.md``). The aim is
the empty recall-and-state-tracking corner: a recurrence whose state is updated by
an *error-correcting* (Widrow-Hoff / delta) rule rather than the additive
outer-product write of the ``ssm`` baseline.

Per head, state ``S in R^{d_key x d_val}``, with unit-norm keys ``||k_t|| = 1``:

    u_t = beta_t ( v_t - S_{t-1}^T k_t )     # prediction error for key k_t
    S_t = S_{t-1} + k_t u_t^T                  # write the correction
    y_t = q_t^T S_t                            # readout

Equivalently ``S_t = (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T``: it erases
the value currently bound to ``k_t`` before writing the new one, so distinct keys
stop superposing (the additive baseline's recall disease). ``beta_t = 2*sigmoid(.)``
lives in ``(0, 2)``: the transition eigenvalue along ``k`` is ``1 - beta in (-1, 1)``,
and the *negative* range (beta > 1) is what unlocks state-tracking in linear RNNs
(Grazzi et al. 2024). With ``||k|| = 1`` the transition is a Householder-type
contraction (spectral radius <= 1), so the state stays bounded -- no winner-take-all
blow-up to guard against.

This is a baseline-grade *reference*: a slow-but-correct sequential scan, vectorized
over batch and heads. The known efficient form is the chunked parallel delta rule
(Yang et al. 2024), which is what Gate 3 screens for; it is deliberately not built.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from gen.primitives.base import GateCard, GateVerdict, SequenceMixer


class DeltaNetMixer(SequenceMixer):
    """Delta-rule linear recurrence with a fixed-size content-addressable state."""

    def __init__(self, d_model: int, n_heads: int = 4, d_key: int | None = None) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_val = d_model // n_heads
        self.d_key = d_key if d_key is not None else self.d_val

        self.q_proj = nn.Linear(d_model, n_heads * self.d_key, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * self.d_key, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * self.d_val, bias=False)
        self.beta_proj = nn.Linear(d_model, n_heads)  # scalar write strength per head
        self.proj = nn.Linear(d_model, d_model, bias=False)

        # Four-gate card, matching docs/screens/state_expansion.md.
        self.gate_card = GateCard(
            expressivity=GateVerdict(
                "pass",
                "error-correcting delta write erases the stale value for a key before storing the new one; recall is its designed strength.",
            ),
            trainability=GateVerdict(
                "risk",
                "BPTT through a chain of data-dependent (I-beta k k^T) transitions; the beta->2 state-tracking regime is delicate.",
            ),
            hardware=GateVerdict(
                "risk",
                "data-dependent rank-1 recurrence; a chunked parallel form (matmuls + triangular solve) exists, but the reference is a slow sequential scan.",
            ),
            scaling=GateVerdict(
                "risk",
                "clean head/d_key/d_val/depth knobs and an error-correcting state, but whether the recall ceiling moves with width is unproven.",
            ),
        )

    @staticmethod
    def _unit(x: Tensor) -> Tensor:
        return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    def _project(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        b, t, _ = x.shape
        h, dk, dv = self.n_heads, self.d_key, self.d_val
        q = self._unit(self.q_proj(x).view(b, t, h, dk))
        k = self._unit(self.k_proj(x).view(b, t, h, dk))  # unit norm required for the householder
        v = self.v_proj(x).view(b, t, h, dv)
        beta = 2.0 * torch.sigmoid(self.beta_proj(x)).view(
            b, t, h, 1
        )  # in (0, 2): enables negative eigenvalue
        return q, k, v, beta

    def forward(self, x: Tensor) -> Tensor:
        self._assert_seq_shape(x, self.d_model)
        b, t, _ = x.shape
        h, dk, dv = self.n_heads, self.d_key, self.d_val
        q, k, v, beta = self._project(x)

        s = x.new_zeros(b, h, dk, dv)  # recurrent state per head
        outputs = []
        for ti in range(t):
            k_t, q_t, v_t, b_t = k[:, ti], q[:, ti], v[:, ti], beta[:, ti]  # (B,H,*)
            pred = torch.einsum("bhk,bhkv->bhv", k_t, s)  # S^T k: current value bound to k_t
            u = b_t * (v_t - pred)  # delta (prediction error), (B,H,dv)
            s = s + k_t.unsqueeze(-1) * u.unsqueeze(-2)  # write correction, (B,H,dk,dv)
            outputs.append(torch.einsum("bhk,bhkv->bhv", q_t, s))  # readout (B,H,dv)
        out = torch.stack(outputs, dim=1).reshape(b, t, self.d_model)
        out = self.proj(out)

        if out.shape != x.shape:
            raise AssertionError(
                f"mixer must preserve shape: in {tuple(x.shape)} out {tuple(out.shape)}"
            )
        return out

    def step(self, x_t: Tensor, state: Tensor | None) -> tuple[Tensor, Tensor]:
        if x_t.dim() != 2 or x_t.size(-1) != self.d_model:
            raise ValueError(f"step expects (B, D={self.d_model}), got {tuple(x_t.shape)}")
        b = x_t.size(0)
        h, dk, dv = self.n_heads, self.d_key, self.d_val
        q, k, v, beta = (z.squeeze(1) for z in self._project(x_t.unsqueeze(1)))  # (B,H,*)
        s = x_t.new_zeros(b, h, dk, dv) if state is None else state
        pred = torch.einsum("bhk,bhkv->bhv", k, s)
        u = beta * (v - pred)
        s = s + k.unsqueeze(-1) * u.unsqueeze(-2)
        y = torch.einsum("bhk,bhkv->bhv", q, s).reshape(b, self.d_model)
        return self.proj(y), s

    def flops(self, seq_len: int) -> int:
        d = self.d_model
        t = seq_len
        h, dk, dv = self.n_heads, self.d_key, self.d_val
        proj_in = 2 * t * d * (h * dk * 2 + h * dv + h)  # q, k, v, beta
        # per step: S^T k, outer write, q^T S  -> O(h*dk*dv) each
        recur = t * (2 * h * dk * dv + 2 * h * dk * dv + 2 * h * dk * dv)
        proj_out = 2 * t * d * d
        return proj_in + recur + proj_out

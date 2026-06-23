"""Candidate primitive: tropical (max, +) sequence mixer.

First screened candidate (see ``docs/screens/tropical_max_plus.md``). Makes hard
selection the primitive instead of a softmax-weighted average: per head, with
compatibility ``a_ij = <q_i, k_j>`` (causal) and values ``V``,

    attention:  out_ic = sum_j softmax_j(a_ij) v_jc        (expectation)
    tropical:   out_ic = max_{j<=i} ( a_ij + v_jc )        ((max,+) product)

This is a recall-axis play: it tests whether hard (max, +) selection matches soft
attention on associative recall, i.e. whether the softmax average is load-bearing.

Trainability (the screen's Gate-2 risk) is handled by the differentiable
temperature relaxation

    out_ic = (1/beta) logsumexp_{j<=i} ( beta (a_ij + v_jc) )  ->  max as beta->inf,

with a learnable per-head ``beta``. ``forward`` computes this in the
Gate-3-efficient matmul form,

    out_ic = A_i + V_c + (1/beta) log( exp(beta(a-A_i)) @ exp(beta(v-V_c)) )_ic,

with *separable* shifts ``A_i = max_{j<=i} a_ij`` and ``V_c = max_j v_jc``. Only
the (T,T) and (T,d) factors are exponentiated and a single matmul contracts them,
so the (T,T,d) grid is never materialized -- matmul-adjacent exactly as Gate 3
argues. The separable shift is loose, so sharp selection can underflow the summed
exponentials; flooring that sum (``P_FLOOR``) bounds the log gradient instead of
letting 1/p blow up. That fragility is the honest Gate-3 caveat: a real candidate
owes a genuinely stable parallel kernel. Slow-but-correct reference, no fused
kernel.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn

from gen.primitives.base import NO_RECURRENT_FORM, GateCard, GateVerdict, SequenceMixer


class TropicalMaxPlus(SequenceMixer):
    """(max, +) semiring sequence mixer with a learnable softmax-temperature."""

    def __init__(self, d_model: int, n_heads: int = 4, beta_init: float = 1.0) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

        # Learnable per-head inverse temperature beta = softplus(beta_raw) > 0.
        # beta -> inf recovers exact (max, +); start near beta_init for smooth training.
        inv = math.log(math.expm1(beta_init)) if beta_init > 0 else 0.0
        self.beta_raw = nn.Parameter(torch.full((n_heads,), float(inv)))

        # Four-gate card, matching docs/screens/tropical_max_plus.md.
        self.gate_card = GateCard(
            expressivity=GateVerdict(
                "pass",
                "hard (max,+) selection is a native nearest-neighbor gather; recall is its core strength.",
            ),
            trainability=GateVerdict(
                "risk",
                "max is subgradient-only; a learnable logsumexp temperature restores dense gradients and anneals toward hard selection.",
            ),
            hardware=GateVerdict(
                "risk",
                "(max,+) matmul is matmul-adjacent (broadcast-add + max-reduce, no scatter/branching) but bandwidth-bound with no tensor-core FMA path.",
            ),
            scaling=GateVerdict(
                "risk",
                "clean width/head/depth/temperature knobs and no fixed-state recall ceiling, but hard-selection scaling is unproven.",
            ),
        )

    # Temperature is capped: sharper selection makes the separable-shift sum
    # underflow harder, so beta is bounded to keep the smoothed (max,+) operator in
    # a numerically usable window (logsumexp-vs-max gap ~ log(T)/BETA_MAX). A fully
    # hard limit would need the (B,H,T,T,d) tensor we deliberately avoid (Gate 3).
    BETA_MAX: float = 20.0
    # Floor on the summed exponentials: bounds the log gradient at 1/P_FLOOR so an
    # underflowed `p` cannot send 1/p -> inf and NaN the step (see forward).
    P_FLOOR: float = 1e-6

    def beta(self) -> Tensor:
        return nn.functional.softplus(self.beta_raw).clamp(max=self.BETA_MAX)  # (H,)

    def forward(self, x: Tensor) -> Tensor:
        self._assert_seq_shape(x, self.d_model)
        b, t, _ = x.shape
        h, dh = self.n_heads, self.d_head

        q = self.q_proj(x).view(b, t, h, dh).transpose(1, 2)  # (B,H,T,dh)
        k = self.k_proj(x).view(b, t, h, dh).transpose(1, 2)
        v = self.v_proj(x).view(b, t, h, dh).transpose(1, 2)

        # QK-normalization (charter §2.5): unit-norm q, k bound the compatibility to
        # a = sqrt(dh) * cos in [-sqrt(dh), sqrt(dh)]. Without this, winner-take-all
        # selection drives the projection weights up until q.k overflows and
        # `a - a_max` becomes inf - inf = NaN. Selection sharpness then comes from
        # beta, not from unbounded score growth.
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        a = (q @ k.transpose(-2, -1)) * math.sqrt(dh)  # (B,H,Ti,Tj); bounded, finite
        causal = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        beta = self.beta().view(1, h, 1, 1)  # (1,H,1,1)

        # The relaxation computed in the Gate-3-efficient matmul form:
        #   out_ic = (1/beta) logsumexp_{j<=i}(beta(a_ij+v_jc))
        #          = A_i + V_c + (1/beta) log( exp(beta(a-A_i)) @ exp(beta(v-V_c)) )_ic,
        # with *separable* max-shifts A_i = max_{j<=i} a_ij and V_c = max_j v_jc
        # (detached: a shift constant over the reduction index leaves the gradient
        # unchanged, and detaching keeps the -inf used for A_i out of the graph,
        # where it would NaN the beta gradient via (-inf)*0). This is matmul-adjacent
        # exactly as argued in Gate 3: exp over the (T,T) and (T,d) factors, then one
        # matmul -- never the (T,T,d) grid.
        #
        # Numerical safeguard (the Gate-3 fragility made concrete): the separable
        # shift is loose, so under sharp selection the summed exponentials can
        # underflow. Flooring `p` at P_FLOOR bounds the log gradient at 1/P_FLOOR
        # (finite, so grad-clipping tames it) instead of letting 1/p blow up to inf
        # and poison training. A real candidate owes a genuinely stable parallel
        # kernel; that is what Gate 3 screens for.
        a_for_max = a.masked_fill(causal, float("-inf"))
        a_max = a_for_max.max(
            dim=-1, keepdim=True
        ).values.detach()  # (B,H,Ti,1); finite (j=i present)
        v_max = v.max(dim=-2, keepdim=True).values.detach()  # (B,H,1,dh)

        # Exponent floor keeps non-selected terms from underflowing to exact 0
        # (which would zero their gradient); paired with P_FLOOR on the sum.
        w = torch.exp((beta * (a - a_max)).clamp_min(-60.0)).masked_fill(causal, 0.0)
        e = torch.exp((beta * (v - v_max)).clamp_min(-60.0))  # (B,H,Tj,dh)
        p = (w @ e).clamp_min(self.P_FLOOR)  # (B,H,Ti,dh)
        out = a_max + v_max + torch.log(p) / beta  # (B,H,Ti,dh)
        out = out.transpose(1, 2).reshape(b, t, self.d_model)
        out = self.proj(out)

        if out.shape != x.shape:
            raise AssertionError(
                f"mixer must preserve shape: in {tuple(x.shape)} out {tuple(out.shape)}"
            )
        return out

    def step(self, x_t: Tensor, state: Any) -> tuple[Tensor, Any] | type(NotImplemented):  # type: ignore[valid-type]
        # No fixed-size recurrent form: like attention, compatibility couples the
        # current query with every past key, so recall stays quadratic. Honest
        # `no` rather than a growing-cache pseudo-recurrence.
        return NO_RECURRENT_FORM

    def flops(self, seq_len: int) -> int:
        d = self.d_model
        t = seq_len
        qkv = 2 * t * d * (3 * d)
        scores = 2 * t * t * d  # q@k^T
        wexp = t * t * d  # exp on W (elementwise over the T,T,dh access pattern)
        wesum = 2 * t * t * d  # W' @ E' (the (max,+)-relaxed contraction)
        proj = 2 * t * d * d
        return qkv + scores + wexp + wesum + proj

"""Reference baseline: minimal selective linear recurrence (linear attention).

The "known recall-limited" control. It is input-dependent (a selective write
gate) and exposes a true ``step`` recurrence with a *fixed-size* matrix state, so
its associative-memory capacity is bounded by the key-feature dimension
``n_heads * d_key``. That bound is the point: it should pass induction and
selective-copy but degrade on MQAR as the key count K grows past its capacity —
the gap the gauntlet must reproduce (charter §1 constraint 3, "the illusion of
state").

This is a baseline, not a contribution. The ``forward`` path uses the vectorized
cumulative-sum (masked) form of the recurrence; it computes exactly the same
function as ``step`` applied token-by-token, just faster and differentiably. A
real candidate would have to supply a genuine parallel scan — which is exactly
what Gate 3 screens for.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

from gen.primitives.base import GateCard, GateVerdict, SequenceMixer


class SSMRef(SequenceMixer):
    """Gated linear attention with a fixed-size matrix state per head.

    State per head is ``S in R^{d_key x d_val}`` accumulated from outer products
    ``(beta_t k_t) v_t^T`` with an input-dependent write gate ``beta_t`` (the
    selectivity). Read out by ``q_t^T S_t``. Associative capacity ~
    ``n_heads * d_key``, kept deliberately small relative to ``d_model`` so recall
    saturates as K grows.
    """

    def __init__(self, d_model: int, n_heads: int = 2, d_key: int = 8) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_key = d_key
        self.d_val = d_model // n_heads

        self.q_proj = nn.Linear(d_model, n_heads * d_key, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * d_key, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * self.d_val, bias=False)
        # Input-dependent write gate (the "selective" part): scalar per head.
        self.gate = nn.Linear(d_model, n_heads)
        self.proj = nn.Linear(d_model, d_model, bias=False)

        # Four-gate paper screen for the recall-limited control.
        self.gate_card = GateCard(
            expressivity=GateVerdict(
                "risk",
                "fixed-size state binds only ~n_heads*d_key associations; recall saturates as K grows.",
            ),
            trainability=GateVerdict(
                "pass", "linear recurrence trains stably; bounded, well-conditioned gradients."
            ),
            hardware=GateVerdict(
                "pass", "reduces to a parallel associative scan / matmul; Gate-3 friendly at scale."
            ),
            scaling=GateVerdict(
                "risk",
                "scales smoothly on local tasks but the recall ceiling does not move with width alone.",
            ),
        )

    @staticmethod
    def _feature(x: Tensor) -> Tensor:
        # Non-negative feature map keeps the linear-attention kernel well-behaved.
        return F.elu(x) + 1.0

    def forward(self, x: Tensor) -> Tensor:
        self._assert_seq_shape(x, self.d_model)
        b, t, _ = x.shape
        h, dk, dv = self.n_heads, self.d_key, self.d_val

        q = self._feature(self.q_proj(x)).view(b, t, h, dk)
        k = self._feature(self.k_proj(x)).view(b, t, h, dk)
        v = self.v_proj(x).view(b, t, h, dv)
        beta = torch.sigmoid(self.gate(x)).view(b, t, h, 1)  # write gate in (0,1)
        k = k * beta

        # Cumulative-sum (causal) linear attention == the recurrence below, vectorized.
        # scores[b,h,i,j] = <q_i, k_j>, masked to j <= i.
        scores = torch.einsum("bihd,bjhd->bhij", q, k)
        causal = torch.tril(torch.ones(t, t, device=x.device, dtype=torch.bool))
        scores = scores * causal  # zero out future; all terms are >= 0
        num = torch.einsum("bhij,bjhv->bihv", scores, v)  # (B,T,H,dv)
        den = scores.sum(dim=-1).transpose(1, 2).unsqueeze(-1).clamp_min(1e-6)  # (B,T,H,1)
        out = (num / den).reshape(b, t, h * dv)
        out = self.proj(out)

        if out.shape != x.shape:
            raise AssertionError(
                f"mixer must preserve shape: in {tuple(x.shape)} out {tuple(out.shape)}"
            )
        return out

    def step(
        self, x_t: Tensor, state: tuple[Tensor, Tensor] | None
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        if x_t.dim() != 2 or x_t.size(-1) != self.d_model:
            raise ValueError(f"step expects (B, D={self.d_model}), got {tuple(x_t.shape)}")
        b = x_t.size(0)
        h, dk, dv = self.n_heads, self.d_key, self.d_val
        q = self._feature(self.q_proj(x_t)).view(b, h, dk)
        k = self._feature(self.k_proj(x_t)).view(b, h, dk)
        v = self.v_proj(x_t).view(b, h, dv)
        beta = torch.sigmoid(self.gate(x_t)).view(b, h, 1)
        k = k * beta

        if state is None:
            S = x_t.new_zeros(b, h, dk, dv)
            z = x_t.new_zeros(b, h, dk)
        else:
            S, z = state
        S = S + k.unsqueeze(-1) * v.unsqueeze(-2)
        z = z + k
        num = torch.einsum("bhk,bhkv->bhv", q, S)
        den = torch.einsum("bhk,bhk->bh", q, z).clamp_min(1e-6).unsqueeze(-1)
        out = (num / den).reshape(b, h * dv)
        out = self.proj(out)
        return out, (S, z)

    def flops(self, seq_len: int) -> int:
        d = self.d_model
        t = seq_len
        h, dk, dv = self.n_heads, self.d_key, self.d_val
        proj_in = 2 * t * d * (h * dk * 2 + h * dv + h)  # q,k,v,gate
        # per step: outer product k v^T, q.S, q.z  -> O(h*dk*dv)
        recur = t * (2 * h * dk * dv + 2 * h * dk * dv + 2 * h * dk)
        proj_out = 2 * t * d * d
        return proj_in + recur + proj_out

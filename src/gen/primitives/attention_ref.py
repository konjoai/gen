"""Reference baseline: plain causal multi-head softmax attention.

The "known-capable" control. This is the architecture the gauntlet exists to
find a *cheaper* equal of; it should ace every task, MQAR included. Plain and
readable, not a contribution and not optimized.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn

from gen.primitives.base import GateCard, GateVerdict, SequenceMixer


class AttentionRef(SequenceMixer):
    """Causal multi-head softmax attention. O(T^2) sequence mixing."""

    def __init__(self, d_model: int, n_heads: int = 4) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

        # Four-gate paper screen for the control. Softmax attention is the
        # reference every gate is defined against, so it passes by construction
        # except hardware (quadratic, but dense-matmul — a `risk` on cost, not a fail).
        self.gate_card = GateCard(
            expressivity=GateVerdict(
                "pass", "exact softmax retrieval over the full context; the recall reference."
            ),
            trainability=GateVerdict(
                "pass", "standard residual+norm transformer block trains to depth routinely."
            ),
            hardware=GateVerdict(
                "risk", "pure dense matmul (Gate-3 friendly) but O(T^2) compute/memory in T."
            ),
            scaling=GateVerdict("pass", "the empirically validated scaling-law architecture."),
        )

    def forward(self, x: Tensor) -> Tensor:
        self._assert_seq_shape(x, self.d_model)
        b, t, _ = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)  # each (B, T, H, Dh)
        q, k, v = (z.transpose(1, 2) for z in (q, k, v))  # (B, H, T, Dh)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B, H, T, T)
        causal = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = attn @ v  # (B, H, T, Dh)
        out = out.transpose(1, 2).reshape(b, t, self.d_model)
        out = self.proj(out)

        if out.shape != x.shape:
            raise AssertionError(
                f"mixer must preserve shape: in {tuple(x.shape)} out {tuple(out.shape)}"
            )
        return out

    def step(self, x_t: Tensor, state: Any) -> tuple[Tensor, Any]:
        """Single-token decode with a growing KV cache (state = (K, V) lists).

        state: tuple(k_cache, v_cache) each (B, H, t_prev, Dh) or None.
        """
        if x_t.dim() != 2 or x_t.size(-1) != self.d_model:
            raise ValueError(f"step expects (B, D={self.d_model}), got {tuple(x_t.shape)}")
        b = x_t.size(0)
        qkv = self.qkv(x_t).view(b, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=1)  # (B, H, Dh)
        q = q.unsqueeze(2)  # (B, H, 1, Dh)
        k = k.unsqueeze(2)
        v = v.unsqueeze(2)
        if state is not None:
            k_prev, v_prev = state
            k = torch.cat([k_prev, k], dim=2)
            v = torch.cat([v_prev, v], dim=2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B, H, 1, t)
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ v).squeeze(2)  # (B, H, Dh)
        out = out.reshape(b, self.d_model)
        out = self.proj(out)
        return out, (k, v)

    def flops(self, seq_len: int) -> int:
        d = self.d_model
        t = seq_len
        qkv = 2 * t * d * (3 * d)
        scores = 2 * t * t * d  # q@k^T over all heads
        av = 2 * t * t * d  # attn@v
        proj = 2 * t * d * d
        return qkv + scores + av + proj

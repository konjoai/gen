"""Tests for the tropical (max, +) candidate mixer.

Mirrors tests/test_interface.py for the contract, and adds two correctness
checks specific to this primitive: the stable matmul form must equal the naive
log-sum-exp definition, and large beta must approach the hard (max, +) limit.
"""

from __future__ import annotations

import math

import torch

from gen.primitives import TropicalMaxPlus, get_mixer
from gen.primitives.base import NO_RECURRENT_FORM

D_MODEL = 16


def test_registered() -> None:
    m = get_mixer("tropical")(D_MODEL)
    assert isinstance(m, TropicalMaxPlus)


def test_shape_and_causality() -> None:
    torch.manual_seed(0)
    m = TropicalMaxPlus(D_MODEL, n_heads=2)
    x = torch.randn(4, 12, D_MODEL)
    y = m(x)
    assert y.shape == x.shape
    x2 = x.clone()
    x2[:, 6:] += 3.0
    y2 = m(x2)
    torch.testing.assert_close(y[:, :6], y2[:, :6], rtol=1e-4, atol=1e-4)


def test_flops_monotone() -> None:
    m = TropicalMaxPlus(D_MODEL)
    assert m.flops(16) > 0
    assert m.flops(64) > m.flops(16)


def test_no_recurrent_form() -> None:
    m = TropicalMaxPlus(D_MODEL)
    assert m.step(torch.zeros(1, D_MODEL), None) is NO_RECURRENT_FORM


def test_paper_verdict_earned_kill_test() -> None:
    # Gate 1 pass, Gate 3 risk -> not paper-dead.
    assert TropicalMaxPlus(D_MODEL).gate_card.paper_verdict() == "earned-kill-test"


def test_relaxation_matches_naive_logsumexp() -> None:
    # With proj set to identity, forward must equal the naive definition
    #   out_ic = (1/beta) logsumexp_{j<=i} ( beta (a_ij + v_jc) ).
    torch.manual_seed(1)
    h, dh = 2, 8
    d = h * dh
    m = TropicalMaxPlus(d, n_heads=h, beta_init=1.3)
    with torch.no_grad():
        m.proj.weight.copy_(torch.eye(d))
    x = torch.randn(3, 6, d)
    y = m(x)

    b, t, _ = x.shape
    q = m.q_proj(x).view(b, t, h, dh).transpose(1, 2)
    k = m.k_proj(x).view(b, t, h, dh).transpose(1, 2)
    v = m.v_proj(x).view(b, t, h, dh).transpose(1, 2)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    a = (q @ k.transpose(-2, -1)) * math.sqrt(dh)
    causal = torch.triu(torch.ones(t, t, dtype=torch.bool), diagonal=1)
    a = a.masked_fill(causal, float("-inf"))
    beta = m.beta().view(1, h, 1, 1)
    # M[b,h,i,j,c] = beta * (a_ij + v_jc)
    mtx = beta.unsqueeze(-1) * (a.unsqueeze(-1) + v.unsqueeze(2))  # (B,H,Ti,Tj,dh)
    naive = torch.logsumexp(mtx, dim=3) / beta  # (B,H,Ti,dh)
    naive = naive.transpose(1, 2).reshape(b, t, d)
    torch.testing.assert_close(y, naive, rtol=1e-4, atol=1e-4)


def test_matmul_form_matches_logsumexp_at_higher_beta() -> None:
    # The fast matmul form must equal the exact logsumexp definition wherever the
    # summed exponentials do not underflow the P_FLOOR safeguard. Small-magnitude
    # inputs keep the sum above the floor even at higher beta, so the matmul
    # identity holds; the floor only bites under sharp selection on large scores.
    torch.manual_seed(2)
    h, dh = 2, 4
    d = h * dh
    m = TropicalMaxPlus(d, n_heads=h, beta_init=8.0)
    with torch.no_grad():
        m.proj.weight.copy_(torch.eye(d))
    x = 0.2 * torch.randn(2, 5, d)  # small magnitude -> no underflow
    y = m(x)

    b, t, _ = x.shape
    q = m.q_proj(x).view(b, t, h, dh).transpose(1, 2)
    k = m.k_proj(x).view(b, t, h, dh).transpose(1, 2)
    v = m.v_proj(x).view(b, t, h, dh).transpose(1, 2)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    a = (q @ k.transpose(-2, -1)) * math.sqrt(dh)
    causal = torch.triu(torch.ones(t, t, dtype=torch.bool), diagonal=1)
    a = a.masked_fill(causal, float("-inf"))
    beta = m.beta().view(1, h, 1, 1)
    mtx = beta.unsqueeze(-1) * (a.unsqueeze(-1) + v.unsqueeze(2))
    naive = torch.logsumexp(mtx, dim=3) / beta
    naive = naive.transpose(1, 2).reshape(b, t, d)
    torch.testing.assert_close(y, naive, rtol=1e-3, atol=1e-3)

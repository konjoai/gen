"""Tests for the delta-rule state-expansion candidate mixer."""

from __future__ import annotations

import torch

from gen.primitives import DeltaNetMixer, get_mixer

D_MODEL = 16


def test_registered() -> None:
    assert isinstance(get_mixer("state_expansion")(D_MODEL), DeltaNetMixer)


def test_shape_and_causality() -> None:
    torch.manual_seed(0)
    m = DeltaNetMixer(D_MODEL, n_heads=2)
    x = torch.randn(4, 12, D_MODEL)
    y = m(x)
    assert y.shape == x.shape
    x2 = x.clone()
    x2[:, 6:] += 3.0
    y2 = m(x2)
    torch.testing.assert_close(y[:, :6], y2[:, :6], rtol=1e-4, atol=1e-4)


def test_flops_monotone() -> None:
    m = DeltaNetMixer(D_MODEL)
    assert m.flops(16) > 0
    assert m.flops(64) > m.flops(16)


def test_paper_verdict_earned_kill_test() -> None:
    assert DeltaNetMixer(D_MODEL).gate_card.paper_verdict() == "earned-kill-test"


def test_step_matches_forward() -> None:
    # Genuine recurrent form: token-by-token step must reproduce the parallel scan.
    torch.manual_seed(0)
    m = DeltaNetMixer(D_MODEL, n_heads=2)
    x = torch.randn(3, 9, D_MODEL)
    y_full = m(x)
    state = None
    for t in range(x.size(1)):
        y_t, state = m.step(x[:, t], state)
        torch.testing.assert_close(y_t, y_full[:, t], rtol=1e-4, atol=1e-4)


def test_forward_matches_delta_reference() -> None:
    # With proj = identity, forward must equal an explicit delta-rule scan.
    torch.manual_seed(1)
    h = 2
    m = DeltaNetMixer(D_MODEL, n_heads=h)
    with torch.no_grad():
        m.proj.weight.copy_(torch.eye(D_MODEL))
    x = torch.randn(2, 7, D_MODEL)
    y = m(x)

    q, k, v, beta = m._project(x)  # (B,T,H,*)
    b, t = x.shape[0], x.shape[1]
    dk, dv = m.d_key, m.d_val
    s = torch.zeros(b, h, dk, dv)
    outs = []
    for ti in range(t):
        pred = torch.einsum("bhk,bhkv->bhv", k[:, ti], s)
        u = beta[:, ti] * (v[:, ti] - pred)
        s = s + k[:, ti].unsqueeze(-1) * u.unsqueeze(-2)
        outs.append(torch.einsum("bhk,bhkv->bhv", q[:, ti], s))
    ref = torch.stack(outs, dim=1).reshape(b, t, D_MODEL)
    torch.testing.assert_close(y, ref, rtol=1e-5, atol=1e-5)


def test_delta_rule_overwrites_on_repeat() -> None:
    # Writing the same key twice at beta=1 (the clean-overwrite point, since
    # k^T S = beta * v for ||k||=1) drives the prediction error to ~0 the second
    # time (error-correcting overwrite). Probe one head directly.
    torch.manual_seed(2)
    m = DeltaNetMixer(8, n_heads=1)
    with torch.no_grad():
        m.beta_proj.weight.zero_()
        m.beta_proj.bias.zero_()  # 2*sigmoid(0) = 1 exactly
    x = torch.randn(1, 1, 8)
    q, k, v, beta = m._project(x.repeat(1, 2, 1))  # same token twice
    s = torch.zeros(1, 1, m.d_key, m.d_val)
    err = []
    for ti in range(2):
        pred = torch.einsum("bhk,bhkv->bhv", k[:, ti], s)
        err.append((v[:, ti] - pred).abs().mean().item())
        u = beta[:, ti] * (v[:, ti] - pred)
        s = s + k[:, ti].unsqueeze(-1) * u.unsqueeze(-2)
    assert err[1] < 0.1 * err[0]  # second write sees a much smaller error

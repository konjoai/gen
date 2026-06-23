"""Tests for the paired Wilcoxon signed-rank gate, including a known exact case."""

from __future__ import annotations

import numpy as np

from gen.stats.wilcoxon import wilcoxon_signed_rank


def test_known_exact_all_positive() -> None:
    # n=6 distinct positive diffs: W- = 0, T = 0. Exactly 2 of 2^6 sign
    # assignments give T<=0 (all-+ and all--), so two-sided p = 2/64 = 0.03125.
    diffs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    res = wilcoxon_signed_rank(diffs)
    assert res.method == "exact"
    assert res.n_effective == 6
    assert res.statistic == 0.0
    assert abs(res.p_value - 2.0 / 64.0) < 1e-12
    assert res.significant()


def test_no_difference_is_not_significant() -> None:
    x = np.arange(10, dtype=float)
    res = wilcoxon_signed_rank(x, x)
    assert res.n_effective == 0
    assert res.p_value == 1.0
    assert not res.significant()


def test_paired_strong_effect_significant() -> None:
    # candidate beats baseline on every one of 30 paired seeds
    rng = np.random.default_rng(0)
    baseline = rng.uniform(0.5, 0.6, size=30)
    candidate = baseline + rng.uniform(0.05, 0.1, size=30)
    res = wilcoxon_signed_rank(candidate, baseline)
    assert res.significant()
    assert res.n_effective == 30


def test_symmetric_noise_not_significant() -> None:
    rng = np.random.default_rng(1)
    diffs = rng.normal(0.0, 1.0, size=40)
    res = wilcoxon_signed_rank(diffs)
    assert not res.significant()


def test_ties_handled() -> None:
    # repeated magnitudes exercise the tie correction without error
    diffs = np.array([1.0, 1.0, 1.0, -1.0, 2.0, 2.0])
    res = wilcoxon_signed_rank(diffs)
    assert 0.0 <= res.p_value <= 1.0

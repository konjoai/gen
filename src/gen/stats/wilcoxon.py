"""The house statistical gate: paired Wilcoxon signed-rank, p < 0.05 over 30 runs.

Charter §4 step 6: "30-run paired Wilcoxon at p<0.05 before any claim merges."
Used later to compare a candidate mixer against a baseline across paired seeds.
Self-contained (no scipy): exact distribution for small n, normal approximation
with continuity + tie corrections otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product

import numpy as np

EXACT_MAX_N = 20  # 2^20 ≈ 1e6 sign assignments — cheap enough for an exact null
HOUSE_ALPHA = 0.05


@dataclass(frozen=True)
class WilcoxonResult:
    statistic: float  # T = min(W+, W-)
    p_value: float  # two-sided
    n_effective: int  # non-zero pairs
    method: str  # "exact" | "normal-approx"

    def significant(self, alpha: float = HOUSE_ALPHA) -> bool:
        return self.p_value < alpha


def _ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks (1-based), ties share the mean rank."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_vals = values[order]
    i = 0
    n = len(values)
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def _exact_two_sided_p(abs_ranks: np.ndarray, t_obs: float) -> float:
    n = len(abs_ranks)
    total = float(abs_ranks.sum())
    count = 0
    n_assign = 0
    for signs in product((0, 1), repeat=n):
        n_assign += 1
        w_plus = float(sum(r for r, s in zip(abs_ranks, signs, strict=True) if s))
        t = min(w_plus, total - w_plus)
        if t <= t_obs:
            count += 1
    return count / n_assign


def wilcoxon_signed_rank(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float] | None = None,
) -> WilcoxonResult:
    """Two-sided paired Wilcoxon signed-rank test.

    Pass paired samples ``x`` and ``y`` (e.g. candidate vs baseline scores over
    matched seeds), or a single array of differences ``x`` with ``y=None``.
    Zero differences are discarded (Wilcoxon convention).
    """
    x_arr = np.asarray(x, dtype=np.float64)
    if y is not None:
        y_arr = np.asarray(y, dtype=np.float64)
        if x_arr.shape != y_arr.shape:
            raise ValueError(f"paired inputs must match: {x_arr.shape} vs {y_arr.shape}")
        diff = x_arr - y_arr
    else:
        diff = x_arr
    if diff.ndim != 1:
        raise ValueError(f"expected 1-D input, got shape {diff.shape}")

    nonzero = diff[diff != 0.0]
    n = len(nonzero)
    if n == 0:
        # no differences at all — no evidence against the null
        return WilcoxonResult(statistic=0.0, p_value=1.0, n_effective=0, method="degenerate")

    abs_ranks = _ranks(np.abs(nonzero))
    w_plus = float(abs_ranks[nonzero > 0].sum())
    w_minus = float(abs_ranks[nonzero < 0].sum())
    t_obs = min(w_plus, w_minus)

    if n <= EXACT_MAX_N:
        p = _exact_two_sided_p(abs_ranks, t_obs)
        return WilcoxonResult(statistic=t_obs, p_value=min(1.0, p), n_effective=n, method="exact")

    # Normal approximation with tie + continuity correction.
    mean = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0
    _, counts = np.unique(np.abs(nonzero), return_counts=True)
    tie_term = float((counts**3 - counts).sum()) / 48.0
    var -= tie_term
    if var <= 0:
        return WilcoxonResult(statistic=t_obs, p_value=1.0, n_effective=n, method="normal-approx")
    z = (t_obs - mean + 0.5) / math.sqrt(var)  # continuity correction toward the mean
    p = 2.0 * 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))  # 2 * Phi(z), z <= 0 for T=min
    return WilcoxonResult(
        statistic=t_obs, p_value=min(1.0, p), n_effective=n, method="normal-approx"
    )

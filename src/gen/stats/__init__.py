"""Statistical gates."""

from __future__ import annotations

from gen.stats.wilcoxon import HOUSE_ALPHA, WilcoxonResult, wilcoxon_signed_rank

__all__ = ["HOUSE_ALPHA", "WilcoxonResult", "wilcoxon_signed_rank"]

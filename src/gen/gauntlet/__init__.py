"""The synthetic gauntlet: harness, tasks, report."""

from __future__ import annotations

from gen.gauntlet.harness import TaskResult, TrainConfig, run_task
from gen.gauntlet.report import GauntletReport, render_markdown, run_gauntlet

__all__ = [
    "GauntletReport",
    "TaskResult",
    "TrainConfig",
    "render_markdown",
    "run_gauntlet",
    "run_task",
]

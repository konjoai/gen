"""Run the full gauntlet for one mixer and render a capability profile + verdict.

The verdict is descriptive, not a promotion decision: it summarizes which
capabilities a mixer demonstrates, with MQAR (the accuracy-vs-K curve) as the
load-bearing axis. The ledger consumes the structured summary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

from gen import env_header
from gen.gauntlet.harness import TaskResult, TrainConfig, run_task
from gen.gauntlet.tasks import formal_languages, induction, mqar, selective_copy
from gen.primitives import MixerFactory, get_mixer

PASS_THRESHOLD = 0.90  # masked-accuracy bar for "demonstrates the capability"


@dataclass(frozen=True)
class GauntletReport:
    mixer_name: str
    env: dict[str, str]
    results: list[TaskResult]
    seed: int
    mqar_k_range: tuple[int, ...]
    extra: dict[str, object] = field(default_factory=dict)

    def by_name(self, name: str) -> TaskResult:
        for r in self.results:
            if r.task == name:
                return r
        raise KeyError(f"no result for task {name!r}")

    def mqar_curve(self) -> list[tuple[int, float]]:
        curve = [
            (int(cast(int, r.metadata["K"])), r.accuracy)
            for r in self.results
            if r.metadata.get("family") == "mqar"
        ]
        return sorted(curve)

    def capability_summary(self) -> dict[str, float]:
        """Single accuracy per capability family (MQAR collapsed to worst-K)."""
        summary: dict[str, float] = {}
        mqar_accs = [r.accuracy for r in self.results if r.metadata.get("family") == "mqar"]
        if mqar_accs:
            summary["mqar_min"] = min(mqar_accs)
            summary["mqar_max"] = max(mqar_accs)
        for r in self.results:
            fam = r.metadata.get("family")
            if fam in ("induction", "selective_copy"):
                summary[r.task] = r.accuracy
            elif fam == "formal":
                summary[f"formal_{r.metadata['lang']}"] = r.accuracy
        return summary

    def verdict(self) -> str:
        curve = self.mqar_curve()
        if not curve:
            return "incomplete (no MQAR runs)"
        worst = min(acc for _, acc in curve)
        best = max(acc for _, acc in curve)
        if worst >= PASS_THRESHOLD:
            return "aces-mqar (recall holds across K)"
        if best >= PASS_THRESHOLD:
            return "recall-limited (MQAR degrades as K grows)"
        return "recall-incapable (fails MQAR even at low K)"


def run_gauntlet(
    mixer_name: str,
    factory: MixerFactory | None = None,
    *,
    train_cfg: TrainConfig | None = None,
    seed: int = 0,
    mqar_k_range: tuple[int, ...] = mqar.DEFAULT_K_RANGE,
    device: str | None = None,
    quick: bool = False,
) -> GauntletReport:
    """Run every task on a mixer. `quick=True` shrinks it for CI sanity checks."""
    fac = factory or get_mixer(mixer_name)
    cfg = train_cfg or (TrainConfig(steps=300, eval_every=100) if quick else TrainConfig())
    k_range = (1, 8) if quick else mqar_k_range

    specs = [induction.build()]
    specs += mqar.build(k_range=k_range)
    specs.append(selective_copy.build())
    specs += formal_languages.build()

    results = [run_task(fac, spec, train_cfg=cfg, seed=seed, device=device) for spec in specs]
    return GauntletReport(
        mixer_name=mixer_name,
        env=env_header(seed=seed),
        results=results,
        seed=seed,
        mqar_k_range=tuple(k_range),
    )


def render_markdown(report: GauntletReport) -> str:
    lines: list[str] = []
    lines.append(f"## Mixer: `{report.mixer_name}`")
    lines.append("")
    lines.append(f"**Verdict:** {report.verdict()}")
    lines.append("")
    lines.append("Environment / versions:")
    lines.append("")
    lines.append("| key | value |")
    lines.append("| --- | --- |")
    for k, v in report.env.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")

    lines.append("### MQAR accuracy-vs-K")
    lines.append("")
    lines.append("| K | accuracy |")
    lines.append("| --- | --- |")
    for k_val, acc in report.mqar_curve():
        lines.append(f"| {k_val} | {acc:.3f} |")
    lines.append("")

    lines.append("### Capability profile")
    lines.append("")
    lines.append("| task | accuracy | steps | recurrent step |")
    lines.append("| --- | --- | --- | --- |")
    for r in report.results:
        lines.append(
            f"| {r.task} | {r.accuracy:.3f} | {r.steps_run} | {'yes' if r.has_recurrent_step else 'no'} |"
        )
    lines.append("")
    return "\n".join(lines)

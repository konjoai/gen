"""Train+eval a mixer on one task spec, return a score. Deterministic.

Given a mixer factory and a `TaskSpec`, build the tiny harness model, train to
(near-)convergence on freshly generated synthetic batches, evaluate on held-out
batches, and return masked-position accuracy. Minutes on CPU/MPS by design.

Fail loud: any NaN/Inf loss or shape mismatch raises; nothing is swallowed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from gen import set_determinism
from gen.gauntlet.tasks import TaskSpec
from gen.model import GenModel, ModelConfig
from gen.primitives import MixerFactory


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 3000
    batch_size: int = 64
    eval_batches: int = 8
    lr: float = 2e-3
    weight_decay: float = 0.01
    d_model: int = 128
    n_layers: int = 2
    d_ff: int = 256
    warmup_frac: float = 0.1
    # stop early once eval accuracy clears this (saves time on easy/aced tasks)
    early_stop_acc: float = 0.99
    eval_every: int = 300
    grad_clip: float = 1.0


@dataclass(frozen=True)
class TaskResult:
    task: str
    accuracy: float
    final_loss: float
    steps_run: int
    has_recurrent_step: bool
    metadata: dict[str, object]


def _masked_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    vocab = logits.size(-1)
    flat_logits = logits.reshape(-1, vocab)
    flat_targets = targets.reshape(-1)
    flat_mask = mask.reshape(-1)
    sel_logits = flat_logits[flat_mask]
    sel_targets = flat_targets[flat_mask]
    return nn.functional.cross_entropy(sel_logits, sel_targets)


@torch.no_grad()
def _evaluate(model: GenModel, spec: TaskSpec, cfg: TrainConfig, gen: torch.Generator) -> float:
    model.eval()
    correct = 0
    total = 0
    for _ in range(cfg.eval_batches):
        batch = spec.generate(cfg.batch_size, gen)
        logits = model(batch.inputs)
        pred = logits.argmax(dim=-1)
        m = batch.mask
        correct += int((pred[m] == batch.targets[m]).sum().item())
        total += int(m.sum().item())
    if total == 0:
        raise RuntimeError(f"task {spec.name!r} produced no scored positions during eval")
    return correct / total


def run_task(
    mixer_factory: MixerFactory,
    spec: TaskSpec,
    train_cfg: TrainConfig | None = None,
    seed: int = 0,
    device: str | None = None,
) -> TaskResult:
    cfg = train_cfg or TrainConfig()
    set_determinism(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model_cfg = ModelConfig(
        vocab_size=spec.vocab_size,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        d_ff=cfg.d_ff,
        max_seq_len=max(spec.seq_len, 8),
    )
    model = GenModel(model_cfg, mixer_factory).to(dev)
    has_step = model.has_recurrent_mixers()

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    warmup = max(1, int(cfg.steps * cfg.warmup_frac))

    def lr_at(step: int) -> float:
        if step < warmup:
            return cfg.lr * (step + 1) / warmup
        # cosine decay to 10% of lr
        prog = (step - warmup) / max(1, cfg.steps - warmup)
        import math

        return cfg.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))

    train_gen = torch.Generator().manual_seed(seed + 1)
    eval_gen = torch.Generator().manual_seed(seed + 12345)

    model.train()
    final_loss = float("nan")
    steps_run = cfg.steps
    for step in range(cfg.steps):
        for group in opt.param_groups:
            group["lr"] = lr_at(step)
        batch = spec.generate(cfg.batch_size, train_gen)
        batch_inputs = batch.inputs.to(dev)
        logits = model(batch_inputs)
        loss = _masked_loss(logits, batch.targets.to(dev), batch.mask.to(dev))
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"non-finite loss at step {step} on task {spec.name!r}: {loss.item()}"
            )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        final_loss = float(loss.item())

        if (step + 1) % cfg.eval_every == 0:
            acc = _evaluate(model, spec, cfg, eval_gen)
            model.train()
            if acc >= cfg.early_stop_acc:
                steps_run = step + 1
                break

    accuracy = _evaluate(model, spec, cfg, eval_gen)
    return TaskResult(
        task=spec.name,
        accuracy=accuracy,
        final_loss=final_loss,
        steps_run=steps_run,
        has_recurrent_step=has_step,
        metadata=dict(spec.metadata),
    )

"""Synthetic gauntlet tasks: self-contained generators + scorers, cheapest first.

Each task exposes builder(s) returning `TaskSpec`s. A `TaskSpec` is a single
trainable probe: a deterministic batch generator plus the metadata needed to
size a tiny model. The harness trains one model per spec and scores accuracy on
the masked positions. Conventions shared by every task:

  * ``logits[t]`` predicts ``targets[t]`` (causal: position t has seen tokens 0..t).
  * ``mask[t]`` marks the positions that are trained and scored; everything else
    is ignored by both loss and accuracy.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass(frozen=True)
class Batch:
    """One synthetic batch. All tensors are ``(B, T)``; mask/targets align to inputs."""

    inputs: Tensor  # long (B, T)
    targets: Tensor  # long (B, T)
    mask: Tensor  # bool (B, T) — positions scored / trained

    def __post_init__(self) -> None:
        if not (self.inputs.shape == self.targets.shape == self.mask.shape):
            raise ValueError(
                f"batch shape mismatch: inputs {tuple(self.inputs.shape)} "
                f"targets {tuple(self.targets.shape)} mask {tuple(self.mask.shape)}"
            )
        if self.inputs.dim() != 2:
            raise ValueError(f"batch tensors must be (B, T), got {tuple(self.inputs.shape)}")
        if not self.mask.any():
            raise ValueError("batch has no scored positions (empty mask)")


GenFn = Callable[[int, torch.Generator], Batch]


@dataclass(frozen=True)
class TaskSpec:
    """A single trainable probe in the gauntlet."""

    name: str
    vocab_size: int
    seq_len: int
    generate: GenFn
    metadata: dict[str, object] = field(default_factory=dict)

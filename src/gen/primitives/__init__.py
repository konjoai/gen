"""Sequence-mixing primitives: the plug interface and the reference baselines."""

from __future__ import annotations

from collections.abc import Callable

from gen.primitives.attention_ref import AttentionRef
from gen.primitives.base import (
    NO_RECURRENT_FORM,
    GateCard,
    GateVerdict,
    SequenceMixer,
)
from gen.primitives.ssm_ref import SSMRef

# A mixer factory takes d_model and returns a SequenceMixer. The harness and
# scripts look candidates up by name here; future candidates register the same way.
MixerFactory = Callable[[int], SequenceMixer]

REGISTRY: dict[str, MixerFactory] = {
    "attention": lambda d_model: AttentionRef(d_model),
    "ssm": lambda d_model: SSMRef(d_model),
}


def get_mixer(name: str) -> MixerFactory:
    if name not in REGISTRY:
        raise KeyError(f"unknown mixer {name!r}; registered: {sorted(REGISTRY)}")
    return REGISTRY[name]


__all__ = [
    "NO_RECURRENT_FORM",
    "REGISTRY",
    "AttentionRef",
    "GateCard",
    "GateVerdict",
    "MixerFactory",
    "SSMRef",
    "SequenceMixer",
    "get_mixer",
]

"""The tiny harness model: ``embed -> [mixer block x L] -> norm -> head``.

A block is pre-norm: ``x += mixer(norm(x)); x += ffn(norm(x))``. The mixer is the
pluggable sequence-mixing slot (charter §1.1); the FFN is channel mixing (§1.2);
residuals + RMSNorm cover gradient flow (§1.6); the embedding carries order
information additively (§1.8). Everything here is fixed reference scaffolding so
that the *only* thing that varies between runs is the mixer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from gen.primitives import MixerFactory
from gen.primitives.base import SequenceMixer


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    d_model: int = 128
    n_layers: int = 2
    d_ff: int = 256
    max_seq_len: int = 256
    dropout: float = 0.0


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: Tensor) -> Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


class FFN(nn.Module):
    """Channel mixing: GELU MLP (charter §1.2)."""

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.drop(torch.nn.functional.gelu(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, mixer: SequenceMixer, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.mixer = mixer
        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn = FFN(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.mixer(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class GenModel(nn.Module):
    """Embed -> blocks -> norm -> head. Causal next-token logits."""

    def __init__(self, cfg: ModelConfig, mixer_factory: MixerFactory) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(
            [Block(mixer_factory(cfg.d_model), cfg) for _ in range(cfg.n_layers)]
        )
        self.norm_f = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.dim() != 2:
            raise ValueError(f"expected (B, T) token ids, got {tuple(tokens.shape)}")
        b, t = tokens.shape
        if t > self.cfg.max_seq_len:
            raise ValueError(f"seq len {t} exceeds max_seq_len {self.cfg.max_seq_len}")
        pos = torch.arange(t, device=tokens.device)
        x = self.tok_emb(tokens) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        logits = self.head(x)  # (B, T, vocab)
        if logits.shape != (b, t, self.cfg.vocab_size):
            raise AssertionError(f"bad logits shape {tuple(logits.shape)}")
        return logits

    def has_recurrent_mixers(self) -> bool:
        """True iff every mixer exposes a recurrent `step` path."""
        from gen.primitives.base import NO_RECURRENT_FORM

        dummy_state = None
        for block in self.blocks:
            probe = block.mixer.step(torch.zeros(1, self.cfg.d_model), dummy_state)
            if probe is NO_RECURRENT_FORM:
                return False
        return True

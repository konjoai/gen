"""Induction task: repeated-bigram copy (the minimum in-context-copy test).

After ``... [a][b] ... [a]``, predict ``[b]``. Concretely: at every position ``t``
whose token last occurred at ``j`` (with ``j + 1 < t``), the target is the token
that followed that previous occurrence, ``seq[j + 1]``. Those are the scored
positions. This is the induction-head capability (charter §1 constraint 3); both
attention and a competent recurrence should pass it.
"""

from __future__ import annotations

import torch

from gen.gauntlet.tasks import Batch, TaskSpec


def build(vocab_size: int = 16, seq_len: int = 64) -> TaskSpec:
    def generate(batch_size: int, gen: torch.Generator) -> Batch:
        inputs = torch.randint(0, vocab_size, (batch_size, seq_len), generator=gen)
        targets = torch.zeros_like(inputs)
        mask = torch.zeros_like(inputs, dtype=torch.bool)
        # last_pos[b, v] = most recent index where token v appeared in row b
        last_pos = torch.full((batch_size, vocab_size), -1, dtype=torch.long)
        for t in range(seq_len):
            tok = inputs[:, t]  # (B,)
            prev = last_pos.gather(1, tok.unsqueeze(1)).squeeze(1)  # (B,)
            valid = (prev >= 0) & (prev + 1 < t)
            follower_idx = (prev + 1).clamp(min=0)
            follower = inputs.gather(1, follower_idx.unsqueeze(1)).squeeze(1)
            targets[:, t] = torch.where(valid, follower, torch.zeros_like(follower))
            mask[:, t] = valid
            last_pos.scatter_(1, tok.unsqueeze(1), torch.full_like(tok.unsqueeze(1), t))
        return Batch(inputs=inputs, targets=targets, mask=mask)

    return TaskSpec(
        name="induction",
        vocab_size=vocab_size,
        seq_len=seq_len,
        generate=generate,
        metadata={"family": "induction"},
    )

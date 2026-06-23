"""Selective-copy task: the Mamba selectivity probe.

A long stream of blank/filler tokens with a few content tokens scattered at
random positions; after a delimiter, the model must emit the content tokens in
their original order. This rewards input-dependent selection (charter §2.3,
Mamba selectivity) and should be passable by both baselines.

Layout (per row):
    [stream of length L with n_content data tokens among blanks] [DELIM] [n_content output slots]
Targets sit at the output slots; output-slot inputs are the DELIM/last-content
carried token but only positions are scored.
"""

from __future__ import annotations

import torch

from gen.gauntlet.tasks import Batch, TaskSpec


def build(
    num_symbols: int = 16,
    n_content: int = 8,
    stream_len: int = 48,
) -> TaskSpec:
    blank_id = num_symbols
    delim_id = num_symbols + 1
    vocab_size = num_symbols + 2
    seq_len = stream_len + 1 + n_content  # stream + delim + output slots

    def generate(batch_size: int, gen: torch.Generator) -> Batch:
        inputs = torch.full((batch_size, seq_len), blank_id, dtype=torch.long)
        targets = torch.zeros((batch_size, seq_len), dtype=torch.long)
        mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
        for b in range(batch_size):
            positions = torch.randperm(stream_len, generator=gen)[:n_content].sort().values
            symbols = torch.randint(0, num_symbols, (n_content,), generator=gen)
            inputs[b, positions] = symbols
            inputs[b, stream_len] = delim_id
            out_start = stream_len + 1
            # output slot inputs are blanks; targets are the content in order
            targets[b, out_start : out_start + n_content] = symbols
            mask[b, out_start : out_start + n_content] = True
        return Batch(inputs=inputs, targets=targets, mask=mask)

    return TaskSpec(
        name="selective_copy",
        vocab_size=vocab_size,
        seq_len=seq_len,
        generate=generate,
        metadata={"family": "selective_copy", "n_content": n_content},
    )

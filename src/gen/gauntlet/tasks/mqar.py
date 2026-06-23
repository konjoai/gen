"""MQAR: multi-query associative recall — the load-bearing gauntlet test.

Insert ``K`` key->value pairs, optionally a gap of filler, then re-present the
keys in shuffled order; at each query position the model must emit that key's
value. The headline output is the **accuracy-vs-K curve**: it is what separates a
real mixer from a toy (charter §4). A primitive that cannot pass MQAR will not
model language.

Layout (per row), all ids in one space:
    [k1 v1 k2 v2 ... kK vK] [<blank> * gap] [q1 q2 ... qK]
Targets sit at the query positions; everything else is masked out.
"""

from __future__ import annotations

import torch

from gen.gauntlet.tasks import Batch, TaskSpec

DEFAULT_K_RANGE: tuple[int, ...] = (1, 2, 4, 8, 16, 32)


def _vocab_layout(num_keys: int, num_values: int) -> tuple[int, int, int]:
    """Return (key_base, value_base, blank_id) for a disjoint id layout."""
    key_base = 0
    value_base = num_keys
    blank_id = num_keys + num_values
    return key_base, value_base, blank_id


def build_one(
    k: int,
    gap: int = 4,
    num_keys: int = 64,
    num_values: int = 16,
) -> TaskSpec:
    if k > num_keys:
        raise ValueError(f"K={k} exceeds num_keys={num_keys}; not enough distinct keys")
    key_base, value_base, blank_id = _vocab_layout(num_keys, num_values)
    vocab_size = num_keys + num_values + 1
    seq_len = 2 * k + gap + k

    def generate(batch_size: int, gen: torch.Generator) -> Batch:
        inputs = torch.full((batch_size, seq_len), blank_id, dtype=torch.long)
        targets = torch.zeros((batch_size, seq_len), dtype=torch.long)
        mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
        for b in range(batch_size):
            keys = torch.randperm(num_keys, generator=gen)[:k]
            values = torch.randint(0, num_values, (k,), generator=gen) + value_base
            keys_tok = keys + key_base
            # KV block
            inputs[b, 0 : 2 * k : 2] = keys_tok
            inputs[b, 1 : 2 * k : 2] = values
            # queries: keys in shuffled order, after the gap
            order = torch.randperm(k, generator=gen)
            q_start = 2 * k + gap
            inputs[b, q_start : q_start + k] = keys_tok[order]
            targets[b, q_start : q_start + k] = values[order]
            mask[b, q_start : q_start + k] = True
        return Batch(inputs=inputs, targets=targets, mask=mask)

    return TaskSpec(
        name=f"mqar_K{k}",
        vocab_size=vocab_size,
        seq_len=seq_len,
        generate=generate,
        metadata={"family": "mqar", "K": k, "gap": gap},
    )


def build(
    k_range: tuple[int, ...] = DEFAULT_K_RANGE,
    gap: int = 4,
    num_keys: int = 64,
    num_values: int = 16,
) -> list[TaskSpec]:
    return [build_one(k, gap=gap, num_keys=num_keys, num_values=num_values) for k in k_range]

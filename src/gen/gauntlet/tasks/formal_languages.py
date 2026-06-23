"""Formal-language probes for a Chomsky-hierarchy expressivity profile.

Three self-contained probes (charter §4):
  * parity   — regular: cumulative XOR of a bit stream (predict running parity).
  * mod_add  — algebraic: predict (a + b) mod p at the '=' position.
  * dyck      — context-free: accept/reject Dyck-n bracket balance (needs a stack).

Each returns a `TaskSpec`; `build()` returns all three.
"""

from __future__ import annotations

import torch

from gen.gauntlet.tasks import Batch, TaskSpec


def build_parity(seq_len: int = 48) -> TaskSpec:
    # tokens: 0/1 bits; labels reuse ids 0/1 (even/odd). vocab = 2.
    vocab_size = 2

    def generate(batch_size: int, gen: torch.Generator) -> Batch:
        bits = torch.randint(0, 2, (batch_size, seq_len), generator=gen)
        running = torch.cumsum(bits, dim=1) % 2
        mask = torch.ones_like(bits, dtype=torch.bool)
        return Batch(inputs=bits, targets=running, mask=mask)

    return TaskSpec("parity", vocab_size, seq_len, generate, {"family": "formal", "lang": "parity"})


def build_mod_add(p: int = 13) -> TaskSpec:
    # layout [a, b, =]; predict (a+b) mod p at '='. ids: 0..p-1 operands, p = '='.
    eq_id = p
    vocab_size = p + 1
    seq_len = 3

    def generate(batch_size: int, gen: torch.Generator) -> Batch:
        a = torch.randint(0, p, (batch_size,), generator=gen)
        b = torch.randint(0, p, (batch_size,), generator=gen)
        inputs = torch.stack([a, b, torch.full_like(a, eq_id)], dim=1)
        targets = torch.zeros_like(inputs)
        mask = torch.zeros_like(inputs, dtype=torch.bool)
        targets[:, 2] = (a + b) % p
        mask[:, 2] = True
        return Batch(inputs=inputs, targets=targets, mask=mask)

    return TaskSpec(
        "mod_add", vocab_size, seq_len, generate, {"family": "formal", "lang": "mod_add", "p": p}
    )


def build_dyck(n_types: int = 2, seq_len: int = 32) -> TaskSpec:
    # bracket ids: open types 0..n-1, close types n..2n-1. Then QUERY, then label.
    # We score the label position: YES if the bracket prefix is balanced else NO.
    open_base = 0
    close_base = n_types
    query_id = 2 * n_types
    no_id = 2 * n_types + 1
    yes_id = 2 * n_types + 2
    vocab_size = 2 * n_types + 3
    bracket_len = seq_len  # number of bracket tokens before the query
    full_len = bracket_len + 2  # brackets + QUERY + label slot

    def _balanced_sequence(gen: torch.Generator) -> list[int]:
        # generate a random balanced Dyck-n string of length bracket_len (even)
        length = bracket_len if bracket_len % 2 == 0 else bracket_len - 1
        seq: list[int] = []
        stack: list[int] = []
        for _ in range(length):
            can_close = len(stack) > 0
            can_open = len(seq) + len(stack) < length  # leave room to close all
            must_close = (length - len(seq)) <= len(stack)
            if must_close or (can_close and not can_open):
                seq.append(close_base + stack.pop())
            elif can_open and (torch.rand(1, generator=gen).item() < 0.5 or not can_close):
                t = int(torch.randint(0, n_types, (1,), generator=gen).item())
                stack.append(t)
                seq.append(open_base + t)
            else:
                seq.append(close_base + stack.pop())
        # pad to bracket_len with matched pairs if we trimmed for oddness
        while len(seq) < bracket_len:
            seq.append(open_base)  # will be corrupted into imbalance only if chosen; handled below
        return seq

    def generate(batch_size: int, gen: torch.Generator) -> Batch:
        inputs = torch.zeros((batch_size, full_len), dtype=torch.long)
        targets = torch.zeros((batch_size, full_len), dtype=torch.long)
        mask = torch.zeros((batch_size, full_len), dtype=torch.bool)
        for b in range(batch_size):
            seq = _balanced_sequence(gen)[:bracket_len]
            balanced = True
            if torch.rand(1, generator=gen).item() < 0.5:
                # corrupt: flip one bracket to break balance (open<->close)
                idx = int(torch.randint(0, bracket_len, (1,), generator=gen).item())
                tok = seq[idx]
                seq[idx] = (tok + n_types) % (2 * n_types)
                # verify it actually became imbalanced; recompute
                balanced = _is_balanced(seq, n_types)
            row = torch.tensor(seq + [query_id, no_id], dtype=torch.long)
            inputs[b] = row
            targets[b, full_len - 1] = yes_id if balanced else no_id
            mask[b, full_len - 1] = True
        return Batch(inputs=inputs, targets=targets, mask=mask)

    return TaskSpec(
        "dyck",
        vocab_size,
        full_len,
        generate,
        {"family": "formal", "lang": "dyck", "n_types": n_types},
    )


def _is_balanced(seq: list[int], n_types: int) -> bool:
    stack: list[int] = []
    for tok in seq:
        if tok < n_types:  # open
            stack.append(tok)
        else:  # close
            if not stack or stack.pop() != tok - n_types:
                return False
    return not stack


def build() -> list[TaskSpec]:
    return [build_parity(), build_mod_add(), build_dyck()]

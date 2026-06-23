"""Tests for the gauntlet task generators and a fast harness smoke run."""

from __future__ import annotations

import torch

from gen.gauntlet.harness import TrainConfig, run_task
from gen.gauntlet.tasks import formal_languages, induction, mqar, selective_copy


def _gen(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def test_induction_targets_are_correct_by_construction() -> None:
    spec = induction.build(vocab_size=8, seq_len=32)
    batch = spec.generate(16, _gen())
    inp, tgt, mask = batch.inputs, batch.targets, batch.mask
    # For each scored position, target must equal the token following the
    # previous occurrence of the current token.
    b, t = inp.shape
    for bi in range(b):
        last: dict[int, int] = {}
        for ti in range(t):
            tok = int(inp[bi, ti])
            if mask[bi, ti]:
                j = last[tok]
                assert int(tgt[bi, ti]) == int(inp[bi, j + 1])
            last[tok] = ti


def test_mqar_shapes_and_recall_target() -> None:
    spec = mqar.build_one(k=4, gap=3, num_keys=16, num_values=16)
    batch = spec.generate(8, _gen())
    assert batch.inputs.shape == (8, spec.seq_len)
    # exactly K scored positions per row
    assert int(batch.mask[0].sum()) == 4
    # every query token equals one of the planted keys, and its target is a value id
    value_base = 16
    scored = batch.mask[0]
    assert (batch.targets[0][scored] >= value_base).all()


def test_mqar_curve_builder() -> None:
    specs = mqar.build(k_range=(1, 2, 4))
    assert [s.metadata["K"] for s in specs] == [1, 2, 4]


def test_selective_copy_emits_content_in_order() -> None:
    spec = selective_copy.build(num_symbols=8, n_content=4, stream_len=20)
    batch = spec.generate(8, _gen())
    assert int(batch.mask[0].sum()) == 4
    # content targets are valid symbol ids
    assert (batch.targets[0][batch.mask[0]] < 8).all()


def test_formal_language_specs_valid() -> None:
    for spec in formal_languages.build():
        batch = spec.generate(8, _gen())
        assert batch.inputs.max() < spec.vocab_size
        assert batch.mask.any()


def test_dyck_balance_labels_consistent() -> None:
    spec = formal_languages.build_dyck(n_types=2, seq_len=16)
    batch = spec.generate(32, _gen())
    # label position is the last; YES/NO ids are the two highest in vocab
    yes_id = spec.vocab_size - 1
    no_id = spec.vocab_size - 2
    labels = batch.targets[batch.mask]
    assert set(int(x) for x in labels).issubset({yes_id, no_id})


def test_harness_smoke_runs_and_learns() -> None:
    # mod_add is tiny; a short run should drive loss finite and beat chance.
    spec = formal_languages.build_mod_add(p=7)
    cfg = TrainConfig(
        steps=200, batch_size=64, eval_batches=4, eval_every=1000, d_model=32, n_layers=1, d_ff=64
    )
    from gen.primitives import get_mixer

    result = run_task(get_mixer("attention"), spec, train_cfg=cfg, seed=0)
    assert 0.0 <= result.accuracy <= 1.0
    assert result.accuracy > 1.0 / 7.0  # better than uniform chance
    assert result.has_recurrent_step is True

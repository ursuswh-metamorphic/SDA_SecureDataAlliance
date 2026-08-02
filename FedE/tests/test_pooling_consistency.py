"""Pooling rule regression tests shared by non-DP and DP training."""

import torch

from flgo.benchmark.fedrag_classification import core


def test_cls_pooling_uses_first_token_only():
    hidden = torch.tensor(
        [[[1.0, 2.0], [9.0, 9.0]], [[3.0, 4.0], [8.0, 8.0]]]
    )
    inputs = {"attention_mask": torch.tensor([[1, 1], [1, 1]])}
    previous = core.DEFAULT_POOLING
    try:
        core.DEFAULT_POOLING = "cls"
        pooled = core.pool_embeddings(hidden, inputs)
    finally:
        core.DEFAULT_POOLING = previous
    assert torch.equal(pooled, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))


def test_masked_mean_is_invariant_to_extra_padding():
    original = torch.tensor([[[2.0, 4.0], [4.0, 8.0]]])
    padded = torch.tensor(
        [[[2.0, 4.0], [4.0, 8.0], [100.0, 100.0]]]
    )
    previous = core.DEFAULT_POOLING
    try:
        core.DEFAULT_POOLING = "masked_mean"
        first = core.pool_embeddings(
            original, {"attention_mask": torch.tensor([[1, 1]])}
        )
        second = core.pool_embeddings(
            padded, {"attention_mask": torch.tensor([[1, 1, 0]])}
        )
    finally:
        core.DEFAULT_POOLING = previous
    assert torch.equal(first, second)

import math

import pytest
import torch

from client_dp_utils import (
    add_gaussian_noise,
    clip_state_delta,
    replacement_sensitivity,
    state_delta,
    state_l2_norm,
    weighted_sum_states,
)


def test_multitensor_delta_norm_and_clipping_closed_form():
    global_state = {
        "a": torch.tensor([1.0, 2.0]),
        "b": torch.tensor([3.0]),
    }
    client_state = {
        "a": torch.tensor([4.0, 6.0]),
        "b": torch.tensor([3.0]),
    }
    delta = state_delta(client_state, global_state)
    assert state_l2_norm(delta) == pytest.approx(5.0)

    clipped, norm, coefficient = clip_state_delta(delta, clip_norm=2.0)
    assert norm == pytest.approx(5.0)
    assert coefficient == pytest.approx(0.4)
    assert state_l2_norm(clipped) == pytest.approx(2.0)


def test_weighted_delta_control_matches_sample_weighted_fedavg():
    deltas = [
        {"lora": torch.tensor([1.0])},
        {"lora": torch.tensor([3.0])},
    ]
    aggregate = weighted_sum_states(deltas, weights=[1, 3])
    assert torch.allclose(aggregate["lora"], torch.tensor([2.5]))


def test_replacement_sensitivity_uses_max_public_weight():
    # Normalized weights are 0.25 and 0.75.
    sensitivity = replacement_sensitivity(clip_norm=2.0, weights=[1, 3])
    assert sensitivity == pytest.approx(2.0 * 2.0 * 0.75)


def test_gaussian_noise_is_reproducible_only_with_explicit_test_generator():
    state = {"x": torch.zeros(1024)}
    generator_a = torch.Generator().manual_seed(123)
    generator_b = torch.Generator().manual_seed(123)
    noisy_a = add_gaussian_noise(state, 0.5, generator_a)["x"]
    noisy_b = add_gaussian_noise(state, 0.5, generator_b)["x"]
    assert torch.equal(noisy_a, noisy_b)
    assert noisy_a.std().item() == pytest.approx(0.5, rel=0.1)
    assert math.isfinite(noisy_a.mean().item())


@pytest.mark.parametrize("clip_norm", [0.0, -1.0, float("inf")])
def test_invalid_clip_norm_fails(clip_norm):
    with pytest.raises(ValueError):
        clip_state_delta({"x": torch.ones(1)}, clip_norm)

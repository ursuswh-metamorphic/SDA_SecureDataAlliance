"""Pure tensor primitives for client-level DP aggregation."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch


def validate_matching_states(states: Sequence[Mapping[str, torch.Tensor]]) -> list[str]:
    if not states:
        raise ValueError("at least one state is required")
    keys = list(states[0])
    key_set = set(keys)
    for index, state in enumerate(states[1:], start=1):
        if set(state) != key_set:
            raise ValueError(
                f"state {index} keys differ: "
                f"missing={key_set - set(state)}, extra={set(state) - key_set}"
            )
    return keys


def state_delta(
    client_state: Mapping[str, torch.Tensor],
    global_state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    validate_matching_states([client_state, global_state])
    return {
        key: client_state[key].detach().float().cpu()
        - global_state[key].detach().float().cpu()
        for key in client_state
    }


def state_l2_norm(state: Mapping[str, torch.Tensor]) -> float:
    squared = sum(
        tensor.detach().double().pow(2).sum().item()
        for tensor in state.values()
    )
    return math.sqrt(squared)


def clip_state_delta(
    delta: Mapping[str, torch.Tensor],
    clip_norm: float,
) -> tuple[dict[str, torch.Tensor], float, float]:
    if not math.isfinite(clip_norm) or clip_norm <= 0:
        raise ValueError("clip_norm must be finite and positive")
    norm = state_l2_norm(delta)
    coefficient = min(1.0, clip_norm / (norm + 1e-12))
    return (
        {key: tensor * coefficient for key, tensor in delta.items()},
        norm,
        coefficient,
    )


def normalize_weights(weights: Sequence[float]) -> list[float]:
    if not weights:
        raise ValueError("at least one aggregation weight is required")
    if any((not math.isfinite(weight)) or weight < 0 for weight in weights):
        raise ValueError("aggregation weights must be finite and non-negative")
    total = sum(weights)
    if total <= 0:
        raise ValueError("aggregation weights must sum to a positive value")
    return [weight / total for weight in weights]


def weighted_sum_states(
    states: Sequence[Mapping[str, torch.Tensor]],
    weights: Sequence[float],
) -> dict[str, torch.Tensor]:
    keys = validate_matching_states(states)
    normalized = normalize_weights(weights)
    if len(states) != len(normalized):
        raise ValueError("state and weight counts must match")
    return {
        key: sum(
            state[key].detach().float().cpu() * weight
            for state, weight in zip(states, normalized)
        )
        for key in keys
    }


def replacement_sensitivity(clip_norm: float, weights: Sequence[float]) -> float:
    """L2 sensitivity for replace-one client adjacency.

    Each client delta is clipped to ``clip_norm``. Replacing one clipped
    vector by another can change the weighted aggregate by at most
    ``2 * clip_norm * max(normalized_weight)``.
    """
    if not math.isfinite(clip_norm) or clip_norm <= 0:
        raise ValueError("clip_norm must be finite and positive")
    normalized = normalize_weights(weights)
    return 2.0 * clip_norm * max(normalized)


def add_gaussian_noise(
    state: Mapping[str, torch.Tensor],
    std: float,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    if not math.isfinite(std) or std < 0:
        raise ValueError("noise std must be finite and non-negative")
    if std == 0:
        return {key: tensor.clone() for key, tensor in state.items()}
    return {
        key: tensor + torch.randn(
            tensor.shape,
            dtype=tensor.dtype,
            device=tensor.device,
            generator=generator,
        ) * std
        for key, tensor in state.items()
    }

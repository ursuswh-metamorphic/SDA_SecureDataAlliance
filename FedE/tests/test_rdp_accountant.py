import math

import pytest

from privacy.rdp_accountant import (
    _rdp_to_epsilon_single,
    compute_epsilon,
)


def test_rdp_to_dp_matches_balle_theorem_21_closed_form():
    alpha = 4.0
    rdp = 1.25
    delta = 1e-5
    expected = (
        rdp
        - (math.log(delta) + math.log(alpha)) / (alpha - 1.0)
        + math.log((alpha - 1.0) / alpha)
    )
    assert _rdp_to_epsilon_single(alpha, rdp, delta) == pytest.approx(
        expected, abs=1e-12
    )


def test_full_participation_composition_matches_manual_gaussian_rdp():
    steps = 3
    sigma = 0.9682040405273435
    delta = 1e-5
    orders = [2.0, 3.0, 4.0, 8.0]
    epsilon, best_alpha = compute_epsilon(
        steps,
        sigma,
        sample_rate=1.0,
        delta=delta,
        orders=orders,
    )
    candidates = {
        alpha: (
            steps * alpha / (2.0 * sigma**2)
            - (math.log(delta) + math.log(alpha)) / (alpha - 1.0)
            + math.log((alpha - 1.0) / alpha)
        )
        for alpha in orders
    }
    expected_alpha = min(candidates, key=candidates.get)
    assert best_alpha == expected_alpha
    assert epsilon == pytest.approx(candidates[expected_alpha], abs=1e-12)

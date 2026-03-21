"""
RDP Privacy Accountant
======================
Implements Rényi Differential Privacy (RDP) accounting for DP-SGD
in federated learning, without external DP library dependencies.

Mathematical Foundations:
  - RDP definition: D_α(M(D) || M(D')) ≤ ε_α  (Mironov, CSF 2017)
  - Gaussian mechanism: (α, α/(2σ²))-RDP  (Proposition 3, Mironov 2017)
  - Subsampled Gaussian: amplification via Poisson subsampling (Wang et al., AISTATS 2019)
  - Composition: RDP values ADD over sequential mechanisms
  - RDP → (ε,δ)-DP conversion: ε(δ) = min_α [rdp(α) - log(δ)/(α-1)]

References:
  [1] Mironov, I. (2017). "Rényi Differential Privacy." IEEE CSF 2017. arXiv:1702.07476
  [2] Wang, Y.X. et al. (2019). "Subsampled Rényi DP." AISTATS 2019.
  [3] Gopi, S. et al. (2021). "Numerical Composition of DP." NeurIPS 2021. arXiv:2106.02848
  [4] Abadi, M. et al. (2016). "Deep Learning with DP." CCS 2016. arXiv:1607.00133
"""

from __future__ import annotations
import math
import numpy as np
from typing import List, Optional, Tuple

# ─── Default Rényi orders to sweep over when optimizing ──────────────────────
# Range chosen following TF-Privacy best practice:
#   orders 2-63 cover moderate alpha, 64-512 cover high alpha for small-eps regime
DEFAULT_ORDERS: List[float] = (
    list(range(2, 64))
    + [64, 96, 128, 192, 256, 512]
    + [float("inf")]
)


# ══════════════════════════════════════════════════════════════════════════════
#   Core RDP Formulae
# ══════════════════════════════════════════════════════════════════════════════

def gaussian_rdp(sigma: float, alpha: float) -> float:
    """
    RDP of the Gaussian mechanism (full-batch, no subsampling).

    If f has L2 sensitivity 1, then M(D) = f(D) + N(0, σ²·I) satisfies
    (α, α/(2σ²))-RDP for all α > 1.

    Args:
        sigma:  noise multiplier σ  (actual std = σ · clip_norm)
        alpha:  Rényi order α > 1

    Returns:
        ε_α = α / (2σ²)
    """
    if sigma <= 0:
        return float("inf")
    if alpha == float("inf"):
        return float("inf")
    if alpha == 1:
        # Limit as α→1: KL divergence = 1/(2σ²)
        return 1.0 / (2.0 * sigma ** 2)
    return alpha / (2.0 * sigma ** 2)


def subsampled_gaussian_rdp(sigma: float, alpha: float, q: float) -> float:
    """
    RDP of subsampled Gaussian mechanism (Poisson subsampling with rate q).

    Uses the upper bound from Wang et al. (AISTATS 2019), Corollary 3:

        ε'(α) ≤ log(1 + q² · α · (e^{ε(α)/α} − 1))

    which is tight for small q and moderate σ.
    Additionally, we apply the trivial bound ε'(α) ≤ ε(α) (no amplification).

    Args:
        sigma:  noise multiplier σ
        alpha:  Rényi order α ≥ 2
        q:      subsampling rate ∈ (0, 1]

    Returns:
        Amplified RDP value ε'(α)
    """
    if sigma <= 0:
        return float("inf")
    if q <= 0:
        return 0.0
    if q >= 1.0:
        return gaussian_rdp(sigma, alpha)
    if alpha == float("inf"):
        return float("inf")
    if alpha == 1:
        # Use first-order approximation: q² / (2σ²)
        return (q ** 2) / (2.0 * sigma ** 2)

    rdp_full = gaussian_rdp(sigma, alpha)

    # Wang et al. Corollary 3 (tighter for α ≥ 2)
    try:
        log1p_term = math.log1p(q ** 2 * alpha * (math.exp(rdp_full / alpha) - 1.0))
    except OverflowError:
        log1p_term = rdp_full  # fall back to non-amplified

    # Take minimum of amplified bound and full bound (trivial upper bound)
    return min(rdp_full, log1p_term)


# ══════════════════════════════════════════════════════════════════════════════
#   RDP → (ε, δ)-DP Conversion
# ══════════════════════════════════════════════════════════════════════════════

def _rdp_to_epsilon_single(alpha: float, rdp: float, delta: float) -> float:
    """
    Convert a single (α, ε_α)-RDP guarantee to (ε, δ)-DP.

    Formula (Mironov 2017, Proposition 3):
        ε(δ) = ε_α + log((α−1)/α) − (log δ + log(α−1)) / α

    This is equivalent to the common form:
        ε(δ) = ε_α − log(δ) / (α − 1)  [loose version, used for simplicity]

    We use the tighter Proposition 3 formula.
    """
    if alpha <= 1.0:
        raise ValueError(f"Rényi order α must be > 1, got {alpha}")
    if delta <= 0 or delta >= 1:
        raise ValueError(f"δ must be in (0,1), got {delta}")
    if math.isinf(rdp) or math.isnan(rdp):
        return float("inf")
    if rdp == 0.0:
        return 0.0

    # Proposition 3: ε = ε_α + log((α-1)/α) − (log(δ) + log(α-1)) / α
    # Equivalent to: ε = ε_α - log(δ)/(α-1)  when simplified
    # We implement the tight version:
    try:
        eps = (
            rdp
            + math.log((alpha - 1.0) / alpha)
            - (math.log(delta) + math.log(alpha - 1.0)) / alpha
        )
    except (ValueError, OverflowError):
        eps = float("inf")

    return float(eps)


def rdp_to_dp(
    rdp_values: List[float],
    orders: List[float],
    delta: float,
) -> Tuple[float, float]:
    """
    Convert a list of (α, ε_α)-RDP values to (ε, δ)-DP by minimizing over α.

    Args:
        rdp_values: list of RDP values, one per order
        orders:     corresponding list of Rényi orders
        delta:      target δ

    Returns:
        (epsilon, best_alpha): the minimal ε and the α that achieves it
    """
    assert len(rdp_values) == len(orders), "rdp_values and orders must match in length"

    eps_list: List[Tuple[float, float]] = []
    for alpha, rdp in zip(orders, rdp_values):
        if alpha <= 1.0 or alpha == float("inf"):
            continue
        if math.isinf(rdp) or rdp < 0:
            continue
        try:
            eps = _rdp_to_epsilon_single(alpha, rdp, delta)
            if not math.isnan(eps):
                eps_list.append((eps, alpha))
        except (ValueError, OverflowError):
            continue

    if not eps_list:
        return float("inf"), float("inf")

    best_eps, best_alpha = min(eps_list, key=lambda x: x[0])
    return best_eps, best_alpha


# ══════════════════════════════════════════════════════════════════════════════
#   High-Level Convenience Function
# ══════════════════════════════════════════════════════════════════════════════

def compute_epsilon(
    num_steps: int,
    noise_multiplier: float,
    sample_rate: float,
    delta: float,
    orders: Optional[List[float]] = None,
) -> Tuple[float, float]:
    """
    Compute the privacy guarantee (ε, δ) for DP-SGD / DP-FedAvg.

    Algorithm:
        1. Compose RDP over num_steps via the subsampled Gaussian mechanism
        2. Convert composed RDP → (ε, δ)-DP by minimizing over α

    Args:
        num_steps:        total number of gradient/aggregation steps T
        noise_multiplier: σ (actual std of additive noise = σ · C)
        sample_rate:      q = batch_size / dataset_size  (or clients/round)
        delta:            target δ  (recommend δ << 1/N)
        orders:           Rényi orders to optimize over (default: DEFAULT_ORDERS)

    Returns:
        (epsilon, best_alpha): privacy guarantee and optimal Rényi order

    Example:
        >>> eps, _ = compute_epsilon(
        ...     num_steps=25, noise_multiplier=1.1,
        ...     sample_rate=0.5, delta=1e-5,
        ... )
        >>> print(f"ε = {eps:.3f}")  # Should print ε ≈ 2-5 depending on params
    """
    if orders is None:
        orders = DEFAULT_ORDERS

    # Compose RDP over num_steps  (Composition Theorem: values add)
    rdp_values = [
        num_steps * subsampled_gaussian_rdp(noise_multiplier, alpha, sample_rate)
        for alpha in orders
    ]

    return rdp_to_dp(rdp_values, orders, delta)


def find_noise_multiplier(
    target_epsilon: float,
    num_steps: int,
    sample_rate: float,
    delta: float,
    tolerance: float = 0.01,
    sigma_lo: float = 0.01,
    sigma_hi: float = 200.0,
) -> float:
    """
    Binary search for the minimum noise multiplier σ that achieves a
    target (ε, δ) privacy guarantee.

    This is the inverse of compute_epsilon: given ε target, find min σ.

    Args:
        target_epsilon:  desired ε budget
        num_steps:       total steps T
        sample_rate:     q = batch_size/N  or  clients/K
        delta:           target δ
        tolerance:       binary search precision on ε
        sigma_lo:        lower bound for search (default: 0.01)
        sigma_hi:        upper bound for search (default: 200)

    Returns:
        Minimum σ achieving (target_epsilon, delta)-DP
    """
    # Sanity: check bounds
    eps_lo, _ = compute_epsilon(num_steps, sigma_hi, sample_rate, delta)
    if eps_lo > target_epsilon:
        raise ValueError(
            f"Cannot achieve ε={target_epsilon} even with σ={sigma_hi}. "
            f"Minimum achievable ε={eps_lo:.4f}. "
            "Consider reducing num_steps or increasing sigma_hi."
        )

    eps_hi, _ = compute_epsilon(num_steps, sigma_lo, sample_rate, delta)
    if eps_hi < target_epsilon:
        return sigma_lo  # σ_lo is sufficient

    # Binary search
    for _ in range(64):  # max 64 iterations → precision ~ (sigma_hi - sigma_lo) / 2^64
        sigma_mid = (sigma_lo + sigma_hi) / 2.0
        eps_mid, _ = compute_epsilon(num_steps, sigma_mid, sample_rate, delta)

        if abs(eps_mid - target_epsilon) <= tolerance:
            return sigma_mid
        elif eps_mid > target_epsilon:
            # σ_mid too small → increase σ
            sigma_lo = sigma_mid
        else:
            # σ_mid more than sufficient → can decrease σ
            sigma_hi = sigma_mid

    return (sigma_lo + sigma_hi) / 2.0


# ══════════════════════════════════════════════════════════════════════════════
#   Stateful RDP Accountant Class
# ══════════════════════════════════════════════════════════════════════════════

class RDPAccountant:
    """
    Stateful RDP accountant that tracks privacy cost across steps/rounds.

    Usage:
        accountant = RDPAccountant(noise_multiplier=1.1, sample_rate=0.5, delta=1e-5)
        for round in training_loop:
            train_one_round(...)
            accountant.step()
            eps = accountant.get_epsilon()
            print(f"Round {round}: ε = {eps:.3f}")

    Reference: Mironov (2017), Wang et al. (2019), Abadi et al. (2016)
    """

    def __init__(
        self,
        noise_multiplier: float,
        sample_rate: float,
        delta: float = 1e-5,
        orders: Optional[List[float]] = None,
    ):
        """
        Args:
            noise_multiplier:  σ for Gaussian mechanism
            sample_rate:       q = batch_size/N or clients/K per round
            delta:             target δ for (ε,δ)-DP conversion
            orders:            Rényi orders (default: DEFAULT_ORDERS)
        """
        self.noise_multiplier = noise_multiplier
        self.sample_rate = sample_rate
        self.delta = delta
        self.orders = orders or DEFAULT_ORDERS

        self._num_steps: int = 0
        self._history: List[Tuple[int, float, float]] = []  # (step, epsilon, best_alpha)

    def step(self, num_steps: int = 1) -> None:
        """Record that `num_steps` gradient/aggregation steps have been taken."""
        self._num_steps += num_steps

    def get_epsilon(self, delta: Optional[float] = None) -> float:
        """
        Compute current accumulated ε for target δ.

        Args:
            delta: override target δ (uses self.delta if None)

        Returns:
            Current accumulated ε
        """
        d = delta if delta is not None else self.delta
        if self._num_steps == 0:
            return 0.0
        eps, _ = compute_epsilon(
            num_steps=self._num_steps,
            noise_multiplier=self.noise_multiplier,
            sample_rate=self.sample_rate,
            delta=d,
            orders=self.orders,
        )
        return eps

    def get_privacy_report(self, delta: Optional[float] = None) -> dict:
        """Return a complete privacy report dictionary."""
        d = delta if delta is not None else self.delta
        if self._num_steps == 0:
            return {
                "epsilon": 0.0,
                "delta": d,
                "num_steps": 0,
                "noise_multiplier": self.noise_multiplier,
                "sample_rate": self.sample_rate,
                "best_alpha": None,
                "assessment": "No steps taken yet.",
            }

        eps, best_alpha = compute_epsilon(
            num_steps=self._num_steps,
            noise_multiplier=self.noise_multiplier,
            sample_rate=self.sample_rate,
            delta=d,
            orders=self.orders,
        )

        return {
            "epsilon": round(eps, 4),
            "delta": d,
            "num_steps": self._num_steps,
            "noise_multiplier": self.noise_multiplier,
            "sample_rate": self.sample_rate,
            "best_alpha": round(best_alpha, 2) if best_alpha != float("inf") else None,
            "assessment": _assess_epsilon(eps),
        }

    def reset(self) -> None:
        """Reset step counter (e.g., for a new experiment)."""
        self._num_steps = 0
        self._history.clear()

    def __repr__(self) -> str:
        eps = self.get_epsilon()
        return (
            f"RDPAccountant(σ={self.noise_multiplier}, q={self.sample_rate}, "
            f"δ={self.delta}, steps={self._num_steps}, ε≈{eps:.3f})"
        )


# ══════════════════════════════════════════════════════════════════════════════
#   Utility: Privacy Level Assessment
# ══════════════════════════════════════════════════════════════════════════════

def _assess_epsilon(eps: float) -> str:
    """Human-readable assessment of a given ε value."""
    if eps <= 1.0:
        return "STRONG  (ε≤1): Near pure-DP. Very high privacy, significant utility cost."
    elif eps <= 3.0:
        return "GOOD    (1<ε≤3): Recommended for sensitive financial/medical data."
    elif eps <= 8.0:
        return "MODERATE (3<ε≤8): Acceptable for most FL-LLM tasks."
    elif eps <= 10.0:
        return "WEAK    (8<ε≤10): Only for low-sensitivity data."
    else:
        return "INSUFFICIENT (ε>10): Minimal formal privacy guarantee — not suitable for publication."

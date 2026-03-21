"""
DP Budget Manager
=================
Centralized management of the differential privacy budget across all
pipeline stages: training, retrieval, and generation.

Motivation:
  Existing DP work treats each stage independently. This module formalizes
  a unified budget allocation strategy:

      ε_total = ε_train + ε_retrieval + ε_generation  (sequential composition)

  and enables adaptive reallocation when stages underspend their budgets.

Reference:
  - Sequential composition: ε_total = Σε_i  (Dwork et al. 2006)
  - arXiv:2412.04697 (Dec 2024): per-token budget allocation in DP-RAG
  - arXiv:2412.19291 (Dec 2024): multi-stage DP in RAG pipelines
"""

from __future__ import annotations
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class PrivacyBudgetExhaustedError(Exception):
    """Raised when a component tries to spend more than its allocated budget."""
    pass


@dataclass
class ComponentBudget:
    """Budget state for a single pipeline component."""
    name: str
    allocated: float      # ε allocated to this component
    spent: float = 0.0    # ε consumed so far

    @property
    def remaining(self) -> float:
        return max(0.0, self.allocated - self.spent)

    @property
    def utilization(self) -> float:
        return self.spent / self.allocated if self.allocated > 0 else 0.0

    @property
    def is_exhausted(self) -> bool:
        return self.spent >= self.allocated


class DPBudgetManager:
    """
    Multi-stage privacy budget manager for DP-FedRAG.

    Tracks and enforces (ε, δ) budgets across the three main pipeline stages:
      1. training    — DP-SGD / DP-FedAvg during model fine-tuning
      2. retrieval   — Laplace/Gaussian noise on similarity scores
      3. generation  — DP token sampling / entity masking

    Composition model:
      ε_total ≤ ε_train + ε_retrieval + ε_generation  (basic sequential composition)

    The tighter advanced composition (Dwork et al. 2010) can be used for
    k >> 1 heterogeneous mechanisms:
      ε_total ≈ √(2k · ln(1/δ)) · ε_i + k · ε_i · (e^{ε_i} − 1)
    but basic composition is sufficient and conservative for our k = 3 stages.

    Args:
        total_epsilon:  total privacy budget ε_total
        total_delta:    total failure probability δ
        allocation:     fraction of total_epsilon per component
                        (must sum to ≤ 1.0)

    Example:
        >>> mgr = DPBudgetManager(total_epsilon=8.0, total_delta=1e-5)
        >>> mgr.spend("training", 4.2)
        >>> mgr.spend("retrieval", 2.1)
        >>> print(mgr.get_report())
    """

    # Default allocation: 60% training / 25% retrieval / 15% generation
    # Rationale:
    #   - Training consumes the most budget (many steps, cumulative composition)
    #   - Retrieval is per-query, needs moderate budget for meaningful noise calibration
    #   - Generation is applied post-hoc with lightweight masking
    DEFAULT_ALLOCATION: Dict[str, float] = {
        "training":   0.60,
        "retrieval":  0.25,
        "generation": 0.15,
    }

    def __init__(
        self,
        total_epsilon: float = 8.0,
        total_delta: float = 1e-5,
        allocation: Optional[Dict[str, float]] = None,
        strict: bool = True,
    ):
        """
        Args:
            total_epsilon:  total ε budget
            total_delta:    total δ
            allocation:     dict of {component: fraction} (must sum ≤ 1.0)
            strict:         if True, raise PrivacyBudgetExhaustedError on over-spend;
                            if False, log warning only
        """
        self.total_epsilon = total_epsilon
        self.total_delta = total_delta
        self.strict = strict

        alloc = allocation or self.DEFAULT_ALLOCATION

        # Validate allocation
        total_alloc = sum(alloc.values())
        if total_alloc > 1.0 + 1e-8:
            raise ValueError(
                f"Allocation fractions sum to {total_alloc:.4f} > 1.0. "
                "Reduce one or more fractions."
            )

        # Build component budgets
        self._components: Dict[str, ComponentBudget] = {
            name: ComponentBudget(
                name=name,
                allocated=total_epsilon * fraction,
            )
            for name, fraction in alloc.items()
        }

        logger.info(
            f"[DPBudgetManager] Initialized: ε_total={total_epsilon}, δ={total_delta}\n"
            + "\n".join(
                f"  {name}: ε_allocated={b.allocated:.3f}"
                for name, b in self._components.items()
            )
        )

    # ── Public API ─────────────────────────────────────────────────────────

    def get_budget(self, component: str) -> float:
        """Return allocated ε for a component."""
        return self._get_or_create(component).allocated

    def get_remaining(self, component: str) -> float:
        """Return remaining ε for a component."""
        return self._get_or_create(component).remaining

    def spend(self, component: str, epsilon: float) -> float:
        """
        Record that `epsilon` DP budget has been consumed by `component`.

        Args:
            component:  name of the pipeline component
            epsilon:    amount of ε consumed

        Returns:
            Remaining budget for the component after this spend.

        Raises:
            PrivacyBudgetExhaustedError: if strict=True and spend exceeds allocation
        """
        if epsilon < 0:
            raise ValueError(f"Cannot spend negative ε={epsilon}")

        comp = self._get_or_create(component)
        new_spent = comp.spent + epsilon
        new_remaining = comp.allocated - new_spent

        if new_remaining < -1e-8:
            msg = (
                f"[DPBudgetManager] Component '{component}' over-spends: "
                f"spent={new_spent:.4f} > allocated={comp.allocated:.4f}. "
                f"Total ε_total={self.total_epsilon}"
            )
            if self.strict:
                raise PrivacyBudgetExhaustedError(msg)
            else:
                logger.warning(msg)

        comp.spent = new_spent
        logger.info(
            f"[DPBudgetManager] {component}: spent={new_spent:.4f} / "
            f"allocated={comp.allocated:.4f} "
            f"(remaining={max(0.0, new_remaining):.4f})"
        )
        return max(0.0, new_remaining)

    def is_exhausted(self, component: str) -> bool:
        """Check if a component's budget is fully spent."""
        return self._get_or_create(component).is_exhausted

    def get_total_spent(self) -> float:
        """
        Total ε consumed across all components.

        Under sequential composition, this is an upper bound on the
        actual ε of the entire pipeline.
        """
        return sum(c.spent for c in self._components.values())

    def get_report(self) -> dict:
        """
        Generate a complete privacy report.

        Returns a dict with per-component and total budget status,
        suitable for logging or including in a paper experiment table.
        """
        total_spent = self.get_total_spent()
        per_component = {
            name: {
                "allocated": round(comp.allocated, 4),
                "spent":     round(comp.spent, 4),
                "remaining": round(comp.remaining, 4),
                "utilization": f"{100.0 * comp.utilization:.1f}%",
                "exhausted":  comp.is_exhausted,
            }
            for name, comp in self._components.items()
        }

        return {
            "total": {
                "epsilon_budget":  self.total_epsilon,
                "epsilon_spent":   round(total_spent, 4),
                "epsilon_remaining": round(self.total_epsilon - total_spent, 4),
                "delta":           self.total_delta,
                "composition":     "sequential",
                "assessment":      _assess_epsilon(total_spent),
            },
            "per_component": per_component,
            "recommendation": _generate_recommendation(per_component, total_spent, self.total_epsilon),
        }

    def reallocate(self, from_component: str, to_component: str, amount: float) -> None:
        """
        Transfer unused budget from one component to another.

        Useful when, e.g., generation needs less budget than expected
        and training can use the surplus for more rounds.

        Args:
            from_component: component donating budget
            to_component:   component receiving budget
            amount:         ε to transfer
        """
        src = self._get_or_create(from_component)
        dst = self._get_or_create(to_component)

        if amount > src.remaining:
            raise ValueError(
                f"Cannot transfer ε={amount} from '{from_component}': "
                f"only {src.remaining:.4f} remaining."
            )

        src.allocated -= amount
        dst.allocated += amount

        logger.info(
            f"[DPBudgetManager] Reallocated ε={amount:.4f}: "
            f"{from_component} → {to_component}"
        )

    def add_component(self, name: str, epsilon: float) -> None:
        """
        Register a new component with its own epsilon allocation.

        Note: this does NOT deduct from other components.
        Ensure total_epsilon remains valid.
        """
        if name in self._components:
            raise ValueError(f"Component '{name}' already exists.")
        self._components[name] = ComponentBudget(name=name, allocated=epsilon)
        self.total_epsilon = sum(c.allocated for c in self._components.values())

    # ── Private Helpers ────────────────────────────────────────────────────

    def _get_or_create(self, component: str) -> ComponentBudget:
        if component not in self._components:
            logger.warning(
                f"[DPBudgetManager] Unknown component '{component}'. "
                "Creating with 0 budget."
            )
            self._components[component] = ComponentBudget(name=component, allocated=0.0)
        return self._components[component]

    def __repr__(self) -> str:
        total_spent = self.get_total_spent()
        return (
            f"DPBudgetManager("
            f"ε_total={self.total_epsilon}, "
            f"ε_spent={total_spent:.3f}, "
            f"δ={self.total_delta}, "
            f"components={list(self._components.keys())})"
        )


# ══════════════════════════════════════════════════════════════════════════════
#   Utility Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _assess_epsilon(eps: float) -> str:
    """Human-readable privacy level for a given ε."""
    if eps <= 1.0:
        return "STRONG  (ε≤1): Near pure-DP."
    elif eps <= 3.0:
        return "GOOD    (1<ε≤3): Recommended for sensitive financial/medical data."
    elif eps <= 8.0:
        return "MODERATE (3<ε≤8): Practical for FL-LLM tasks."
    elif eps <= 10.0:
        return "WEAK    (8<ε≤10): Minimal guarantees."
    else:
        return "INSUFFICIENT (ε>10): Not suitable for DP claims in publication."


def _generate_recommendation(per_component: dict, total_spent: float, total_budget: float) -> str:
    """Generate actionable recommendation from budget state."""
    utilization = total_spent / total_budget if total_budget > 0 else 0

    if utilization < 0.5:
        return (
            "Budget under-utilized. Consider increasing num_rounds or "
            "using a smaller noise_multiplier to improve utility."
        )
    elif utilization > 0.95:
        return (
            "Budget nearly exhausted. Stop training to preserve formal guarantee. "
            "Consider increasing total_epsilon or reducing num_rounds."
        )
    else:
        return "Budget usage is healthy. Continue training within allocated budget."

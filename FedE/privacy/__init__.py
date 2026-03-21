"""
DP Privacy Module for FinSafeRAG Federated Learning
=====================================================
Implements formal Differential Privacy for FL-RAG system.

Components:
  - rdp_accountant: RDP-based privacy accounting (Mironov 2017)
  - dp_budget_manager: Multi-stage privacy budget management
"""

from .rdp_accountant import RDPAccountant, compute_epsilon, gaussian_rdp, subsampled_gaussian_rdp
from .dp_budget_manager import DPBudgetManager, PrivacyBudgetExhaustedError

__all__ = [
    "RDPAccountant",
    "compute_epsilon",
    "gaussian_rdp",
    "subsampled_gaussian_rdp",
    "DPBudgetManager",
    "PrivacyBudgetExhaustedError",
]

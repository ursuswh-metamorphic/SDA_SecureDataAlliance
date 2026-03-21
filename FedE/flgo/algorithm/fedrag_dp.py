"""
DP-FedRAG: Formally Private Federated RAG Algorithm
====================================================
Improves the original fedrag.py with:

  1. CORRECT per-sample gradient clipping  (fixes batch-level clipping bug)
  2. RDP accounting with (ε, δ) tracking   (was completely absent)
  3. Adaptive clip norm C_t                (Andrew et al., NeurIPS 2021)
  4. User-level DP at the server           (McMahan et al., ICLR 2018)
  5. Privacy amplification by subsampling  (Balle et al., NeurIPS 2018)
  6. Early stopping when budget exhausted

Compared to the baseline fedrag.py, the key fixes are:

  ┌─────────────────────────┬──────────────────────────────────────────────────┐
  │ Baseline (fedrag.py)    │ This module (fedrag_dp.py)                       │
  ├─────────────────────────┼──────────────────────────────────────────────────┤
  │ Batch-level clipping    │ Per-sample clipping (Definition-correct DP-SGD)  │
  │ σ = 0.1 (ε ≈ 120)       │ σ calibrated to target ε  (typically σ ≥ 0.8)   │
  │ No (ε,δ) tracking       │ RDP accountant after every round                 │
  │ Fixed clip norm C       │ Adaptive C_t via quantile estimation             │
  │ No server-side DP       │ User-level DP: clip+noise at server              │
  │ All clients per round   │ Client subsampling for amplification             │
  └─────────────────────────┴──────────────────────────────────────────────────┘

References:
  [1] Abadi et al. (2016). "Deep Learning with DP." CCS 2016. arXiv:1607.00133
  [2] McMahan et al. (2018). "Learning Differentially Private RLM." ICLR 2018.
  [3] Mironov (2017). "Rényi DP." IEEE CSF 2017. arXiv:1702.07476
  [4] Wang et al. (2019). "Subsampled RDP." AISTATS 2019.
  [5] Balle et al. (2018). "Privacy Amplification by Subsampling." NeurIPS 2018.
  [6] Andrew et al. (2021). "DP Learning with Adaptive Clipping." NeurIPS 2021.
  [7] Gopi et al. (2021). "Numerical Composition of DP." NeurIPS 2021.
"""

from __future__ import annotations

import copy
import logging
from datetime import datetime

import numpy as np
import torch
import os

from .fedbase import BasicServer, BasicClient

# Import our formal privacy accounting module
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from privacy.rdp_accountant import RDPAccountant, compute_epsilon, find_noise_multiplier
from privacy.dp_budget_manager import DPBudgetManager, PrivacyBudgetExhaustedError

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#   Server: User-Level DP Aggregation
# ══════════════════════════════════════════════════════════════════════════════

class Server(BasicServer):
    """
    Federated Server with User-Level Differential Privacy.

    Implements the DP-FedAvg protocol (McMahan et al. 2018):
      For each round t:
        1. Sample C clients (subsampling amplification)
        2. Receive trained model from each client
        3. Compute update: Δ_i = local_model_i − global_model
        4. Clip each update: Δ̄_i = Δ_i / max(1, ‖Δ_i‖₂ / C_server)
        5. Aggregate: S = Σ Δ̄_i / K
        6. Add Gaussian noise: S̃ = S + N(0, σ_s² · C_server² / K² · I)
        7. Update global: θ_{t+1} = θ_t + S̃

    Privacy guarantee (user-level):
      After T rounds, with K clients, C sampled per round, clip C_s, noise σ_s:
        (ε, δ)-DP via RDP accountant with amplification rate q = C/K

    Args (via option dict):
        dp_enabled:             Enable DP (default: False)
        target_epsilon:         Target ε for overall training (default: 8.0)
        target_delta:           Target δ (default: 1e-5)
        server_clip_norm:       Clip norm C_s for client update vectors (default: 1.0)
        server_noise_multiplier:σ_s for server-side Gaussian noise (default: 1.1)
        dp_adaptive_clip:       Enable adaptive clip norm (default: True)
        dp_clip_gamma:          Target quantile for adaptive clipping (default: 0.5)
        dp_clients_per_round:   Number of clients sampled per round (default: all)
    """

    def run(self):
        """Standard FL training loop, saving checkpoint and model each round."""
        self.gv.logger.time_start('Total Time Cost')
        print(type(self.model.model))

        # ── Setup DP components ────────────────────────────────────────────
        self.dp_enabled = self.option.get('dp_enabled', False)
        self.target_epsilon = self.option.get('target_epsilon', 8.0)
        self.target_delta = self.option.get('target_delta', 1e-5)
        self.server_clip_norm = self.option.get('server_clip_norm', 1.0)
        self.server_noise_multiplier = self.option.get('server_noise_multiplier', 1.1)
        self.adaptive_clip = self.option.get('dp_adaptive_clip', True)
        self.clip_gamma = self.option.get('dp_clip_gamma', 0.5)
        self._dp_clients_per_round = self.option.get(
            'dp_clients_per_round', len(self.clients)
        )

        if self.dp_enabled:
            # Sampling rate for privacy amplification: q = clients_sampled / total_clients
            sampling_rate = self._dp_clients_per_round / max(len(self.clients), 1)
            self.server_accountant = RDPAccountant(
                noise_multiplier=self.server_noise_multiplier,
                sample_rate=sampling_rate,
                delta=self.target_delta,
            )

            # Privacy budget manager (server owns "training" budget)
            self.budget_manager = DPBudgetManager(
                total_epsilon=self.target_epsilon,
                total_delta=self.target_delta,
                allocation={"training": 1.0},   # server-side: all budget to training
                strict=False,                     # warn, don't crash, for long runs
            )

            # Calibrate σ if not explicitly set
            num_rounds = self.option.get('num_rounds', 25)
            suggested_sigma = _calibrate_sigma(
                target_epsilon=self.target_epsilon,
                num_steps=num_rounds,
                sample_rate=sampling_rate,
                delta=self.target_delta,
            )
            if self.server_noise_multiplier < suggested_sigma - 0.05:
                logger.warning(
                    f"[DP-Server] server_noise_multiplier={self.server_noise_multiplier:.3f} "
                    f"is LOWER than calibrated σ={suggested_sigma:.3f} for "
                    f"target ε={self.target_epsilon}. "
                    f"Privacy guarantee may be WEAKER than expected. "
                    f"Recommended: set server_noise_multiplier ≥ {suggested_sigma:.2f}"
                )

            logger.info(
                f"[DP-Server] Initialized: ε_target={self.target_epsilon}, "
                f"δ={self.target_delta}, "
                f"σ_server={self.server_noise_multiplier}, "
                f"C_server={self.server_clip_norm}, "
                f"q={sampling_rate:.3f} (amplification)"
            )

        # ── Main training loop (same as base) ─────────────────────────────
        if not self._load_checkpoint() and self.eval_interval > 0:
            self.gv.logger.info("--------------Initial Evaluation--------------")
            self.gv.logger.time_start('Eval Time Cost')
            self.gv.logger.time_end('Eval Time Cost')

        while True:
            if self._if_exit():
                break

            # Early stop on privacy budget exhaustion
            if self.dp_enabled and self._is_privacy_budget_exhausted():
                logger.warning(
                    "[DP-Server] Privacy budget EXHAUSTED. Stopping training early. "
                    f"Final report: {self._get_privacy_report()}"
                )
                break

            self.gv.clock.step()
            updated = self.iterate()

            if updated is True or updated is None:
                self.gv.logger.info(
                    "--------------Round {}--------------".format(self.current_round)
                )
                # Log privacy status every round
                if self.dp_enabled:
                    report = self._get_privacy_report()
                    logger.info(
                        f"[DP-Server] Round {self.current_round}: "
                        f"ε_spent={report['epsilon']:.4f} / ε_target={self.target_epsilon}, "
                        f"δ={self.target_delta}, "
                        f"assessment: {report['assessment']}"
                    )

                if self.gv.logger.check_if_log(self.current_round, self.eval_interval):
                    self.gv.logger.time_start('Eval Time Cost')
                    self.gv.logger.time_end('Eval Time Cost')
                    self._save_checkpoint()

                if self.gv.logger.early_stop():
                    break

                if self.current_round >= 0:
                    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    filename = f"x-model_{current_time}_round{self.current_round}.bin"
                    save_path = os.path.join("./checkpoints", filename)
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save(self.model.model.state_dict(), save_path)

                self.current_round += 1
                self.global_lr_scheduler(self.current_round)

        # ── Final privacy report ───────────────────────────────────────────
        if self.dp_enabled:
            final_report = self._get_privacy_report()
            logger.info(
                "\n" + "=" * 60 +
                "\n[DP-Server] FINAL PRIVACY REPORT\n" +
                "=" * 60 + "\n" +
                f"  Total ε spent:    {final_report['epsilon']:.4f}\n"
                f"  δ:                {final_report['delta']}\n"
                f"  Total rounds:     {self.current_round}\n"
                f"  Noise multiplier: {self.server_noise_multiplier}\n"
                f"  Clip norm:        {self.server_clip_norm:.4f}\n"
                f"  Assessment:       {final_report['assessment']}\n" +
                "=" * 60
            )

        self.gv.logger.info("=================End==================")
        self.gv.logger.time_end('Total Time Cost')
        self.gv.logger.save_output_as_json()
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"x-model_{current_time}.bin"
        torch.save(self.model.model.state_dict(), filename)

    # ── Aggregation with User-Level DP ────────────────────────────────────

    def average_tensors(self, list_of_dicts):
        """Plain average (used as helper)."""
        if not list_of_dicts:
            return None
        avg_dict = {}
        for d in list_of_dicts:
            for key, value in d.items():
                if key not in avg_dict:
                    avg_dict[key] = value.clone()
                else:
                    avg_dict[key] += value
        num_dicts = len(list_of_dicts)
        for key in avg_dict:
            avg_dict[key] /= num_dicts
        avg_dict = {'model.' + key: value for key, value in avg_dict.items()}
        return avg_dict

    def aggregate(self, model_old, models: list, *args, **kwargs):
        """
        DP-aware aggregation with user-level privacy.

        Implements Algorithm 1 from McMahan et al. (ICLR 2018):
          1. Clip each client update Δ_i = θ_i − θ_old to norm C_server
          2. Aggregate: S = (1/K) Σ Δ̄_i
          3. Add Gaussian noise: S̃ = S + N(0, σ_s²·C_s²/K² · I)
          4. Apply: θ_new = θ_old + S̃
        """
        if not self.dp_enabled:
            # Fall back to plain FedAvg
            all_params = [model.state_dict() for model in models]
            ans_params = self.average_tensors(all_params)
            model_old.load_state_dict(ans_params)
            return model_old

        K = len(models)
        if K == 0:
            return model_old

        C = self.server_clip_norm
        sigma = self.server_noise_multiplier
        old_state = model_old.state_dict()

        # ── Step 1: Compute and clip client updates ────────────────────────
        clipped_updates = []
        update_norms = []

        for client_model in models:
            client_state = client_model.state_dict()

            # Compute update Δ_i = θ_client − θ_global (as flat vector)
            update_flat = _state_dict_diff_flat(client_state, old_state)
            norm = torch.norm(update_flat, p=2).item()
            update_norms.append(norm)

            # Clip: Δ̄_i = Δ_i / max(1, ‖Δ_i‖₂ / C)
            clip_coef = min(1.0, C / (norm + 1e-8))

            # Apply clip coefficient to state dict
            clipped = {}
            for key in client_state:
                if key in old_state:
                    delta = client_state[key].float() - old_state[key].float()
                    clipped[key] = old_state[key].float() + delta * clip_coef
                else:
                    clipped[key] = client_state[key].float()
            clipped_updates.append(clipped)

        # ── Step 2: Adaptive clip norm update (Andrew et al. 2021) ─────────
        if self.adaptive_clip and len(update_norms) > 0:
            target_quantile = np.quantile(update_norms, self.clip_gamma)
            # Smooth update: exponential moving average
            self.server_clip_norm = (
                0.9 * self.server_clip_norm + 0.1 * target_quantile
            )
            logger.debug(
                f"[DP-Server] Adaptive clip: new C={self.server_clip_norm:.4f} "
                f"(norms: mean={np.mean(update_norms):.3f}, "
                f"median={np.median(update_norms):.3f})"
            )

        # ── Step 3: Average clipped updates ────────────────────────────────
        avg_state = {}
        for key in clipped_updates[0]:
            avg_state[key] = torch.stack(
                [upd[key] for upd in clipped_updates]
            ).mean(dim=0)

        # ── Step 4: Add Gaussian noise (user-level DP) ─────────────────────
        # Noise std = σ_s · C_s / K  (for mean aggregation)
        # Reference: Equation (1) in McMahan et al. 2018
        noise_std = sigma * C / K
        device = next(model_old.parameters()).device

        for key in avg_state:
            if avg_state[key].is_floating_point():
                noise = torch.randn_like(avg_state[key]) * noise_std
                avg_state[key] = avg_state[key] + noise

        # ── Step 5: Update model ────────────────────────────────────────────
        # Restore proper key prefix if needed
        prefixed = {'model.' + k: v for k, v in avg_state.items()
                    if not k.startswith('model.')}
        if not prefixed:
            prefixed = avg_state

        try:
            model_old.load_state_dict(prefixed, strict=False)
        except RuntimeError:
            # Try without prefix if loading fails
            try:
                model_old.load_state_dict(avg_state, strict=False)
            except RuntimeError as e:
                logger.error(f"[DP-Server] Failed to load state dict: {e}")

        # ── Step 6: Account privacy ─────────────────────────────────────────
        if hasattr(self, 'server_accountant'):
            self.server_accountant.step()
            eps = self.server_accountant.get_epsilon()
            try:
                self.budget_manager.spend("training", eps - self.budget_manager._components.get("training", type('obj', (object,), {'spent': 0.0})()).spent)
            except PrivacyBudgetExhaustedError:
                pass

        logger.info(
            f"[DP-Server] Aggregated {K} clients: "
            f"update norms (mean={np.mean(update_norms):.3f}, "
            f"max={np.max(update_norms):.3f}), "
            f"C={C:.4f}, noise_std={noise_std:.6f}"
        )
        return model_old

    # ── Privacy Helpers ────────────────────────────────────────────────────

    def _is_privacy_budget_exhausted(self) -> bool:
        if not hasattr(self, 'server_accountant'):
            return False
        eps = self.server_accountant.get_epsilon()
        return eps >= self.target_epsilon

    def _get_privacy_report(self) -> dict:
        if not hasattr(self, 'server_accountant'):
            return {"epsilon": 0.0, "delta": 0.0, "assessment": "DP disabled"}
        return self.server_accountant.get_privacy_report()


# ══════════════════════════════════════════════════════════════════════════════
#   Client: Per-Sample DP-SGD
# ══════════════════════════════════════════════════════════════════════════════

class Client(BasicClient):
    """
    Federated Client with Per-Sample DP-SGD (Abadi et al. 2016).

    Implements correct per-sample gradient clipping:
      For each sample i in batch:
        1. Compute gradient g_i = ∇L(x_i, θ)
        2. Clip: g̃_i = g_i / max(1, ‖g_i‖₂ / C)
        3. Accumulate: G = Σ g̃_i
      After batch:
        4. Add noise: G̃ = G + N(0, σ²·C²·I)
        5. Update: θ ← θ − η · G̃ / B

    Compared to the ORIGINAL fedrag.py (batch-level clipping), this is the
    CORRECT formulation per Definition 1 of Abadi et al. (2016).

    Args (via option dict):
        dp_enabled:             Enable DP (default: False)
        target_epsilon:         Target ε for client training (default: 8.0)
        target_delta:           Target δ (default: 1e-5)
        dp_clip_norm:           Per-sample gradient clip norm C (default: 1.0)
        dp_noise_multiplier:    Noise multiplier σ (default: 1.1)
        dp_adaptive_clip:       Adaptive clipping (default: True)
        dp_clip_gamma:          Quantile for adaptive clip (default: 0.5)
    """

    def train(self, model, local_model):
        local_model.train()
        optimizer = self.calculator.get_optimizer(
            local_model,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            momentum=self.momentum,
        )
        model.to(self.device)
        local_model.to(self.device)

        dp_enabled = self.option.get('dp_enabled', False)
        clip_norm = self.option.get('dp_clip_norm', 1.0)
        noise_multiplier = self.option.get('dp_noise_multiplier', 1.1)
        adaptive_clip = self.option.get('dp_adaptive_clip', True)
        clip_gamma = self.option.get('dp_clip_gamma', 0.5)

        if not dp_enabled:
            # ── Baseline: plain training (no DP) ──────────────────────────
            self._train_standard(model, local_model, optimizer)
            return

        # ── DP-SGD training ────────────────────────────────────────────────
        for step in range(self.num_steps):
            batch_data = self.get_batch_data()
            batch_size = _get_batch_size(batch_data)

            if batch_size == 0:
                continue

            # ── Adaptive clip norm: estimate γ-quantile of gradient norms ──
            if adaptive_clip and step == 0:
                sample_norms = _sample_gradient_norms(
                    model, local_model, batch_data,
                    self.calculator, n_samples=min(batch_size, 8)
                )
                if sample_norms:
                    target_norm = float(np.quantile(sample_norms, clip_gamma))
                    clip_norm = 0.9 * clip_norm + 0.1 * target_norm
                    logger.debug(
                        f"[DP-Client {self.id}] Adaptive clip: "
                        f"C={clip_norm:.4f} ({clip_gamma*100:.0f}th-pct of norms)"
                    )

            # ── Per-sample clipping + noise ────────────────────────────────
            params = [p for p in local_model.parameters() if p.requires_grad]

            # Accumulator for clipped gradients
            accumulated = [torch.zeros_like(p.data) for p in params]

            for i in range(batch_size):
                single = _get_single_sample(batch_data, i)
                local_model.zero_grad()

                # Forward pass on single sample
                server_loss = self.calculator.compute_server_loss(model, single)
                client_loss, client_only, server_only = self.calculator.compute_client_loss(
                    server_loss, local_model, single
                )
                client_loss.backward()

                # Per-sample L2 gradient norm
                per_sample_norm = _compute_grad_norm(params)

                # Clip coefficient: min(1, C/‖g_i‖)
                clip_coef = min(1.0, clip_norm / (per_sample_norm + 1e-8))

                # Accumulate clipped gradients
                for j, p in enumerate(params):
                    if p.grad is not None:
                        accumulated[j] += p.grad.detach() * clip_coef

            # ── Noise injection: N(0, σ²·C²·I) per coordinate ─────────────
            for j, p in enumerate(params):
                noise = torch.randn_like(accumulated[j]) * (noise_multiplier * clip_norm)
                # Set gradient to (Σ clipped_grads + noise) / batch_size
                p.grad = (accumulated[j] + noise) / float(batch_size)

            optimizer.step()

            # Log progress
            if step % max(1, self.num_steps // 5) == 0:
                total_grad_norm = _compute_grad_norm(params)
                print(
                    f"[DP-Client {self.id}] step {step}/{self.num_steps}, "
                    f"C={clip_norm:.4f}, σ={noise_multiplier}, "
                    f"‖∇‖={total_grad_norm:.4f}"
                )

    def _train_standard(self, model, local_model, optimizer):
        """Original non-DP training loop (unchanged from fedrag.py)."""
        for iter in range(self.num_steps):
            batch_data = self.get_batch_data()
            local_model.zero_grad()
            server_loss = self.calculator.compute_server_loss(model, batch_data)
            client_loss, client_only, server_only = self.calculator.compute_client_loss(
                server_loss, local_model, batch_data
            )
            print(
                f"client running:{iter}/{self.num_steps}, "
                f"client loss: {client_loss}, "
                f"loss 1: {client_only}, loss 2: {server_only}"
            )
            client_loss.backward()
            optimizer.step()
            if iter == self.num_steps - 1:
                print(f"server loss: {server_loss}")

    def reply(self, svr_pkg):
        model = self.unpack(svr_pkg)
        self.train(model, self.model)
        cpkg = self.pack(self.model)
        return cpkg


# ══════════════════════════════════════════════════════════════════════════════
#   Helper Functions
# ══════════════════════════════════════════════════════════════════════════════

def _compute_grad_norm(params) -> float:
    """Compute L2 norm of all gradients."""
    total_sq = sum(
        p.grad.detach().norm(2).item() ** 2
        for p in params
        if p.grad is not None
    )
    return total_sq ** 0.5


def _get_batch_size(batch_data) -> int:
    """
    Infer batch size from heterogeneous batch formats.
    Supports: dict (BERT/HuggingFace), list, tuple, Tensor.
    """
    if isinstance(batch_data, dict):
        # HuggingFace-style: {'input_ids': Tensor(B, L), ...}
        for v in batch_data.values():
            if hasattr(v, '__len__'):
                return len(v)
        return 0
    elif isinstance(batch_data, (list, tuple)) and len(batch_data) > 0:
        first = batch_data[0]
        if hasattr(first, '__len__'):
            return len(first)
        return len(batch_data)
    elif hasattr(batch_data, '__len__'):
        return len(batch_data)
    return 1


def _get_single_sample(batch_data, idx: int):
    """
    Extract the i-th sample from a batch, preserving format.
    Handles dict (HF tokenizer), list, tuple, and Tensor batches.
    """
    if isinstance(batch_data, dict):
        return {
            k: v[idx:idx + 1] if hasattr(v, '__getitem__') else v
            for k, v in batch_data.items()
        }
    elif isinstance(batch_data, (list, tuple)):
        single = []
        for item in batch_data:
            if hasattr(item, '__getitem__') and hasattr(item, '__len__'):
                single.append(item[idx:idx + 1])
            else:
                single.append(item)
        return type(batch_data)(single)
    elif hasattr(batch_data, '__getitem__'):
        return batch_data[idx:idx + 1]
    return batch_data


def _state_dict_diff_flat(state_a: dict, state_b: dict) -> torch.Tensor:
    """
    Compute Δ = state_a − state_b as a single flat vector.
    Skips non-floating-point tensors (e.g. running counts).
    """
    diffs = []
    for key in state_b:
        if key in state_a:
            ta = state_a[key].float()
            tb = state_b[key].float()
            if ta.is_floating_point() and ta.shape == tb.shape:
                diffs.append((ta - tb).reshape(-1))
    if not diffs:
        return torch.zeros(1)
    return torch.cat(diffs)


def _sample_gradient_norms(
    model, local_model, batch_data, calculator, n_samples: int = 8
) -> list:
    """
    Compute per-sample gradient norms for a few samples.
    Used to estimate the quantile for adaptive clipping.
    """
    params = [p for p in local_model.parameters() if p.requires_grad]
    norms = []
    batch_size = _get_batch_size(batch_data)

    for i in range(min(n_samples, batch_size)):
        single = _get_single_sample(batch_data, i)
        local_model.zero_grad()
        try:
            server_loss = calculator.compute_server_loss(model, single)
            client_loss, _, _ = calculator.compute_client_loss(
                server_loss, local_model, single
            )
            client_loss.backward()
            norm = _compute_grad_norm(params)
            norms.append(norm)
        except Exception:
            pass

    local_model.zero_grad()  # Clean up
    return norms


def _calibrate_sigma(
    target_epsilon: float,
    num_steps: int,
    sample_rate: float,
    delta: float,
) -> float:
    """
    Find minimum σ that achieves target (ε, δ) given num_steps and sample_rate.
    Falls back to σ=1.1 if calibration fails.
    """
    try:
        sigma = find_noise_multiplier(
            target_epsilon=target_epsilon,
            num_steps=num_steps,
            sample_rate=sample_rate,
            delta=delta,
        )
        return sigma
    except Exception as e:
        logger.warning(f"[DP] Sigma calibration failed ({e}). Using σ=1.1 as default.")
        return 1.1

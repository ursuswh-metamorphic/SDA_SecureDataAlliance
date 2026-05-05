"""
fedrag_lora.py — FedRAG with LoRA-only state transport + per-sample DP.

Phase 1: only LoRA-adapter weights are aggregated; frozen base preserved.
Phase 2: when option['dp_enabled']=True, Client.train applies per-sample
gradient clipping + Gaussian noise on LoRA gradients only. σ is calibrated
once via privacy.rdp_accountant.find_noise_multiplier; an RDPAccountant
ticks per round on the server and logs eps_spent.

Mirrors the validated standalone recipe at main_dp_lora_eps20.py:43-46
(σ calibration), :100-127 (per-sample clip + noise), :130 (post-noise
stability clip), :178-209 (RDP accountant tick).

flgo dispatch contract (see fedbase.py:312):
    models = self.communicate(self.selected_clients)['model']
    self.model = self.aggregate(self.model, models)

So pack/unpack stay on the 'model' key; LoRA filtering happens in `aggregate`.
"""
from datetime import datetime
import os
import sys
import torch

from .fedbase import BasicServer
from .fedbase import BasicClient

# Reuse per-sample helpers from fedrag_dp (do NOT modify that file).
from .fedrag_dp import _get_single_sample, _get_batch_size, _compute_grad_norm

# Import privacy accountant from project-level privacy/ module.
_PRIVACY_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..')
)
if _PRIVACY_DIR not in sys.path:
    sys.path.insert(0, _PRIVACY_DIR)
from privacy.rdp_accountant import (  # noqa: E402
    RDPAccountant, find_noise_multiplier, compute_epsilon,
)


# ══════════════════════════════════════════════════════════════════════════════
#   Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _is_lora_key(key: str) -> bool:
    """Predicate matching all PEFT-injected LoRA parameters.

    Matches both the typical `base_model.model.<...>.lora_A.default.weight`
    and any other naming variant containing 'lora'. Mirrors the filter at
    main_dp_lora_eps20.py:137.
    """
    return 'lora' in key.lower()


def _lora_state_only(state_dict: dict) -> dict:
    """Return CPU-resident copy of the LoRA-only entries of a state_dict."""
    return {k: v.detach().cpu() for k, v in state_dict.items() if _is_lora_key(k)}


def _calibrate_sigma_quiet(option: dict):
    """Compute σ for DP-SGD; cache on the option dict to avoid recomputation.

    Deterministic — both Server and Client arrive at the same value when
    given the same hyperparameters, so it does not matter whether they share
    the option dict or each have their own copy.

    Returns None when DP is disabled.
    """
    if not option.get('dp_enabled', False):
        return None
    cached = option.get('dp_noise_multiplier')
    if cached is not None:
        return cached

    target_eps = float(option.get('target_epsilon', 20.0))
    target_delta = float(option.get('target_delta', 1e-5))
    num_rounds = int(option.get('num_rounds', 25))
    n_clients = int(option.get('num_clients', 5))
    clients_per_round = int(option.get('dp_clients_per_round', n_clients))
    sample_rate = clients_per_round / max(1, n_clients)

    sigma = find_noise_multiplier(
        target_epsilon=target_eps,
        num_steps=num_rounds,
        sample_rate=sample_rate,
        delta=target_delta,
    )
    option['dp_noise_multiplier'] = sigma
    option['_dp_sample_rate'] = sample_rate
    return sigma


# ══════════════════════════════════════════════════════════════════════════════
#   Server
# ══════════════════════════════════════════════════════════════════════════════

class Server(BasicServer):
    def _setup_dp(self):
        """Calibrate σ and instantiate the RDPAccountant. Idempotent."""
        if getattr(self, '_dp_initialized', False):
            return
        self._dp_initialized = True

        if not self.option.get('dp_enabled', False):
            self._dp_sigma = None
            self._rdp_accountant = None
            print('[fedrag_lora.Server] DP disabled (option[\'dp_enabled\']=False).')
            return

        sigma = _calibrate_sigma_quiet(self.option)
        # Recompute sample_rate inline (cheaper than caching, robust to flgo
        # option-dict copying that may drop auxiliary keys between calls).
        n_clients_for_rate = int(self.option.get('num_clients', 5))
        clients_per_round_for_rate = int(
            self.option.get('dp_clients_per_round', n_clients_for_rate)
        )
        sample_rate = clients_per_round_for_rate / max(1, n_clients_for_rate)
        target_eps = float(self.option.get('target_epsilon', 20.0))
        target_delta = float(self.option.get('target_delta', 1e-5))
        num_rounds = int(self.option.get('num_rounds', 25))

        # Verify the calibration produces ε ≤ target.
        eps_check, _ = compute_epsilon(num_rounds, sigma, sample_rate, target_delta)
        print(
            f'[fedrag_lora.Server] DP calibrated: σ={sigma:.4f}, '
            f'verification eps={eps_check:.4f} for target eps={target_eps}, '
            f'q={sample_rate:.2f}, T={num_rounds}, δ={target_delta}.'
        )

        self._dp_sigma = sigma
        self._rdp_accountant = RDPAccountant(
            noise_multiplier=sigma,
            sample_rate=sample_rate,
            delta=target_delta,
        )

    def run(self):
        """Federated training loop. Mirrors fedrag.Server.run line-for-line
        except checkpointing saves only the LoRA-adapter state (small file)
        and DP setup runs in the preamble."""
        self.gv.logger.time_start('Total Time Cost')

        print(type(self.model.model))

        # Phase 1: log payload size once.
        lora_state = _lora_state_only(self.model.state_dict())
        n_lora = sum(v.numel() for v in lora_state.values())
        print(f'[fedrag_lora.Server] LoRA payload per round: {n_lora:,} params, '
              f'{len(lora_state)} tensors.')

        # Phase 2: calibrate σ and init accountant before the loop starts.
        self._setup_dp()

        if not self._load_checkpoint() and self.eval_interval > 0:
            self.gv.logger.info("--------------Initial Evaluation--------------")
            self.gv.logger.time_start('Eval Time Cost')
            self.gv.logger.time_end('Eval Time Cost')

        while True:
            if self._if_exit():
                break
            self.gv.clock.step()
            updated = self.iterate()
            if updated is True or updated is None:
                self.gv.logger.info("--------------Round {}--------------".format(self.current_round))
                if self.gv.logger.check_if_log(self.current_round, self.eval_interval):
                    self.gv.logger.time_start('Eval Time Cost')
                    self.gv.logger.time_end('Eval Time Cost')
                    self._save_checkpoint()
                if self.gv.logger.early_stop():
                    break

                if self.current_round >= 0:
                    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    filename = f"x-lora_{current_time}_round{self.current_round}.bin"
                    save_path = os.path.join("./checkpoints", filename)
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save(_lora_state_only(self.model.model.state_dict()), save_path)
                self.current_round += 1
                self.global_lr_scheduler(self.current_round)

        self.gv.logger.info("=================End==================")
        self.gv.logger.time_end('Total Time Cost')
        self.gv.logger.save_output_as_json()

        if self._rdp_accountant is not None:
            final_eps = self._rdp_accountant.get_epsilon()
            print(
                f'[fedrag_lora.Server] Final privacy: eps_spent={final_eps:.4f} '
                f'/ target={self.option.get("target_epsilon", 20.0)}, '
                f'δ={self.option.get("target_delta", 1e-5)}.'
            )

        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"x-lora_{current_time}.bin"
        torch.save(_lora_state_only(self.model.model.state_dict()), filename)
        return

    def aggregate(self, model_old, models: list, *args, **kwargs):
        """Average LoRA-only weights across clients; preserve frozen base.
        Tick the RDP accountant once per round when DP is enabled.
        """
        if len(models) == 0:
            return model_old

        # Collect LoRA-only state_dicts from each client.
        lora_dicts = [_lora_state_only(m.state_dict()) for m in models]

        # Sanity: every client must ship the same set of LoRA keys.
        keys = set(lora_dicts[0].keys())
        for i, d in enumerate(lora_dicts[1:], start=1):
            if set(d.keys()) != keys:
                raise RuntimeError(
                    f'[fedrag_lora.Server.aggregate] LoRA key mismatch on '
                    f'client {i}: missing={keys - set(d.keys())}, '
                    f'extra={set(d.keys()) - keys}'
                )

        # Plain mean across clients (Phase-1: no server-side DP, no weighting).
        K = len(lora_dicts)
        averaged = {}
        for k in keys:
            stacked = torch.stack([d[k].float() for d in lora_dicts], dim=0)
            averaged[k] = stacked.mean(dim=0)

        # Apply averaged LoRA back into model_old. Server.model is a
        # fedllm.Model FModule wrapper whose state_dict prefixes every key with
        # 'model.', while Client packs state_dict from PeftModel directly
        # (no prefix). Detect and adjust.
        model_old_sd = model_old.state_dict()
        prefix = ''
        sample_key = next(iter(averaged))
        if sample_key not in model_old_sd and ('model.' + sample_key) in model_old_sd:
            prefix = 'model.'
        for k, v in averaged.items():
            full_k = prefix + k
            if full_k not in model_old_sd:
                continue   # base weights or stale/extra keys: leave alone
            target = model_old_sd[full_k]
            model_old_sd[full_k] = v.to(device=target.device, dtype=target.dtype)
        model_old.load_state_dict(model_old_sd, strict=False)

        print(f'[fedrag_lora.Server.aggregate] Averaged {len(averaged)} LoRA tensors '
              f'across K={K} clients (base weights untouched).')

        # Phase 2: tick the privacy accountant once per round.
        if getattr(self, '_rdp_accountant', None) is not None:
            self._rdp_accountant.step()
            eps_spent = self._rdp_accountant.get_epsilon()
            target_eps = self.option.get('target_epsilon', 20.0)
            print(
                f'[fedrag_lora.Server] [Privacy] eps_spent={eps_spent:.4f} '
                f'/ {target_eps}'
            )

        return model_old


# ══════════════════════════════════════════════════════════════════════════════
#   Client
# ══════════════════════════════════════════════════════════════════════════════

class Client(BasicClient):
    def __init__(self, option={}):
        super().__init__(option)
        # BasicClient.__init__ hardcodes self.model = BertModel.from_pretrained(...)
        # (see fedbase.py:769) which bypasses our PEFT wrapping. Replace it here so
        # Client.self.model has LoRA adapters and matches Server.model.model.
        # Don't .to(device) — flgo's runner handles device placement later.
        from flgo.benchmark.fedrag_classification.config import get_model
        self.model = get_model()

    def train(self, model, local_model):
        """Train local LoRA adapters using the global model as KD teacher.

        Branches on option['dp_enabled']:
          * False (Phase 1) — standard backward+step (mirror fedrag.Client.train).
          * True  (Phase 2) — per-sample clip + Gaussian noise on LoRA gradients
            only (mirror main_dp_lora_eps20.py:100-127).
        """
        local_model.train()
        optimizer = self.calculator.get_optimizer(
            local_model,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            momentum=self.momentum,
        )

        model.to(self.device)
        local_model.to(self.device)

        # Sync local LoRA from the global model (mirror main_dp_lora_eps20.py:80).
        global_lora_state = _lora_state_only(model.state_dict())
        local_model.load_state_dict(global_lora_state, strict=False)

        if self.option.get('dp_enabled', False):
            self._train_dp(model, local_model, optimizer)
        else:
            self._train_plain(model, local_model, optimizer)

    def _train_plain(self, model, local_model, optimizer):
        """Phase-1 non-DP training (mirror fedrag.Client.train:97-133)."""
        for it in range(self.num_steps):
            batch_data = self.get_batch_data()
            local_model.zero_grad()
            server_loss = self.calculator.compute_server_loss(model, batch_data)
            client_loss, client_only, server_only = self.calculator.compute_client_loss(
                server_loss, local_model, batch_data
            )
            print(
                f"client running:{it}/{self.num_steps}, client loss: {client_loss}, "
                f"loss 1: {client_only}, loss 2: {server_only}"
            )
            client_loss.backward()
            optimizer.step()

            if it == self.num_steps - 1:
                print(f"server loss: {server_loss}")

    def _train_dp(self, model, local_model, optimizer):
        """Phase-2 per-sample DP-SGD on LoRA params only.

        For each step:
          1. Get a batch of size `bs`.
          2. For each sample i in [0, bs):
               * forward + backward
               * compute per-sample L2 grad norm
               * clip: g_i ← g_i · min(1, C / ‖g_i‖)
               * accumulate
          3. Add Gaussian noise N(0, σ²·C²·I), normalize by bs:
                p.grad = (Σ_i g̃_i + ξ) / bs
          4. Post-noise stability: clip total grad norm to 1.0
          5. optimizer.step()

        DP-SGD privacy: per-sample sensitivity ≤ C, Gaussian mechanism
        with σ adds N(0, (σ·C)²) noise → (α, α·C²/(2σ²·C²))=(α, α/(2σ²))-RDP
        per step, composed over T steps and amplified by sampling rate q.
        """
        params = [p for p in local_model.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError(
                '[fedrag_lora.Client._train_dp] no trainable params found; '
                'is config.get_model() returning a PEFT-wrapped model?'
            )

        clip_norm = float(self.option.get('dp_clip_norm', 0.1))
        sigma = _calibrate_sigma_quiet(self.option)
        if sigma is None:
            raise RuntimeError(
                '[fedrag_lora.Client._train_dp] σ not calibrated; '
                'check option[\'dp_enabled\'] and target_epsilon.'
            )

        log_every = max(1, self.num_steps // 5)

        for step in range(self.num_steps):
            batch_data = self.get_batch_data()
            bs = _get_batch_size(batch_data)
            if bs == 0:
                continue

            accumulated = [torch.zeros_like(p.data) for p in params]

            for i in range(bs):
                single = _get_single_sample(batch_data, i)
                local_model.zero_grad()

                # NOTE: compute_client_loss is in-batch contrastive
                # (CrossEntropy on cosine sim + MSE distillation). On a
                # single sample the contrastive part degenerates
                # (label=[0], 1×1 logits), but DP-SGD only requires
                # per-sample sensitivity bounds — degeneracy is acceptable.
                # Phase 3 swaps loss to InfoNCE+KL but keeps this loop shape.
                server_loss = self.calculator.compute_server_loss(model, single)
                client_loss, _, _ = self.calculator.compute_client_loss(
                    server_loss, local_model, single
                )
                client_loss.backward()

                # Per-sample L2 norm across all LoRA gradients.
                per_sample_norm = _compute_grad_norm(params)
                clip_coef = min(1.0, clip_norm / (per_sample_norm + 1e-8))

                for j, p in enumerate(params):
                    if p.grad is not None:
                        accumulated[j] += p.grad.detach() * clip_coef

            # Gaussian noise on accumulated; divide by bs to recover average.
            for j, p in enumerate(params):
                noise = torch.randn_like(accumulated[j]) * (sigma * clip_norm)
                p.grad = (accumulated[j] + noise) / float(bs)

            # Post-noise stability clip (mirror main_dp_lora_eps20.py:130).
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            if step % log_every == 0 or step == self.num_steps - 1:
                total_norm = _compute_grad_norm(params)
                print(
                    f'[DP-Client {self.id}] step {step}/{self.num_steps}, '
                    f'σ={sigma:.4f}, C={clip_norm}, ‖∇‖={total_norm:.4f}'
                )

    def reply(self, svr_pkg):
        model = self.unpack(svr_pkg)
        self.train(model, self.model)
        cpkg = self.pack(self.model)
        return cpkg

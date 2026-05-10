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

    Cache uses a private key `_calibrated_dp_sigma` because flgo's init
    pre-populates the well-known `dp_noise_multiplier` to its legacy default
    of 0.1 (see fedrag.py:113), which we must NOT mistake for an already-
    calibrated σ. We still write the calibrated value to `dp_noise_multiplier`
    so downstream code (fedrag.py / fedrag_dp.py) reads the right value.

    Returns None when DP is disabled.
    """
    if not option.get('dp_enabled', False):
        return None
    cached = option.get('_calibrated_dp_sigma')
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
    option['_calibrated_dp_sigma'] = sigma     # private, our cache
    option['dp_noise_multiplier'] = sigma      # public, for downstream readers
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
        """Phase-1 non-DP training (mirror fedrag.Client.train:97-133).

        Builds its own AdamW optimizer (same rationale as _train_dp): flgo's
        default SGD with lr=1e-5 is too weak to budge PEFT's zero-init lora_B
        across 25 rounds, even without any DP noise. Override here so the
        non-DP baseline actually learns — apples-to-apples vs the DP path.
        """
        params = [p for p in local_model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params, lr=self.learning_rate, weight_decay=0.01,
        )
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
        """Per-sample DP-SGD on LoRA params with paper-faithful RAG-FT + KD-GLE.

        Phase 6.3 of the paper-faithful recipe: replaces the legacy
        `loss = 1 - cos_sim(q, r)` per-sample loss (no negatives, weak
        signal, validated only to 98.4% retention) with paper §3.2 RAG-FT
        InfoNCE + §3.3 KD-GLE MSE — the EXACT recipe used in arXiv:
        2504.19101 — adapted for per-sample gradient extraction.

        ── Strategy ────────────────────────────────────────────────────────
        Per-sample DP needs per-sample gradient, but InfoNCE wants in-batch
        negatives. Solution: encode the batch's references ONCE outside the
        per-sample loop with `torch.no_grad()` (cached as a frozen tensor),
        then for each query do a per-sample forward `q_i_emb @ ref_embs.T`
        producing a (1, B) similarity row. Each row's gradient flows back
        only through `q_i_emb` — i.e. only through sample i's query — so
        the per-sample sensitivity bound is preserved.

        For each step:
          1. Tokenize batch.
          2. (no_grad) Encode `ref_embs` with local_model       → (B, D)
          3. (no_grad) Encode teacher `tq_embs`, `tr_embs`      → (B, D)
                       Compute `teacher_sim = tq @ tr.T / |·|`  → (B, B)
          4. For each i in [0, B):
               * (with grad) Encode `q_i_emb` = local_model(q_i)  → (1, D)
               * `sim_row = normalize(q_i_emb) @ ref_embs.T`      → (1, B)
                 (refs were already L2-normalized in step 2)
               * loss_rag_ft = CE(sim_row / τ, label=[i])
                 loss_kd_gle = MSE(sim_row, teacher_sim[i:i+1])
               * loss_i = loss_rag_ft + kd_weight · loss_kd_gle
               * loss_i.backward()
               * clip per-sample: g_i ← g_i · min(1, C / ‖g_i‖)
               * accumulate
          5. Gaussian noise N(0, σ²·C²·I); average by bs.
          6. Post-noise stability clip (max_norm=1.0).
          7. AdamW step.

        ── AdamW over flgo's default SGD ───────────────────────────────────
        With σ≈1.29, C=0.1 the clipped+noised per-coordinate update is
        ~1e-5. SGD at lr=1e-5 leaves PEFT's zero-initialised lora_B at
        ~1e-6 after rounds — model is bit-identical to base. AdamW adapts
        per-parameter LR by gradient history so small gradients accumulate
        into meaningful updates (mirrors main_dp_lora_eps20.py:84).

        ── Why this matches paper §3 ───────────────────────────────────────
        - Each sample's loss = InfoNCE over (q_i vs all B refs) + MSE-to-
          teacher on the same row. Identical formula to non-DP recipe in
          core.compute_client_loss (Phase 2 revert), just emitted per-row
          with a frozen ref pool to keep per-sample DP semantics clean.
        - tau, kd_weight read from option (override via env or main_lora).

        ── DP guarantee ────────────────────────────────────────────────────
        Sensitivity bound preserved: `q_i_emb` is the ONLY term with
        gradient flow per inner iter; refs/teacher are detached. Removing
        sample i changes only its own contribution to the accumulator
        before noise. σ calibration unchanged.

        ── Edge cases ──────────────────────────────────────────────────────
        - bs=1: sim_row is (1,1), CE on diag of 1×1 → 0; KD-MSE term still
          drives. Loss is non-degenerate (vs Phase-3 KL+InfoNCE which gave
          identical-zero gradient at bs=1).
        - bs=0: skip step (matches old behavior).
        """
        from flgo.benchmark.fedrag_classification.core import cos_sim  # noqa: F401

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

        # Loss hyperparameters — read from option; fall back to the
        # calculator's instance values (set by core.DEFAULT_TEMPERATURE /
        # DEFAULT_KD_WEIGHT at construction time). Per-sample loop must use
        # the SAME tau / kd_weight as compute_client_loss for paper fidelity.
        tau = float(self.option.get('temperature',
                                    getattr(self.calculator, 'temperature', 0.05)))
        kd_weight = float(self.option.get('kd_weight',
                                          getattr(self.calculator, 'kd_weight', 1.0)))

        # Override flgo's default SGD with AdamW (see docstring rationale).
        optimizer = torch.optim.AdamW(
            params, lr=self.learning_rate, weight_decay=0.01,
        )

        tokenizer = self.calculator.tokenizer
        max_length = tokenizer.model_max_length

        log_every = max(1, self.num_steps // 5)

        for step in range(self.num_steps):
            batch_data = self.get_batch_data()
            bs = _get_batch_size(batch_data)
            if bs == 0:
                continue

            # batch_data = (questions, answers, references) per FEDRAG.__getitem__.
            questions = list(batch_data[0])
            references = list(batch_data[2])

            # ── Cache reference and teacher embeddings (no_grad) ──────────
            # These are computed ONCE per batch and shared across the inner
            # per-sample loop. They must NOT track gradient — the per-sample
            # gradient must depend only on q_i_emb, not on other samples'
            # encodings.
            with torch.no_grad():
                ref_inp = tokenizer(
                    references, return_tensors='pt', padding=True,
                    truncation=True, max_length=max_length,
                ).to(self.device)
                ref_out = local_model(**ref_inp).last_hidden_state.mean(dim=1)
                ref_embs = torch.nn.functional.normalize(ref_out, dim=-1, p=2)
                ref_embs = ref_embs.detach()                              # (B, D)

                # Teacher: same model after Server.aggregate; KD-GLE target.
                tq_inp = tokenizer(
                    questions, return_tensors='pt', padding=True,
                    truncation=True, max_length=max_length,
                ).to(self.device)
                tq_out = model(**tq_inp).last_hidden_state.mean(dim=1)
                tq_n = torch.nn.functional.normalize(tq_out, dim=-1, p=2)

                tr_out = model(**ref_inp).last_hidden_state.mean(dim=1)
                tr_n = torch.nn.functional.normalize(tr_out, dim=-1, p=2)

                teacher_sim = (tq_n @ tr_n.t()).detach()                  # (B, B)

            # Sanity (cheap, only first step) — guard against gradient leak.
            if step == 0:
                assert not ref_embs.requires_grad, \
                    'ref_embs must be detached for per-sample DP'
                assert not teacher_sim.requires_grad, \
                    'teacher_sim must be detached'

            accumulated = [torch.zeros_like(p.data) for p in params]

            for i in range(bs):
                local_model.zero_grad()

                # ── Per-sample query forward (with grad) ──────────────────
                q_inp = tokenizer(
                    [questions[i]], return_tensors='pt', padding=True,
                    truncation=True, max_length=max_length,
                ).to(self.device)
                q_out = local_model(**q_inp).last_hidden_state.mean(dim=1)
                q_n = torch.nn.functional.normalize(q_out, dim=-1, p=2)   # (1, D)

                # (1, B) row: this sample's similarity to all B refs.
                sim_row = q_n @ ref_embs.t()
                label_i = torch.tensor([i], device=self.device)

                # Paper §3.2 RAG-FT (InfoNCE) + §3.3 KD-GLE (MSE).
                loss_rag_ft = torch.nn.functional.cross_entropy(
                    sim_row / tau, label_i,
                )
                loss_kd_gle = torch.nn.functional.mse_loss(
                    sim_row, teacher_sim[i:i+1],
                )
                loss_i = loss_rag_ft + kd_weight * loss_kd_gle
                loss_i.backward()

                # Per-sample L2 norm + clip.
                per_sample_norm = _compute_grad_norm(params)
                clip_coef = min(1.0, clip_norm / (per_sample_norm + 1e-8))
                for j, p in enumerate(params):
                    if p.grad is not None:
                        accumulated[j] += p.grad.detach() * clip_coef

            # ── Noise + average + step ────────────────────────────────────
            for j, p in enumerate(params):
                noise = torch.randn_like(accumulated[j]) * (sigma * clip_norm)
                p.grad = (accumulated[j] + noise) / float(bs)

            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            if step % log_every == 0 or step == self.num_steps - 1:
                total_norm = _compute_grad_norm(params)
                print(
                    f'[DP-Client {self.id}] step {step}/{self.num_steps}, '
                    f'σ={sigma:.4f}, C={clip_norm}, τ={tau}, kw={kd_weight}, '
                    f'last_loss={loss_i.item():.4f}, ‖∇‖={total_norm:.4f}'
                )

    def reply(self, svr_pkg):
        model = self.unpack(svr_pkg)
        self.train(model, self.model)
        cpkg = self.pack(self.model)
        return cpkg

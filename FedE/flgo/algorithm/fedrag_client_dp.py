"""Client-level DP for the tuned FedAvg-LoRA+KD pipeline.

Privacy unit: one complete federated client. Adjacency is replace-one over a
fixed, public client roster and fixed public aggregation weights. Each local
LoRA update is clipped once across the full adapter vector. The server adds
one Gaussian noise vector to the weighted aggregate and accounts exactly one
mechanism release per successful round.

Controls:
  * noise0:    same delta-aggregation path, no clipping, no noise.
  * clip_only: fixed client-delta clipping, no noise and no DP claim.
  * dp:        fixed clipping + calibrated Gaussian noise + RDP accounting.
"""

from __future__ import annotations

import json
import secrets

import torch

from client_dp_utils import (
    add_gaussian_noise,
    clip_state_delta,
    normalize_weights,
    replacement_sensitivity,
    state_delta,
    weighted_sum_states,
)
from .fedrag_lora import (
    Client,
    Server as LoRAServer,
    _lora_state_only,
)
from privacy.rdp_accountant import (
    RDPAccountant,
    compute_epsilon,
    find_noise_multiplier,
)


VALID_MODES = {"noise0", "clip_only", "dp"}


class Server(LoRAServer):
    def _setup_dp(self):
        if getattr(self, "_client_dp_initialized", False):
            return
        self._client_dp_initialized = True

        mode = str(self.option.get("client_dp_mode", "noise0")).lower()
        if mode not in VALID_MODES:
            raise ValueError(
                f"client_dp_mode must be one of {sorted(VALID_MODES)}, got {mode}"
            )
        self.client_dp_mode = mode
        self.client_dp_clip_norm = float(
            self.option.get("client_dp_clip_norm", 1.0)
        )
        if self.client_dp_clip_norm <= 0:
            raise ValueError("client_dp_clip_norm must be positive")

        self._client_dp_release_count = 0
        self._client_dp_round_reports = []
        self._rdp_accountant = None
        self._dp_sigma = None
        self._client_dp_noise_generator = None

        if mode != "dp":
            print(
                f"[fedrag_client_dp.Server] control mode={mode}; "
                "no formal DP claim and accountant disabled."
            )
            return

        n_clients = int(self.option.get("num_clients", len(self.clients)))
        clients_per_round = int(
            self.option.get("client_dp_clients_per_round", n_clients)
        )
        if not 1 <= clients_per_round <= n_clients:
            raise ValueError(
                "client_dp_clients_per_round must be in [1, num_clients]"
            )
        sample_rate = clients_per_round / n_clients
        target_epsilon = float(
            self.option.get("client_dp_target_epsilon", 20.0)
        )
        target_delta = float(
            self.option.get("client_dp_target_delta", 1e-5)
        )
        rounds = int(self.option.get("num_rounds", 15))
        configured_sigma = self.option.get("client_dp_noise_multiplier")
        if configured_sigma is None:
            sigma = find_noise_multiplier(
                target_epsilon=target_epsilon,
                num_steps=rounds,
                sample_rate=sample_rate,
                delta=target_delta,
            )
        else:
            sigma = float(configured_sigma)
            if sigma <= 0:
                raise ValueError(
                    "client_dp_noise_multiplier must be positive in dp mode"
                )

        epsilon_check, best_alpha = compute_epsilon(
            rounds, sigma, sample_rate, target_delta
        )
        if epsilon_check > target_epsilon + 0.02:
            raise ValueError(
                f"configured sigma={sigma} gives epsilon={epsilon_check:.4f}, "
                f"above target={target_epsilon}"
            )

        self._dp_sigma = sigma
        self._rdp_accountant = RDPAccountant(
            noise_multiplier=sigma,
            sample_rate=sample_rate,
            delta=target_delta,
        )
        self._client_dp_noise_generator = torch.Generator(device="cpu")
        self._client_dp_noise_generator.manual_seed(secrets.randbits(63))
        print(
            f"[fedrag_client_dp.Server] DP calibrated: sigma={sigma:.6f}, "
            f"epsilon={epsilon_check:.4f}, target={target_epsilon}, "
            f"delta={target_delta}, q={sample_rate:.4f}, T={rounds}, "
            f"best_alpha={best_alpha}, adjacency=replace-one."
        )

    def _received_weights(self, count: int) -> tuple[list[int], list[float]]:
        received = getattr(self, "received_clients", None)
        if received is None or len(received) != count:
            raise RuntimeError(
                "received client IDs must align with client model payloads"
            )
        sizes = [int(self.clients[cid].datavol) for cid in received]
        return sizes, normalize_weights(sizes)

    @staticmethod
    def _aligned_global_state(model_old, client_keys):
        model_state = model_old.state_dict()
        aligned = {}
        key_map = {}
        for key in client_keys:
            if key in model_state:
                model_key = key
            elif f"model.{key}" in model_state:
                model_key = f"model.{key}"
            else:
                raise RuntimeError(
                    f"global model does not contain client adapter key {key}"
                )
            aligned[key] = model_state[model_key].detach().float().cpu()
            key_map[key] = model_key
        return aligned, key_map

    def aggregate(self, model_old, models: list, *args, **kwargs):
        if not models:
            return model_old
        if not getattr(self, "_client_dp_initialized", False):
            self._setup_dp()

        client_states = [_lora_state_only(model.state_dict()) for model in models]
        client_keys = list(client_states[0])
        global_state, key_map = self._aligned_global_state(model_old, client_keys)
        raw_deltas = [
            state_delta(client_state, global_state)
            for client_state in client_states
        ]
        sizes, weights = self._received_weights(len(raw_deltas))

        norms = []
        coefficients = []
        if self.client_dp_mode == "noise0":
            processed_deltas = raw_deltas
            for delta in raw_deltas:
                norm = sum(
                    tensor.detach().double().pow(2).sum().item()
                    for tensor in delta.values()
                ) ** 0.5
                norms.append(norm)
                coefficients.append(1.0)
        else:
            processed_deltas = []
            for delta in raw_deltas:
                clipped, norm, coefficient = clip_state_delta(
                    delta, self.client_dp_clip_norm
                )
                processed_deltas.append(clipped)
                norms.append(norm)
                coefficients.append(coefficient)

        aggregate_delta = weighted_sum_states(processed_deltas, weights)
        sensitivity = 0.0
        noise_std = 0.0
        if self.client_dp_mode == "dp":
            sensitivity = replacement_sensitivity(
                self.client_dp_clip_norm, weights
            )
            noise_std = self._dp_sigma * sensitivity
            aggregate_delta = add_gaussian_noise(
                aggregate_delta,
                noise_std,
                generator=self._client_dp_noise_generator,
            )
            self._rdp_accountant.step()
            self._client_dp_release_count += 1

        model_state = model_old.state_dict()
        for key, delta in aggregate_delta.items():
            model_key = key_map[key]
            target = model_state[model_key]
            updated = global_state[key] + delta
            model_state[model_key] = updated.to(
                device=target.device, dtype=target.dtype
            )
        model_old.load_state_dict(model_state, strict=False)

        epsilon = (
            self._rdp_accountant.get_epsilon()
            if self._rdp_accountant is not None
            else None
        )
        report = {
            "round_release": self._client_dp_release_count,
            "mode": self.client_dp_mode,
            "client_ids": list(self.received_clients),
            "datavol": sizes,
            "weights": weights,
            "raw_update_norms": norms,
            "clip_coefficients": coefficients,
            "clip_norm": (
                self.client_dp_clip_norm
                if self.client_dp_mode != "noise0"
                else None
            ),
            "adjacency": "replace-one",
            "sensitivity": sensitivity,
            "noise_multiplier": self._dp_sigma,
            "noise_std": noise_std,
            "epsilon_spent": epsilon,
            "delta": self.option.get("client_dp_target_delta", 1e-5),
        }
        self._client_dp_round_reports.append(report)
        print(
            "[fedrag_client_dp.Server] "
            f"mode={self.client_dp_mode}, norms="
            f"{[round(value, 6) for value in norms]}, "
            f"clip_coef={[round(value, 6) for value in coefficients]}, "
            f"noise_std={noise_std:.8f}, epsilon={epsilon}"
        )
        return model_old

    def run(self):
        result = super().run()
        report = {
            "privacy_unit": "one complete federated client",
            "adjacency": "replace-one; fixed public roster and datavol weights",
            "mode": self.client_dp_mode,
            "formal_dp_claim": self.client_dp_mode == "dp",
            "mechanism_releases": self._client_dp_release_count,
            "accountant_steps": (
                self._rdp_accountant.get_privacy_report()["num_steps"]
                if self._rdp_accountant is not None
                else 0
            ),
            "final_epsilon": (
                self._rdp_accountant.get_epsilon()
                if self._rdp_accountant is not None
                else None
            ),
            "target_delta": self.option.get("client_dp_target_delta", 1e-5),
            "rounds": self._client_dp_round_reports,
        }
        with open("client_dp_report.json", "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        return result

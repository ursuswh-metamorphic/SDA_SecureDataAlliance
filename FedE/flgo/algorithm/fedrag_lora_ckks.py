"""
fedrag_lora_ckks.py — Phase 4: true homomorphic aggregation on LoRA ciphertexts.

Architecture (compared to fedrag_CKKS.py which decrypts on server):
  * CKKS context lives module-level. The SECRET key stays on clients;
    the SERVER receives a public-only context (deserialised with
    save_secret_key=False).
  * Client.pack encrypts the local LoRA-adapter state, **pre-divided by K**.
  * Server.aggregate sums those K ciphertexts homomorphically. Because the
    clients pre-divided, the homomorphic sum directly yields the mean — the
    server never multiplies (which would consume a CKKS modulus level).
  * Server.aggregate asserts `not self._public_ctx.has_secret_key()` every
    round to guard the cryptographic invariant.

Trust model (single-tenant Vast.ai PoC):
  * One CKKS keypair shared across all clients (module-level singleton).
  * Server is an honest-but-curious aggregator with the public/eval key only.
  * Production deployment would have each tenant generate its own keypair;
    the homomorphic-sum property survives intact under multi-key CKKS.

Mod chain `[60, 40, 40, 60]` (200 bits) is identical to fedrag_CKKS.py and
sits below the 218-bit cap for poly_modulus_degree=8192. Because clients
pre-divide by K before encrypting, the server never performs a plaintext-mul,
so two mul levels (one for the implicit rescale at encrypt, one for safety)
are sufficient. If future per-client weighting (plaintext-mul) is added,
move to poly_modulus_degree=16384 to allow a longer mod chain.

flgo dispatch contract is preserved: pack/unpack/aggregate stay on the
'model' key but the value is the dict
    {'cipher': list[bytes], 'manifest': list[dict]}
instead of a torch.nn.Module.
"""
from __future__ import annotations

import copy
import os
from typing import List, Dict

import numpy as np
import torch

from . import fedrag_lora
from .fedrag_lora import (
    Server as _LoraServer,
    Client as _LoraClient,
    _is_lora_key,
    _lora_state_only,
)

# CKKS primitives live in a flgo-independent module so the correctness test
# can verify them without triggering flgo's heavy __init__ chain.
from .fedrag_ckks_primitives import (
    POLY_MODULUS_DEGREE,
    COEFF_MOD_BIT_SIZES,
    GLOBAL_SCALE,
    CHUNK_SIZE,
    _make_ckks_context,
    _ensure_ctx,
    reset_ctx,
    _chunked_encrypt,
    _chunked_decrypt,
    _homomorphic_sum_ciphertexts,
    _manifest_signature,
    _TENSEAL_AVAILABLE,
)


# ══════════════════════════════════════════════════════════════════════════════
#   Server — aggregates on ciphertexts, never decrypts
# ══════════════════════════════════════════════════════════════════════════════

class Server(_LoraServer):
    """Subclass of fedrag_lora.Server that ships LoRA encrypted under CKKS
    and aggregates homomorphically. Inherits Phase 1 LoRA mechanics + Phase 2
    DP accountant from the parent."""

    def _ensure_ckks_setup(self):
        if getattr(self, '_ckks_setup', False):
            return
        self._ckks_setup = True
        _, public_ctx = _ensure_ctx()
        # Server gets the PUBLIC context only. The secret key lives on clients.
        self._public_ctx = public_ctx
        # Cryptographic invariant — re-asserted in aggregate() each round.
        assert not self._public_ctx.has_secret_key(), (
            'fedrag_lora_ckks.Server initialised with a context that holds '
            'the CKKS secret key. This breaks the FHE threat model.'
        )
        self._last_aggregated_cipher: List[bytes] | None = None
        self._manifest: List[Dict] | None = None
        self._client_K = int(self.option.get('num_clients', 5))
        print(
            f'[fedrag_lora_ckks.Server] CKKS setup: '
            f'poly_modulus_degree={POLY_MODULUS_DEGREE}, '
            f'mod_chain={COEFF_MOD_BIT_SIZES}, '
            f'scale=2^{int(np.log2(GLOBAL_SCALE))}, '
            f'K={self._client_K}.'
        )

    def pack(self, client_id, mtype=0, *args, **kwargs):
        """Send the (encrypted) global LoRA state to a client."""
        self._ensure_ckks_setup()
        if self._last_aggregated_cipher is None:
            # Round 0: encrypt server's initial LoRA state.
            # Clients will decrypt with their secret key. The server has the
            # plaintext at construction time (it built the model), so this
            # broadcast does not leak anything not already public.
            initial_lora = _lora_state_only(self.model.state_dict())
            cipher, manifest = _chunked_encrypt(initial_lora, self._public_ctx)
            self._last_aggregated_cipher = cipher
            self._manifest = manifest
            print(
                f'[fedrag_lora_ckks.Server] Round 0 broadcast: '
                f'{len(cipher)} chunks, '
                f'{sum(len(c) for c in cipher) / 1024:.1f} KB total.'
            )
        return {"model": {
            "cipher": self._last_aggregated_cipher,
            "manifest": self._manifest,
        }}

    def aggregate(self, model_old, models: list, *args, **kwargs):
        """Homomorphic sum across clients. NEVER decrypts on server."""
        self._ensure_ckks_setup()
        if not models:
            return model_old

        client_ciphers = [m["cipher"] for m in models]
        manifests = [m["manifest"] for m in models]

        # Verify all clients agree on the manifest signature (key set + chunk lengths).
        ref_sig = _manifest_signature(manifests[0])
        for i, m in enumerate(manifests[1:], start=1):
            if _manifest_signature(m) != ref_sig:
                raise RuntimeError(
                    f'[fedrag_lora_ckks.Server.aggregate] Manifest mismatch '
                    f'on client {i}.'
                )

        # Cryptographic guard: server still has no secret key.
        assert not self._public_ctx.has_secret_key(), (
            'CRITICAL: server CKKS context now holds the secret key!'
        )

        # Homomorphic sum across clients. Clients pre-divided their state by K
        # so the sum directly equals the mean (no plaintext-mul on server).
        summed = _homomorphic_sum_ciphertexts(self._public_ctx, client_ciphers)
        self._last_aggregated_cipher = summed
        self._manifest = manifests[0]

        K = len(client_ciphers)
        n_bytes = sum(len(c) for c in summed)
        print(
            f'[fedrag_lora_ckks.Server.aggregate] Homomorphic-summed '
            f'{len(summed)} ciphertext chunks across K={K} clients '
            f'(no decrypt; {n_bytes / 1024:.1f} KB aggregated payload).'
        )

        # Inherited from Phase 2: tick the RDP accountant.
        if getattr(self, '_rdp_accountant', None) is not None:
            self._rdp_accountant.step()
            eps_spent = self._rdp_accountant.get_epsilon()
            target_eps = self.option.get('target_epsilon', 20.0)
            print(
                f'[fedrag_lora_ckks.Server] [Privacy] '
                f'eps_spent={eps_spent:.4f} / {target_eps}'
            )

        # model_old (server's plaintext copy) is intentionally NOT updated.
        # Server has no plaintext access to the trained LoRA in Phase 4.
        # Final evaluation requires a client (with secret key) to decrypt
        # `self._last_aggregated_cipher`.
        return model_old

    def save_final_cipher(self, path: str):
        """Persist the final aggregated cipher to disk for offline decryption."""
        if self._last_aggregated_cipher is None:
            print(f'[fedrag_lora_ckks.Server.save_final_cipher] no cipher to save.')
            return
        os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
        torch.save({
            'cipher': self._last_aggregated_cipher,
            'manifest': self._manifest,
            'ckks_params': {
                'poly_modulus_degree': POLY_MODULUS_DEGREE,
                'coeff_mod_bit_sizes': COEFF_MOD_BIT_SIZES,
                'global_scale': GLOBAL_SCALE,
                'chunk_size': CHUNK_SIZE,
            },
        }, path)
        print(f'[fedrag_lora_ckks.Server.save_final_cipher] saved to {path}')


# ══════════════════════════════════════════════════════════════════════════════
#   Client — holds secret key, decrypts incoming, encrypts outgoing
# ══════════════════════════════════════════════════════════════════════════════

class Client(_LoraClient):
    def _ensure_ckks_setup(self):
        if getattr(self, '_ckks_setup', False):
            return
        self._ckks_setup = True
        secret_ctx, public_ctx = _ensure_ctx()
        self._secret_ctx = secret_ctx
        self._public_ctx = public_ctx
        self._client_K = int(self.option.get('num_clients', 5))

    def unpack(self, received_pkg):
        """Decrypt the incoming aggregated cipher, load into self.model.
        Return a frozen deepcopy as the KD teacher for compute_server_loss."""
        self._ensure_ckks_setup()
        payload = received_pkg["model"]
        cipher = payload["cipher"]
        manifest = payload["manifest"]
        lora_state = _chunked_decrypt(cipher, manifest, self._secret_ctx)
        # Cast back to the local model's dtype/device before loading.
        local_sd = self.model.state_dict()
        casted = {}
        for k, v in lora_state.items():
            if k in local_sd:
                target = local_sd[k]
                casted[k] = v.to(device=target.device, dtype=target.dtype)
            else:
                casted[k] = v
        self.model.load_state_dict(casted, strict=False)

        # KD teacher: a frozen, no-grad snapshot of the just-loaded global state.
        teacher = copy.deepcopy(self.model)
        for p in teacher.parameters():
            p.requires_grad = False
        teacher.eval()
        return teacher

    def pack(self, model, *args, **kwargs):
        """Encrypt local LoRA state pre-divided by K so server can pure-add."""
        self._ensure_ckks_setup()
        K = float(self._client_K)
        lora_state = _lora_state_only(model.state_dict())
        scaled = {k: v.float() / K for k, v in lora_state.items()}
        cipher, manifest = _chunked_encrypt(scaled, self._public_ctx)
        return {"model": {"cipher": cipher, "manifest": manifest}}

    def reply(self, svr_pkg):
        teacher = self.unpack(svr_pkg)
        self.train(teacher, self.model)
        return self.pack(self.model)

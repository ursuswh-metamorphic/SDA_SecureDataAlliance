"""
fedrag_ckks_primitives.py — pure TenSEAL helpers for Phase 4.

Extracted to its own module so:
  * `test_ckks_correctness.py` can verify the math without triggering
    flgo's heavy `__init__.py` (which transitively imports `requests`,
    `transformers`, etc.).
  * `fedrag_lora_ckks.py` re-exports them for the Server/Client subclasses.

Dependencies: tenseal, numpy, torch — nothing from flgo.
"""
from __future__ import annotations

from typing import List, Tuple, Dict, Any

import numpy as np
import torch

try:
    import tenseal as ts
    _TENSEAL_AVAILABLE = True
    _TENSEAL_IMPORT_ERR = None
except ImportError as _e:  # pragma: no cover
    ts = None  # type: ignore
    _TENSEAL_AVAILABLE = False
    _TENSEAL_IMPORT_ERR = _e


# ── CKKS parameters ──────────────────────────────────────────────────────────
# Microsoft SEAL hard cap for poly_modulus_degree=8192 is 218 total bits in
# the coeff_modulus chain. We use 4 primes summing to 200 bits, identical to
# the original fedrag_CKKS.py chain. With client-side pre-divide-by-K we never
# perform a plaintext-mul on the server, so two mul levels (one for the scale
# rescale at encrypt, one for safety) suffice.
POLY_MODULUS_DEGREE = 8192
COEFF_MOD_BIT_SIZES = [60, 40, 40, 60]      # 200 bits total (≤ 218 cap)
GLOBAL_SCALE = 2 ** 40
CHUNK_SIZE = POLY_MODULUS_DEGREE // 2       # = 4096 slots per ciphertext


# Module-level singletons (single-tenant simulation).
_SECRET_CTX = None
_PUBLIC_CTX = None


def _require_tenseal():
    if not _TENSEAL_AVAILABLE:
        raise ImportError(
            f'tenseal is required for fedrag_ckks_primitives but is not '
            f'installed: {_TENSEAL_IMPORT_ERR}. Install with: pip install tenseal'
        )


def _make_ckks_context():
    """Create a fresh CKKS context with secret key + Galois keys."""
    _require_tenseal()
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=POLY_MODULUS_DEGREE,
        coeff_mod_bit_sizes=COEFF_MOD_BIT_SIZES,
    )
    ctx.global_scale = GLOBAL_SCALE
    ctx.generate_galois_keys()
    return ctx


def _ensure_ctx() -> Tuple[Any, Any]:
    """Return (secret_ctx, public_ctx) singletons; create on first call."""
    global _SECRET_CTX, _PUBLIC_CTX
    if _SECRET_CTX is None:
        _SECRET_CTX = _make_ckks_context()
        # Derive a public-only context by serialising without the secret key
        # then deserialising. This is the canonical way to ship public-only
        # ctx to a non-trusted aggregator.
        pub_bytes = _SECRET_CTX.serialize(save_secret_key=False)
        _PUBLIC_CTX = ts.context_from(pub_bytes)
    return _SECRET_CTX, _PUBLIC_CTX


def reset_ctx():
    """Reset the singleton — only used in tests."""
    global _SECRET_CTX, _PUBLIC_CTX
    _SECRET_CTX = None
    _PUBLIC_CTX = None


# ══════════════════════════════════════════════════════════════════════════════
#   Chunked encryption helpers
# ══════════════════════════════════════════════════════════════════════════════

def _chunked_encrypt(state_dict: Dict[str, torch.Tensor], ctx) -> Tuple[List[bytes], List[Dict]]:
    """Flatten each tensor, split into CHUNK_SIZE-slot chunks, encrypt each chunk.

    Args:
        state_dict: dict of name -> tensor (LoRA-only).
        ctx: TenSEAL context (public or secret); only public keys are used to encrypt.

    Returns:
        cipher_bytes: flat list of serialised ciphertext bytes.
        manifest: list of dicts with {'key', 'shape', 'numel', 'chunks': [(cipher_idx, length), ...]}.
    """
    _require_tenseal()
    cipher_bytes: List[bytes] = []
    manifest: List[Dict] = []
    for key, tensor in state_dict.items():
        flat = tensor.detach().cpu().float().flatten().numpy().astype(np.float64)
        n = int(flat.shape[0])
        n_chunks = (n + CHUNK_SIZE - 1) // CHUNK_SIZE
        chunk_indices: List[Tuple[int, int]] = []
        for i in range(n_chunks):
            start = i * CHUNK_SIZE
            end = min(start + CHUNK_SIZE, n)
            chunk = flat[start:end]
            length = int(end - start)
            # Zero-pad to full slot width so unused slots are deterministic.
            if length < CHUNK_SIZE:
                padded = np.zeros(CHUNK_SIZE, dtype=np.float64)
                padded[:length] = chunk
                chunk = padded
            ckks_vec = ts.ckks_vector(ctx, chunk.tolist())
            chunk_indices.append((len(cipher_bytes), length))
            cipher_bytes.append(ckks_vec.serialize())
        manifest.append({
            'key': key,
            'shape': tuple(int(s) for s in tensor.shape),
            'numel': int(tensor.numel()),
            'chunks': chunk_indices,
        })
    return cipher_bytes, manifest


def _chunked_decrypt(cipher_bytes: List[bytes], manifest: List[Dict], secret_ctx) -> Dict[str, torch.Tensor]:
    """Decrypt each chunk with secret_ctx, reassemble tensors per manifest."""
    _require_tenseal()
    state: Dict[str, torch.Tensor] = {}
    for entry in manifest:
        flat_values: List[float] = []
        for cipher_idx, length in entry['chunks']:
            ckks_vec = ts.ckks_vector_from(secret_ctx, cipher_bytes[cipher_idx])
            decrypted = ckks_vec.decrypt()
            flat_values.extend(decrypted[:length])
        # Defensive truncate to the exact numel.
        flat_values = flat_values[:entry['numel']]
        tensor = torch.tensor(flat_values, dtype=torch.float32).reshape(entry['shape'])
        state[entry['key']] = tensor
    return state


def _homomorphic_sum_ciphertexts(public_ctx, list_of_cipher_lists: List[List[bytes]]) -> List[bytes]:
    """Element-wise homomorphic sum across K clients' ciphertext lists.

    Each `list_of_cipher_lists[i]` is one client's full LoRA cipher list.
    All K lists must have identical length and chunk indexing.
    Returns a single aggregated cipher list of the same length.

    Server uses public_ctx; secret key is never required for addition.
    """
    _require_tenseal()
    K = len(list_of_cipher_lists)
    if K == 0:
        return []
    n_chunks = len(list_of_cipher_lists[0])
    summed_bytes: List[bytes] = []
    for chunk_idx in range(n_chunks):
        acc = ts.ckks_vector_from(public_ctx, list_of_cipher_lists[0][chunk_idx])
        for k in range(1, K):
            other = ts.ckks_vector_from(public_ctx, list_of_cipher_lists[k][chunk_idx])
            acc = acc + other  # homomorphic add (no secret key needed)
        summed_bytes.append(acc.serialize())
    return summed_bytes


def _manifest_signature(manifest: List[Dict]) -> List[Tuple]:
    """Stable representation for cross-client equality checks.
    Excludes the cipher_idx (depends on chunk ordering, identical given
    identical state_dict iteration order) but includes shape and length."""
    return [(e['key'], e['shape'], e['numel'], tuple(l for _, l in e['chunks']))
            for e in manifest]

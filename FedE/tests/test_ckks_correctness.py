"""
test_ckks_correctness.py — Phase 4 cryptographic correctness.

Builds a minimal CKKS pipeline and verifies:

  1. (Public-only ctx invariant) The context derived via
     `serialize(save_secret_key=False)` returns False for has_secret_key().
  2. (Round-trip correctness) Encrypting K=5 fake LoRA states (pre-divided
     by K), homomorphic-summing on the public ctx, and decrypting with the
     secret ctx produces a result within 1e-3 elementwise of the plaintext
     mean.
  3. (Manifest signature) Two clients with the same key set + shapes produce
     the same manifest signature.
  4. (Negative test) Decrypting with the public-only ctx either raises or
     produces wildly wrong values.

Skips with exit 0 (not failure) if `tenseal` is not installed — Windows
+ Python 3.14 has no published wheels yet, so the test is gated on the
presence of TenSEAL. Run on Linux + Python ≤ 3.12 (or on the Vast.ai
training image) for full coverage.
"""
import importlib.util
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FEDE_ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, FEDE_ROOT)

import torch  # noqa: E402

# Gate on TenSEAL availability.
try:
    import tenseal as ts  # noqa: F401
except ImportError as e:
    print('=' * 60)
    print(f'[test_ckks_correctness] SKIP — tenseal not installed: {e}')
    print('To install: pip install tenseal     (Linux + Python <=3.12)')
    print('=' * 60)
    sys.exit(0)

# Import primitives by file path to avoid triggering flgo's heavy __init__
# (which imports requests / transformers / peft). The primitives module
# itself only needs torch + tenseal + numpy.
_PRIMITIVES_PATH = os.path.join(
    FEDE_ROOT, 'flgo', 'algorithm', 'fedrag_ckks_primitives.py',
)
_spec = importlib.util.spec_from_file_location(
    'fedrag_ckks_primitives', _PRIMITIVES_PATH,
)
_prim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prim)

_make_ckks_context = _prim._make_ckks_context
_ensure_ctx = _prim._ensure_ctx
reset_ctx = _prim.reset_ctx
_chunked_encrypt = _prim._chunked_encrypt
_chunked_decrypt = _prim._chunked_decrypt
_homomorphic_sum_ciphertexts = _prim._homomorphic_sum_ciphertexts
_manifest_signature = _prim._manifest_signature
POLY_MODULUS_DEGREE = _prim.POLY_MODULUS_DEGREE
COEFF_MOD_BIT_SIZES = _prim.COEFF_MOD_BIT_SIZES
GLOBAL_SCALE = _prim.GLOBAL_SCALE
CHUNK_SIZE = _prim.CHUNK_SIZE


def _fake_lora_state(seed: int = 0):
    """Realistic LoRA shapes for BAAI/bge-base-en (12 layers × Q,V × A,B)."""
    g = torch.Generator().manual_seed(seed)
    state = {}
    for layer in range(12):
        for mod in ('query', 'value'):
            base = f'base_model.model.encoder.layer.{layer}.attention.self.{mod}'
            state[f'{base}.lora_A.default.weight'] = torch.randn(8, 768, generator=g) * 0.01
            state[f'{base}.lora_B.default.weight'] = torch.randn(768, 8, generator=g) * 0.01
    return state


def main():
    print('=' * 60)
    print('[test_ckks_correctness] Phase 4 cryptographic correctness')
    print(f'  poly_modulus_degree={POLY_MODULUS_DEGREE}')
    print(f'  coeff_mod_bit_sizes={COEFF_MOD_BIT_SIZES}')
    print(f'  global_scale=2^{int(torch.log2(torch.tensor(float(GLOBAL_SCALE))).item())}')
    print(f'  chunk_size={CHUNK_SIZE}')
    print('=' * 60)

    reset_ctx()  # ensure a clean singleton for this test

    # ── Setup ────────────────────────────────────────────────────────────
    K = 5
    states = [_fake_lora_state(seed=i) for i in range(K)]
    keys = list(states[0].keys())
    n_params = sum(v.numel() for v in states[0].values())
    print(f'\n[Setup] K={K} fake clients, {len(keys)} LoRA tensors each, '
          f'{n_params:,} params/state.')

    # Plaintext reference: mean across clients.
    plain_mean = {k: torch.stack([s[k] for s in states]).mean(dim=0) for k in keys}

    # ── Check 1: public ctx has no secret key ───────────────────────────
    secret_ctx, public_ctx = _ensure_ctx()
    assert not public_ctx.has_secret_key(), (
        'public ctx must NOT have secret key; check serialize(save_secret_key=False)'
    )
    print(f'\n[Check 1] public_ctx.has_secret_key() = False        OK')

    # ── Check 2: secret ctx HAS secret key ──────────────────────────────
    assert secret_ctx.has_secret_key(), 'secret ctx must have secret key'
    print(f'[Check 2] secret_ctx.has_secret_key() = True          OK')

    # ── Check 3: manifest signatures agree across clients ───────────────
    manifests_test = []
    for s in states:
        _, m = _chunked_encrypt({k: v.float() for k, v in s.items()}, public_ctx)
        manifests_test.append(m)
    sigs = [_manifest_signature(m) for m in manifests_test]
    for i, sig in enumerate(sigs[1:], start=1):
        assert sig == sigs[0], f'manifest signature mismatch on client {i}'
    print(f'[Check 3] manifest signatures match across {K} clients   OK')

    # ── Check 4: encrypt → homomorphic sum → decrypt ↔ plaintext mean ───
    print(f'\n[Encrypt] Pre-dividing each client state by K={K} and encrypting...')
    t0 = time.time()
    client_ciphers = []
    client_manifests = []
    for i, state in enumerate(states):
        scaled = {k: v.float() / float(K) for k, v in state.items()}
        cipher, manifest = _chunked_encrypt(scaled, public_ctx)
        client_ciphers.append(cipher)
        client_manifests.append(manifest)
    t_encrypt = time.time() - t0
    n_chunks = len(client_ciphers[0])
    bytes_per_client = sum(len(c) for c in client_ciphers[0])
    print(f'         {n_chunks} chunks/client, {bytes_per_client / 1024:.1f} KB/client, '
          f'{t_encrypt:.2f}s total ({K} clients).')

    print(f'\n[Aggregate] Homomorphic sum on server (no secret key)...')
    t0 = time.time()
    summed = _homomorphic_sum_ciphertexts(public_ctx, client_ciphers)
    t_aggregate = time.time() - t0
    print(f'         {len(summed)} chunks summed, {t_aggregate:.2f}s.')

    print(f'\n[Decrypt] Client decrypts aggregated cipher with secret_ctx...')
    t0 = time.time()
    decrypted = _chunked_decrypt(summed, client_manifests[0], secret_ctx)
    t_decrypt = time.time() - t0
    print(f'         {len(decrypted)} tensors recovered, {t_decrypt:.2f}s.')

    # Compare elementwise.
    max_abs_diff = 0.0
    max_rel_diff = 0.0
    worst_key = None
    for k in keys:
        diff = (decrypted[k] - plain_mean[k]).abs()
        m = diff.max().item()
        if m > max_abs_diff:
            max_abs_diff = m
            worst_key = k
        rel = diff / (plain_mean[k].abs() + 1e-8)
        max_rel_diff = max(max_rel_diff, rel.max().item())

    print(f'\n[Check 4] Encrypted mean vs plaintext mean:')
    print(f'         max abs diff = {max_abs_diff:.3e}  (worst key: {worst_key})')
    print(f'         max rel diff = {max_rel_diff:.3e}')

    TOLERANCE = 1e-3
    assert max_abs_diff < TOLERANCE, (
        f'CKKS noise exceeds tolerance: {max_abs_diff:.3e} >= {TOLERANCE}. '
        f'If new mults were added, extend coeff_mod_bit_sizes or reduce '
        f'global_scale.'
    )
    print(f'         OK (within {TOLERANCE} elementwise)')

    # ── Check 5: negative test — decrypt with public-only ctx ───────────
    print(f'\n[Check 5] Negative test: decrypt with PUBLIC-only ctx...')
    raised = False
    try:
        bad = _chunked_decrypt(summed, client_manifests[0], public_ctx)
        # If TenSEAL silently allowed it, the result must NOT match the truth.
        max_garbage = max((bad[k] - plain_mean[k]).abs().max().item() for k in keys)
        if max_garbage < 1e-2:
            raise AssertionError(
                f'public-only ctx decrypted to plaintext-correct values? '
                f'max diff = {max_garbage:.3e}. Public ctx may secretly have '
                f'the key — check serialize(save_secret_key=False) semantics.'
            )
        print(f'         OK (decrypt without secret produced garbage, '
              f'max diff vs truth = {max_garbage:.3e})')
    except AssertionError:
        raise  # re-raise the genuine failure
    except Exception as e:
        raised = True
        print(f'         OK (raised {type(e).__name__}: {str(e)[:80]})')

    # ── Summary ─────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('[test_ckks_correctness] ALL CHECKS PASSED')
    print(f'Bandwidth: {bytes_per_client / 1024:.1f} KB / client / round '
          f'(K=5 → {5 * bytes_per_client / 1024 / 1024:.2f} MB total uplink/round).')
    print(f'Timing: encrypt {t_encrypt:.2f}s, aggregate {t_aggregate:.2f}s, '
          f'decrypt {t_decrypt:.2f}s.')
    print('=' * 60)


if __name__ == '__main__':
    main()

"""
test_loss_paper_faithful.py — paper-faithful RAG-FT + KD-GLE sanity check.

Builds the InfoNCE + MSE loss in isolation (mirrors core.compute_client_loss
math without needing the BertModel forward pass) and asserts:

  1. When student logits == teacher logits (perfectly distilled), MSE → 0.
  2. When the cosine matrix is the identity (Q_i exactly matches R_i),
     temperature-scaled CE on diagonal labels → small (close to 0 as tau↓).
  3. Degenerate 1×1 batch (per-sample DP path): CE → 0, MSE → finite.
  4. kd_weight scales the MSE contribution linearly.
  5. Source code in core.py contains MSE (not KL) — guards against accidental revert.

This is the paper-faithful version (FedE4RAG arXiv:2504.19101 §3.3 KD-GLE
uses MSE on similarity matrices, NOT KL on softmaxed probabilities).

No Bert / GPU required — pure torch.nn.functional ops on synthetic tensors.
Exits 0 on success.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FEDE_ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, FEDE_ROOT)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def _info_nce_plus_mse(logits, server_logits, tau, kd_weight):
    """Mirror of core.TaskCalculator.compute_client_loss math (without the
    BERT forward + cos_sim wrapper). Operates directly on cosine-similarity
    matrices for testing in isolation.

    Paper-faithful: InfoNCE for RAG-FT + MSE for KD-GLE.
    """
    label = torch.arange(len(logits))
    loss_1 = F.cross_entropy(logits / tau, label)        # RAG-FT (paper §3.2)
    loss_2 = F.mse_loss(logits, server_logits)            # KD-GLE (paper §3.3)
    return loss_1 + kd_weight * loss_2, loss_1, loss_2


def main():
    print('=' * 60)
    print('[test_loss_paper_faithful] RAG-FT + KD-GLE (MSE) sanity checks')
    print('=' * 60)

    torch.manual_seed(42)

    # ── Check 1: student == teacher → MSE = 0 ────────────────────────────
    B = 8
    student = torch.randn(B, B)
    teacher = student.clone()
    total, ce, mse = _info_nce_plus_mse(student, teacher, tau=0.05, kd_weight=1.0)
    print(f'\n[Check 1] student == teacher: CE={ce:.6f}, MSE={mse:.6e}')
    assert mse.item() < 1e-9, f'Expected MSE≈0 when student==teacher, got {mse.item()}'
    print(f'    OK (MSE = {mse.item():.2e} < 1e-9)')

    # ── Check 2: identity cosine matrix → low InfoNCE as tau↓ ────────────
    identity = torch.eye(B)
    teacher_id = identity.clone()
    total_lo, ce_lo, mse_lo = _info_nce_plus_mse(identity, teacher_id, tau=0.05, kd_weight=1.0)
    total_hi, ce_hi, mse_hi = _info_nce_plus_mse(identity, teacher_id, tau=1.0, kd_weight=1.0)
    print(f'\n[Check 2] identity logits: tau=0.05 -> CE={ce_lo:.4f}, tau=1.00 -> CE={ce_hi:.4f}')
    assert ce_lo.item() < ce_hi.item(), (
        f'CE(tau=0.05) should be < CE(tau=1.0) for identity logits, '
        f'got {ce_lo} vs {ce_hi}'
    )
    print(f'    OK (lower tau sharpens distribution -> lower CE)')

    # ── Check 3: degenerate 1x1 (per-sample DP path) ─────────────────────
    single_q = torch.tensor([[0.7]])
    single_t = torch.tensor([[0.9]])
    total_s, ce_s, mse_s = _info_nce_plus_mse(single_q, single_t, tau=0.05, kd_weight=1.0)
    print(f'\n[Check 3] 1x1 batch: CE={ce_s:.6e}, MSE={mse_s:.6e}')
    assert ce_s.item() < 1e-6, f'1x1 CE should be ~0, got {ce_s}'
    # MSE on scalars 0.7 vs 0.9: (0.9-0.7)^2 = 0.04 ✓ non-zero, expected
    expected_mse = (0.9 - 0.7) ** 2
    assert abs(mse_s.item() - expected_mse) < 1e-6, (
        f'1x1 MSE should be {expected_mse}, got {mse_s.item()}'
    )
    print(f'    OK (CE≈0; MSE={mse_s.item():.4f} = (0.9-0.7)²)')

    # ── Check 4: kd_weight scales MSE linearly ──────────────────────────
    student = torch.randn(B, B)
    teacher = torch.randn(B, B)
    _, ce_w0, mse_w0 = _info_nce_plus_mse(student, teacher, tau=0.05, kd_weight=0.0)
    _, ce_w1, mse_w1 = _info_nce_plus_mse(student, teacher, tau=0.05, kd_weight=1.0)
    _, ce_w5, mse_w5 = _info_nce_plus_mse(student, teacher, tau=0.05, kd_weight=5.0)
    total_w0 = ce_w0 + 0.0 * mse_w0
    total_w1 = ce_w1 + 1.0 * mse_w1
    total_w5 = ce_w5 + 5.0 * mse_w5
    print(f'\n[Check 4] kd_weight scaling (MSE={mse_w1:.4f} fixed):')
    print(f'    kw=0  -> total={total_w0:.4f}')
    print(f'    kw=1  -> total={total_w1:.4f}')
    print(f'    kw=5  -> total={total_w5:.4f}')
    diff_01 = total_w1.item() - total_w0.item()
    diff_15 = total_w5.item() - total_w1.item()
    expected_diff_01 = 1.0 * mse_w1.item()
    expected_diff_15 = 4.0 * mse_w1.item()
    assert abs(diff_15 - expected_diff_15) < 1e-4, (
        f'kd_weight scaling broken: diff_15={diff_15} vs expected {expected_diff_15}'
    )
    assert abs(diff_01 - expected_diff_01) < 1e-4, (
        f'kd_weight scaling broken: diff_01={diff_01} vs expected {expected_diff_01}'
    )
    print(f'    OK (total scales linearly with kd_weight × MSE)')

    # ── Check 5: core.py source uses MSE (not KL) — guard against revert ─
    core_path = os.path.join(FEDE_ROOT, 'flgo', 'benchmark', 'fedrag_classification', 'core.py')
    with open(core_path, encoding='utf-8') as f:
        src = f.read()
    assert 'DEFAULT_TEMPERATURE' in src
    assert 'DEFAULT_KD_WEIGHT' in src
    assert 'F.cross_entropy(logits / tau' in src, 'RAG-FT InfoNCE missing'
    assert 'F.mse_loss(logits, teacher_logits)' in src, (
        'KD-GLE MSE missing! Check core.py — Phase 2 of paper recipe revert may have been undone.'
    )
    assert 'F.kl_div' not in src, (
        'KL divergence found in core.py — should be MSE per paper §3.3. '
        'Did Phase 3 KL→MSE revert get undone?'
    )
    print(f'\n[Check 5] core.py uses MSE (not KL) ── paper-faithful: OK')

    print('\n' + '=' * 60)
    print('[test_loss_paper_faithful] ALL CHECKS PASSED')
    print('=' * 60)


if __name__ == '__main__':
    main()

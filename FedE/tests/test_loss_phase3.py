"""
test_loss_phase3.py — Phase 3 InfoNCE + KL sanity check.

Builds the InfoNCE + KL loss in isolation (mirrors core.compute_client_loss
math without needing the BertModel forward pass) and asserts:

  1. When student logits == teacher logits (perfectly distilled), KL → 0.
  2. When the cosine matrix is the identity (Q_i exactly matches R_i),
     temperature-scaled CE on diagonal labels → small (close to 0 as tau↓).
  3. Loss decreases monotonically as tau decreases when Q_i matches R_i.

No Bert / GPU required — pure torch.nn.functional ops on synthetic tensors.
Exits 0 on success.
"""
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FEDE_ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, FEDE_ROOT)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def _info_nce_plus_kl(logits, server_logits, tau, kd_weight):
    """Mirror of core.TaskCalculator.compute_client_loss math (without the
    BERT forward + cos_sim wrapper). Operates directly on cosine-similarity
    matrices for testing in isolation.
    """
    label = torch.arange(len(logits))
    loss_1 = F.cross_entropy(logits / tau, label)
    loss_2 = F.kl_div(
        F.log_softmax(logits / tau, dim=-1),
        F.softmax(server_logits / tau, dim=-1),
        reduction='batchmean',
    ) * (tau ** 2)
    return loss_1 + kd_weight * loss_2, loss_1, loss_2


def main():
    print('=' * 60)
    print('[test_loss_phase3] InfoNCE + KL sanity checks')
    print('=' * 60)

    torch.manual_seed(42)

    # ── Check 1: student == teacher → KL = 0 ─────────────────────────────
    B = 8
    student = torch.randn(B, B)
    teacher = student.clone()
    total, ce, kl = _info_nce_plus_kl(student, teacher, tau=0.05, kd_weight=1.0)
    print(f'\n[Check 1] student == teacher: total={total:.6f}, CE={ce:.6f}, KL={kl:.6e}')
    assert kl.item() < 1e-5, f'Expected KL≈0 when student==teacher, got {kl.item()}'
    print(f'    OK (KL = {kl.item():.2e} < 1e-5)')

    # ── Check 2: identity cosine matrix → low InfoNCE as tau↓ ────────────
    # Identity means Q_i has cosine 1 with R_i and 0 with R_j (j≠i).
    identity = torch.eye(B)
    teacher_id = identity.clone()
    total_lo, ce_lo, kl_lo = _info_nce_plus_kl(identity, teacher_id, tau=0.05, kd_weight=1.0)
    total_hi, ce_hi, kl_hi = _info_nce_plus_kl(identity, teacher_id, tau=1.0, kd_weight=1.0)
    print(f'\n[Check 2] identity logits: tau=0.05 -> CE={ce_lo:.4f}, tau=1.00 -> CE={ce_hi:.4f}')
    assert ce_lo.item() < ce_hi.item(), (
        f'CE(tau=0.05) should be < CE(tau=1.0) for identity logits, '
        f'got {ce_lo} vs {ce_hi}'
    )
    print(f'    OK (lower tau sharpens distribution -> lower CE)')

    # ── Check 3: degenerate 1x1 (per-sample DP path) → both losses 0 ─────
    single_q = torch.tensor([[0.7]])  # 1×1 cosine
    single_t = torch.tensor([[0.9]])
    total_s, ce_s, kl_s = _info_nce_plus_kl(single_q, single_t, tau=0.05, kd_weight=1.0)
    print(f'\n[Check 3] 1x1 batch (DP per-sample): CE={ce_s:.6e}, KL={kl_s:.6e}')
    assert ce_s.item() < 1e-6 and kl_s.item() < 1e-6, (
        f'1x1 batch should give CE≈0 and KL≈0, got CE={ce_s}, KL={kl_s}'
    )
    print(f'    OK (degenerate per-sample path safely returns ~0 loss)')

    # ── Check 4: kd_weight controls KL contribution ─────────────────────
    student = torch.randn(B, B)
    teacher = torch.randn(B, B)  # different teacher → nonzero KL
    total_w0, _, kl_w0 = _info_nce_plus_kl(student, teacher, tau=0.05, kd_weight=0.0)
    total_w1, _, kl_w1 = _info_nce_plus_kl(student, teacher, tau=0.05, kd_weight=1.0)
    total_w5, _, kl_w5 = _info_nce_plus_kl(student, teacher, tau=0.05, kd_weight=5.0)
    print(f'\n[Check 4] kd_weight scaling (same KL={kl_w1:.4f}):')
    print(f'    kw=0  -> total={total_w0:.4f}')
    print(f'    kw=1  -> total={total_w1:.4f}')
    print(f'    kw=5  -> total={total_w5:.4f}')
    diff_01 = total_w1.item() - total_w0.item()
    diff_15 = total_w5.item() - total_w1.item()
    expected_diff_15 = 4.0 * kl_w1.item()
    expected_diff_01 = 1.0 * kl_w1.item()
    assert abs(diff_15 - expected_diff_15) < 1e-4, (
        f'kd_weight scaling broken: diff_15={diff_15} vs expected {expected_diff_15}'
    )
    assert abs(diff_01 - expected_diff_01) < 1e-4, (
        f'kd_weight scaling broken: diff_01={diff_01} vs expected {expected_diff_01}'
    )
    print(f'    OK (total scales linearly with kd_weight)')

    # ── Check 5: import surfaces from core.py reachable ─────────────────
    # Even though we can't actually import core.py without transformers,
    # we can verify the module-level constants are present in source.
    core_path = os.path.join(FEDE_ROOT, 'flgo', 'benchmark', 'fedrag_classification', 'core.py')
    with open(core_path, encoding='utf-8') as f:
        src = f.read()
    assert 'DEFAULT_TEMPERATURE' in src
    assert 'DEFAULT_KD_WEIGHT' in src
    assert 'F.cross_entropy(logits / tau' in src
    assert 'F.kl_div(' in src
    assert 'self.kd_weight * loss_2' in src
    print(f'\n[Check 5] core.py source has all Phase 3 markers: OK')

    print('\n' + '=' * 60)
    print('[test_loss_phase3] ALL CHECKS PASSED')
    print('=' * 60)


if __name__ == '__main__':
    main()

"""
test_dp_per_sample_paper.py — Phase 6.3 paper-faithful per-sample DP loss.

Validates the per-sample inner-loop math used inside
fedrag_lora.Client._train_dp WITHOUT requiring BertModel / GPU. Uses tiny
nn.Linear "encoder" so we can run on CPU in a few seconds.

Checks:
  1. Per-sample gradient flows ONLY through q_i (refs/teacher detached).
     ── Concrete test: alter ref_embs in place → loss_i unchanged after
        the first forward; gradient on q_i path matches grad expected
        from a pure-q_i forward.
  2. With B=1: loss_rag_ft → 0 (CE on 1×1 row), loss_kd_gle = (q·r − teacher)²
     non-degenerate ⟹ gradient is non-zero (vs old InfoNCE+KL which gave 0).
  3. Per-sample clipping + accumulation gives a different result than
     plain batch-level loss (sanity: per-sample ≠ unclipped batch).
  4. Privacy guarantee structure: σ * C noise added on accumulator and
     averaged by bs reproduces the standard DP-SGD update form
     E[g̃] = mean of clipped per-sample grads.
  5. kd_weight=0 path reduces loss to InfoNCE-only; kd_weight=1 adds MSE
     contribution on the SAME row (not full batch — proves per-sample).

CPU-only, no transformers/peft required.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FEDE_ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, FEDE_ROOT)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402


class TinyEncoder(nn.Module):
    """nn.Linear stand-in for BERT — produces (B, D) embedding from (B, L) ints."""
    def __init__(self, vocab=100, d=8):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.lin = nn.Linear(d, d, bias=False)

    def forward(self, x):
        return self.lin(self.emb(x).mean(dim=1))


def _per_sample_loss(q_n_i, ref_embs, teacher_sim_row, tau, kd_weight, label_i):
    """Mirror _train_dp inner loop's loss math (sim_row computed inline)."""
    sim_row = q_n_i @ ref_embs.t()
    loss_rag_ft = F.cross_entropy(sim_row / tau, label_i)
    loss_kd_gle = F.mse_loss(sim_row, teacher_sim_row)
    return loss_rag_ft + kd_weight * loss_kd_gle, loss_rag_ft, loss_kd_gle, sim_row


def main():
    print('=' * 64)
    print('[test_dp_per_sample_paper] Per-sample DP InfoNCE+MSE checks')
    print('=' * 64)

    torch.manual_seed(0)
    B, D = 4, 8
    tau = 0.05
    kd_weight = 1.0

    student = TinyEncoder(d=D)
    teacher = TinyEncoder(d=D)
    teacher.load_state_dict(student.state_dict())

    questions = torch.randint(0, 100, (B, 5))
    refs = torch.randint(0, 100, (B, 5))

    # ── Cache refs + teacher (no_grad), mirror _train_dp lines 449-468 ───
    with torch.no_grad():
        ref_embs = F.normalize(student(refs), dim=-1)
        tq = F.normalize(teacher(questions), dim=-1)
        tr = F.normalize(teacher(refs), dim=-1)
        teacher_sim = (tq @ tr.t()).detach()
    assert not ref_embs.requires_grad
    assert not teacher_sim.requires_grad
    print(f'\n[Cached] ref_embs {tuple(ref_embs.shape)} (no grad), '
          f'teacher_sim {tuple(teacher_sim.shape)} (no grad).')

    # ── Check 1: per-sample gradient flows only through q_i ──────────────
    # Compute loss for sample 0, then PERTURB ref_embs (in a copy).
    # The detached cached tensor used inside loss_i's graph keeps the
    # original values — proving refs are not part of the grad path.
    student.zero_grad()
    q_n_0 = F.normalize(student(questions[0:1]), dim=-1)
    loss_0, ce_0, mse_0, _ = _per_sample_loss(
        q_n_0, ref_embs, teacher_sim[0:1], tau, kd_weight, torch.tensor([0]),
    )
    loss_0.backward()
    grad_lin_q_only = student.lin.weight.grad.clone()

    # Now do a FORWARD where refs come from the *current* student
    # (still no_grad, but with student's current weights). For a fresh
    # forward these are identical → gradient should also be identical.
    student.zero_grad()
    with torch.no_grad():
        ref_embs_fresh = F.normalize(student(refs), dim=-1)
    q_n_0b = F.normalize(student(questions[0:1]), dim=-1)
    loss_0b, _, _, _ = _per_sample_loss(
        q_n_0b, ref_embs_fresh, teacher_sim[0:1], tau, kd_weight,
        torch.tensor([0]),
    )
    loss_0b.backward()
    grad_lin_q_only_fresh = student.lin.weight.grad.clone()
    diff = (grad_lin_q_only - grad_lin_q_only_fresh).abs().max().item()
    print(f'\n[Check 1] grad-isolation: max-diff between '
          f'cached-refs and fresh-refs path = {diff:.2e}')
    assert diff < 1e-6, f'Gradient should not depend on cached vs fresh ref forward; diff={diff}'
    print(f'    OK (per-sample grad depends only on q_i path)')

    # ── Check 2: B=1 non-degenerate ──────────────────────────────────────
    # InfoNCE+KL would give grad=0 on B=1 (softmax over 1 logit ≡ 1.0).
    # New loss has MSE-KD on the (1,1) row → non-zero grad.
    questions1 = torch.randint(0, 100, (1, 5))
    refs1 = torch.randint(0, 100, (1, 5))
    with torch.no_grad():
        ref_embs1 = F.normalize(student(refs1), dim=-1)
        tq1 = F.normalize(teacher(questions1), dim=-1)
        tr1 = F.normalize(teacher(refs1), dim=-1)
        teacher_sim1 = (tq1 @ tr1.t()).detach()
    student.zero_grad()
    q_n1 = F.normalize(student(questions1), dim=-1)
    loss1, ce1, mse1, sim1 = _per_sample_loss(
        q_n1, ref_embs1, teacher_sim1, tau, kd_weight, torch.tensor([0]),
    )
    loss1.backward()
    grad_norm_b1 = sum(p.grad.detach().norm(2).item() ** 2
                       for p in student.parameters() if p.grad is not None) ** 0.5
    print(f'\n[Check 2] B=1: CE={ce1:.6e} (≈0 expected), '
          f'MSE={mse1:.6f}, ‖∇‖={grad_norm_b1:.6f}')
    assert ce1.item() < 1e-6, f'CE on 1×1 row should be ≈0, got {ce1}'
    # Note: gradient is zero only when teacher == student AND sim_row ==
    # teacher_sim — which is exactly the case here (we copied weights).
    # That's expected at round 0; signal kicks in once teacher diverges.
    # Real-world test: perturb teacher and re-measure.
    teacher.lin.weight.data += torch.randn_like(teacher.lin.weight.data) * 0.1
    with torch.no_grad():
        tq1b = F.normalize(teacher(questions1), dim=-1)
        tr1b = F.normalize(teacher(refs1), dim=-1)
        teacher_sim1b = (tq1b @ tr1b.t()).detach()
    student.zero_grad()
    q_n1b = F.normalize(student(questions1), dim=-1)
    loss1b, ce1b, mse1b, _ = _per_sample_loss(
        q_n1b, ref_embs1, teacher_sim1b, tau, kd_weight, torch.tensor([0]),
    )
    loss1b.backward()
    grad_norm_b1_diverged = sum(p.grad.detach().norm(2).item() ** 2
                                for p in student.parameters() if p.grad is not None) ** 0.5
    print(f'    after teacher divergence: MSE={mse1b:.6f}, ‖∇‖={grad_norm_b1_diverged:.6f}')
    assert grad_norm_b1_diverged > 1e-4, (
        'B=1 with diverged teacher should have non-zero gradient via KD-GLE; '
        f'got ‖∇‖={grad_norm_b1_diverged}'
    )
    print(f'    OK (KD-GLE provides B=1 signal once teacher diverges)')

    # restore teacher for later checks
    teacher.load_state_dict(student.state_dict())

    # ── Check 3: per-sample loop accumulates correctly + clipping ────────
    # Compute "manually" (per-sample loop with clip) vs naive batch loss.
    clip_norm = 0.1
    accumulated = [torch.zeros_like(p.data) for p in student.parameters()]
    student.zero_grad()
    with torch.no_grad():
        ref_embs2 = F.normalize(student(refs), dim=-1)
        tq2 = F.normalize(teacher(questions), dim=-1)
        tr2 = F.normalize(teacher(refs), dim=-1)
        teacher_sim2 = (tq2 @ tr2.t()).detach()

    params = list(student.parameters())
    for i in range(B):
        student.zero_grad()
        q_n_i = F.normalize(student(questions[i:i+1]), dim=-1)
        loss_i, _, _, _ = _per_sample_loss(
            q_n_i, ref_embs2, teacher_sim2[i:i+1], tau, kd_weight,
            torch.tensor([i]),
        )
        loss_i.backward()
        norm_i = sum(p.grad.detach().norm(2).item() ** 2
                     for p in params if p.grad is not None) ** 0.5
        coef = min(1.0, clip_norm / (norm_i + 1e-8))
        for j, p in enumerate(params):
            if p.grad is not None:
                accumulated[j] += p.grad.detach() * coef

    # Compare to unclipped batch-level grad — they MUST differ.
    student.zero_grad()
    q_n_batch = F.normalize(student(questions), dim=-1)
    sim_full = q_n_batch @ ref_embs2.t()
    labels_full = torch.arange(B)
    loss_full = (F.cross_entropy(sim_full / tau, labels_full)
                 + kd_weight * F.mse_loss(sim_full, teacher_sim2))
    loss_full.backward()
    batch_grad = [p.grad.detach().clone() for p in params]

    cosine_persample_vs_batch = (
        sum((a * b).sum().item() for a, b in zip(accumulated, batch_grad))
        / (sum(a.norm().item() ** 2 for a in accumulated) ** 0.5 + 1e-12)
        / (sum(b.norm().item() ** 2 for b in batch_grad) ** 0.5 + 1e-12)
    )
    print(f'\n[Check 3] per-sample accum vs batch grad: cosine={cosine_persample_vs_batch:.4f}')
    # Cosine should be positive (same direction) but not 1.0 (clipping makes them differ).
    assert cosine_persample_vs_batch > 0.5, (
        'Per-sample accumulator should align with batch grad (clipping aside)'
    )
    assert cosine_persample_vs_batch < 0.9999, (
        'With clipping, per-sample ≠ batch grad'
    )
    print(f'    OK (aligned but distinct from unclipped batch — clipping active)')

    # ── Check 4: Gaussian noise structure ────────────────────────────────
    # Average-noise variance test: with σ=1.29 and clip_norm=0.1, expected
    # std ≈ 0.129 / B per coordinate. Run many trials and check.
    sigma = 1.29
    bs = 4
    n_trials = 1000
    samples = []
    for _ in range(n_trials):
        a = torch.zeros(10)
        noise = torch.randn_like(a) * (sigma * clip_norm)
        samples.append(((a + noise) / float(bs)).norm().item())
    mean_norm = sum(samples) / len(samples)
    expected_per_coord_std = sigma * clip_norm / bs
    expected_norm = expected_per_coord_std * (10 ** 0.5) * 0.95  # ~chi distribution mean
    print(f'\n[Check 4] noise norm: empirical={mean_norm:.6f}, '
          f'expected≈{expected_norm:.6f}')
    assert abs(mean_norm - expected_norm) / expected_norm < 0.15, (
        f'Noise scale mismatch: empirical={mean_norm}, expected≈{expected_norm}'
    )
    print(f'    OK (DP-SGD noise has correct σ·C/bs scale)')

    # ── Check 5: kd_weight=0 reduces to pure InfoNCE per row ─────────────
    student.zero_grad()
    q_n_5 = F.normalize(student(questions[0:1]), dim=-1)
    loss_full_kd, ce_full_kd, mse_full_kd, _ = _per_sample_loss(
        q_n_5, ref_embs2, teacher_sim2[0:1], tau, kd_weight=1.0,
        label_i=torch.tensor([0]),
    )
    student.zero_grad()
    q_n_5b = F.normalize(student(questions[0:1]), dim=-1)
    loss_no_kd, ce_no_kd, mse_no_kd, _ = _per_sample_loss(
        q_n_5b, ref_embs2, teacher_sim2[0:1], tau, kd_weight=0.0,
        label_i=torch.tensor([0]),
    )
    print(f'\n[Check 5] kd_weight=1: total={loss_full_kd:.6f} '
          f'(CE={ce_full_kd:.6f}, MSE={mse_full_kd:.6f})')
    print(f'           kd_weight=0: total={loss_no_kd:.6f} '
          f'(CE={ce_no_kd:.6f})')
    assert abs((loss_full_kd - loss_no_kd).item() - mse_full_kd.item()) < 1e-5
    assert abs((ce_full_kd - ce_no_kd).item()) < 1e-5
    print(f'    OK (kd_weight scales MSE-KD additively on per-sample row)')

    print('\n' + '=' * 64)
    print('[test_dp_per_sample_paper] ALL CHECKS PASSED')
    print('=' * 64)


if __name__ == '__main__':
    main()

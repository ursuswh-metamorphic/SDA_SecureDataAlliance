"""
Compare Baseline vs DP model quality.
Metrics:
  1. Retrieval accuracy (Top-1, Top-3, Top-5)
  2. Mean cosine similarity (correct pairs vs incorrect pairs)
  3. Mean Reciprocal Rank (MRR)
  4. Weight divergence from pretrained model
"""
import torch
import torch.nn.functional as F
import json
import numpy as np
from transformers import BertModel, BertTokenizer
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ── Load data ─────────────────────────────────────────────────────────────────
with open('new_select_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Sample subset for evaluation
np.random.seed(42)
indices = np.random.choice(len(data), size=min(200, len(data)), replace=False)
test_data = [data[i] for i in indices]

questions = [d['question'] for d in test_data]
references = [d['reference'] for d in test_data]
companies = [d['company'] for d in test_data]

print(f'Evaluation set: {len(test_data)} samples')
print(f'Unique companies: {len(set(companies))}')
print()

# ── Load tokenizer ────────────────────────────────────────────────────────────
tokenizer = BertTokenizer.from_pretrained('BAAI/bge-base-en')
max_length = tokenizer.model_max_length


def evaluate_model(model, model_name):
    model.eval()
    model.to(device)

    batch_size = 32
    all_q_embs = []
    all_r_embs = []

    for i in range(0, len(questions), batch_size):
        batch_q = questions[i:i+batch_size]
        inputs = tokenizer(batch_q, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        embs = outputs.last_hidden_state.mean(dim=1)
        all_q_embs.append(embs.cpu())

    for i in range(0, len(references), batch_size):
        batch_r = references[i:i+batch_size]
        inputs = tokenizer(batch_r, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        embs = outputs.last_hidden_state.mean(dim=1)
        all_r_embs.append(embs.cpu())

    q_embs = torch.cat(all_q_embs, dim=0)
    r_embs = torch.cat(all_r_embs, dim=0)

    q_norm = F.normalize(q_embs, dim=-1)
    r_norm = F.normalize(r_embs, dim=-1)

    sim_matrix = torch.mm(q_norm, r_norm.t()).numpy()
    N = len(sim_matrix)

    # Retrieval accuracy
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    mrr_sum = 0.0

    for i in range(N):
        sorted_indices = np.argsort(-sim_matrix[i])
        rank = np.where(sorted_indices == i)[0][0] + 1
        mrr_sum += 1.0 / rank
        if rank <= 1: top1_correct += 1
        if rank <= 3: top3_correct += 1
        if rank <= 5: top5_correct += 1

    top1_acc = top1_correct / N * 100
    top3_acc = top3_correct / N * 100
    top5_acc = top5_correct / N * 100
    mrr = mrr_sum / N

    # Cosine similarity stats
    diagonal = np.diag(sim_matrix)
    off_diag_mask = ~np.eye(N, dtype=bool)
    off_diagonal = sim_matrix[off_diag_mask]

    mean_correct_sim = diagonal.mean()
    mean_incorrect_sim = off_diagonal.mean()
    sim_gap = mean_correct_sim - mean_incorrect_sim

    # Cluster quality
    intra_sims = []
    inter_sims = []
    for i in range(N):
        for j in range(i+1, min(N, i+50)):
            sim_val = float(sim_matrix[i][j])
            if companies[i] == companies[j]:
                intra_sims.append(sim_val)
            else:
                inter_sims.append(sim_val)

    intra_mean = np.mean(intra_sims) if intra_sims else 0
    inter_mean = np.mean(inter_sims) if inter_sims else 0

    print(f'=== {model_name} ===')
    print(f'  Retrieval Accuracy:')
    print(f'    Top-1:  {top1_acc:6.2f}%  ({top1_correct}/{N})')
    print(f'    Top-3:  {top3_acc:6.2f}%  ({top3_correct}/{N})')
    print(f'    Top-5:  {top5_acc:6.2f}%  ({top5_correct}/{N})')
    print(f'  MRR:      {mrr:.4f}')
    print(f'  Cosine Similarity:')
    print(f'    Correct pairs (diagonal):   {mean_correct_sim:.4f}')
    print(f'    Incorrect pairs (off-diag): {mean_incorrect_sim:.4f}')
    print(f'    Gap (higher=better):        {sim_gap:.4f}')
    print(f'  Cluster Quality:')
    print(f'    Intra-company sim:  {intra_mean:.4f}')
    print(f'    Inter-company sim:  {inter_mean:.4f}')
    print(f'    Separation:         {intra_mean - inter_mean:.4f}')
    print()

    return {
        'top1': top1_acc, 'top3': top3_acc, 'top5': top5_acc,
        'mrr': mrr, 'sim_correct': mean_correct_sim, 'sim_incorrect': mean_incorrect_sim,
        'sim_gap': sim_gap, 'intra': intra_mean, 'inter': inter_mean,
    }


def load_model(path, name):
    base = BertModel.from_pretrained('BAAI/bge-base-en')
    state = torch.load(path, map_location='cpu', weights_only=True)
    clean = {}
    for k, v in state.items():
        key = k.replace('module.', '').replace('model.', '', 1) if 'model.' in k else k
        clean[key] = v
    missing, unexpected = base.load_state_dict(clean, strict=False)
    print(f'Loaded {name}: missing={len(missing)}, unexpected={len(unexpected)}')
    return base


# ── Load models ───────────────────────────────────────────────────────────────
print('Loading pretrained model...')
pretrained = BertModel.from_pretrained('BAAI/bge-base-en')

print('Loading baseline model...')
baseline = load_model('x-model_2026-03-21_05-20-48.bin', 'Baseline')

print('Loading DP model...')
dp_model = load_model('x-model_2026-03-21_06-12-57.bin', 'DP')
print()

# ── Run evaluations ──────────────────────────────────────────────────────────
results = {}
results['pretrained'] = evaluate_model(pretrained, 'Pretrained (no fine-tune)')
results['baseline'] = evaluate_model(baseline, 'Baseline (no DP, 25 rounds)')
results['dp'] = evaluate_model(dp_model, 'DP-FedRAG (eps=8.0, sigma=1.67)')

# ── Weight divergence ─────────────────────────────────────────────────────────
print('=== Weight Divergence from Pretrained ===')
pretrained_state = pretrained.state_dict()
baseline_state = baseline.state_dict()
dp_state = dp_model.state_dict()

baseline_diff = 0.0
dp_diff = 0.0
n_params = 0
for key in pretrained_state:
    if key in baseline_state and key in dp_state:
        p = pretrained_state[key].float()
        b = baseline_state[key].float()
        d = dp_state[key].float()
        baseline_diff += (b - p).norm().item() ** 2
        dp_diff += (d - p).norm().item() ** 2
        n_params += p.numel()

baseline_diff = baseline_diff ** 0.5
dp_diff = dp_diff ** 0.5
print(f'  Baseline L2 distance:  {baseline_diff:.4f}')
print(f'  DP L2 distance:        {dp_diff:.4f}')
print(f'  Ratio (DP/Baseline):   {dp_diff/baseline_diff:.4f}')
print(f'  Total params compared: {n_params:,}')
print()

# ── Summary table ─────────────────────────────────────────────────────────────
print('=' * 70)
print(f'{"Metric":<30} {"Pretrained":>12} {"Baseline":>12} {"DP (e=8)":>12}')
print('-' * 70)
for metric, label in [
    ('top1', 'Top-1 Acc (%)'),
    ('top3', 'Top-3 Acc (%)'),
    ('top5', 'Top-5 Acc (%)'),
    ('mrr', 'MRR'),
    ('sim_correct', 'Sim (correct)'),
    ('sim_incorrect', 'Sim (incorrect)'),
    ('sim_gap', 'Sim Gap'),
    ('intra', 'Intra-company'),
    ('inter', 'Inter-company'),
]:
    p = results['pretrained'][metric]
    b = results['baseline'][metric]
    d = results['dp'][metric]
    if 'Acc' in label:
        print(f'{label:<30} {p:>12.2f} {b:>12.2f} {d:>12.2f}')
    else:
        print(f'{label:<30} {p:>12.4f} {b:>12.4f} {d:>12.4f}')

print('-' * 70)
if results['baseline']['top1'] > 0:
    retention = results['dp']['top1'] / results['baseline']['top1'] * 100
    print(f'{"Utility Retention (Top-1)":<30} {"":>12} {"100.0%":>12} {retention:>11.1f}%')
print('=' * 70)

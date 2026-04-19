"""
Compare Baseline vs DP model quality.
Metrics:
  1. Retrieval accuracy (Top-1, Top-3, Top-5, Top-10)
  2. Precision, Recall, F1 at Top-k (k=1,3,5,10)
  3. Mean cosine similarity (correct pairs vs incorrect pairs)
  4. Mean Reciprocal Rank (MRR)
  5. NDCG@k (Normalized Discounted Cumulative Gain)
  6. Weight divergence from pretrained model
"""
import torch
import torch.nn.functional as F
import json
import numpy as np
from transformers import AutoModel, AutoTokenizer

# Upstream embedding backbone: MedCPT Article Encoder
# https://huggingface.co/ncbi/MedCPT-Article-Encoder
EMBEDDING_MODEL_NAME = "ncbi/MedCPT-Article-Encoder"
from collections import defaultdict
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ── Load data ─────────────────────────────────────────────────────────────────
with open('pubmed_train.json', 'r', encoding='utf-8') as f:
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
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
# MedCPT max sequence length (tokenizer.model_max_length may return a huge default)
max_length = min(tokenizer.model_max_length, 512)


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
        # MedCPT uses [CLS] pooling (see HF model card)
        embs = outputs.last_hidden_state[:, 0, :]
        all_q_embs.append(embs.cpu())

    for i in range(0, len(references), batch_size):
        batch_r = references[i:i+batch_size]
        inputs = tokenizer(batch_r, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        embs = outputs.last_hidden_state[:, 0, :]
        all_r_embs.append(embs.cpu())

    q_embs = torch.cat(all_q_embs, dim=0)
    r_embs = torch.cat(all_r_embs, dim=0)

    q_norm = F.normalize(q_embs, dim=-1)
    r_norm = F.normalize(r_embs, dim=-1)

    sim_matrix = torch.mm(q_norm, r_norm.t()).numpy()
    N = len(sim_matrix)

    # ── Build ground-truth relevance ──────────────────────────────────────────
    # For each query i, relevant docs = {i} (exact match) + same-company docs
    company_to_indices = defaultdict(set)
    for idx, c in enumerate(companies):
        company_to_indices[c].add(idx)

    ks = [1, 3, 5, 10]
    topk_correct = {k: 0 for k in ks}  # Hit@k (at least 1 relevant in top-k)
    precision_at_k = {k: [] for k in ks}
    recall_at_k = {k: [] for k in ks}
    f1_at_k = {k: [] for k in ks}
    ndcg_at_k = {k: [] for k in ks}
    mrr_sum = 0.0

    for i in range(N):
        sorted_indices = np.argsort(-sim_matrix[i])

        # Relevant set: exact match (i) + same company
        relevant = company_to_indices[companies[i]]

        # MRR: rank of exact match
        rank = np.where(sorted_indices == i)[0][0] + 1
        mrr_sum += 1.0 / rank

        for k in ks:
            top_k_set = set(sorted_indices[:k].tolist())

            # Hit@k: is the exact match in top-k?
            if i in top_k_set:
                topk_correct[k] += 1

            # Precision@k: fraction of top-k that are relevant
            relevant_in_topk = len(top_k_set & relevant)
            p = relevant_in_topk / k
            precision_at_k[k].append(p)

            # Recall@k: fraction of relevant docs found in top-k
            r = relevant_in_topk / len(relevant) if len(relevant) > 0 else 0
            recall_at_k[k].append(r)

            # F1@k
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            f1_at_k[k].append(f1)

            # NDCG@k
            dcg = 0.0
            for rank_pos, doc_idx in enumerate(sorted_indices[:k]):
                if doc_idx in relevant:
                    dcg += 1.0 / np.log2(rank_pos + 2)
            # Ideal DCG
            ideal_relevant = min(k, len(relevant))
            idcg = sum(1.0 / np.log2(r + 2) for r in range(ideal_relevant))
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcg_at_k[k].append(ndcg)

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
    print(f'  Retrieval Accuracy (Hit@k — exact match in top-k):')
    for k in ks:
        acc = topk_correct[k] / N * 100
        print(f'    Hit@{k:<2d}:  {acc:6.2f}%  ({topk_correct[k]}/{N})')
    print(f'  Precision / Recall / F1 @k (company-level relevance):')
    for k in ks:
        p = np.mean(precision_at_k[k]) * 100
        r = np.mean(recall_at_k[k]) * 100
        f = np.mean(f1_at_k[k]) * 100
        print(f'    @{k:<2d}:  P={p:5.2f}%  R={r:5.2f}%  F1={f:5.2f}%')
    print(f'  NDCG@k:')
    for k in ks:
        n = np.mean(ndcg_at_k[k])
        print(f'    NDCG@{k:<2d}: {n:.4f}')
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
        'top1': topk_correct[1]/N*100, 'top3': topk_correct[3]/N*100,
        'top5': topk_correct[5]/N*100, 'top10': topk_correct[10]/N*100,
        'p1': np.mean(precision_at_k[1])*100, 'p3': np.mean(precision_at_k[3])*100,
        'p5': np.mean(precision_at_k[5])*100, 'p10': np.mean(precision_at_k[10])*100,
        'r1': np.mean(recall_at_k[1])*100, 'r3': np.mean(recall_at_k[3])*100,
        'r5': np.mean(recall_at_k[5])*100, 'r10': np.mean(recall_at_k[10])*100,
        'f1_1': np.mean(f1_at_k[1])*100, 'f1_3': np.mean(f1_at_k[3])*100,
        'f1_5': np.mean(f1_at_k[5])*100, 'f1_10': np.mean(f1_at_k[10])*100,
        'ndcg1': np.mean(ndcg_at_k[1]), 'ndcg3': np.mean(ndcg_at_k[3]),
        'ndcg5': np.mean(ndcg_at_k[5]), 'ndcg10': np.mean(ndcg_at_k[10]),
        'mrr': mrr, 'sim_correct': mean_correct_sim, 'sim_incorrect': mean_incorrect_sim,
        'sim_gap': sim_gap, 'intra': intra_mean, 'inter': inter_mean,
    }


def load_model(path, name):
    base = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)
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
pretrained = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)

print('Loading baseline model...')
baseline = load_model('x-model_2026-03-29_04-39-38.bin', 'Baseline')

print('Loading DP model...')
dp_model = load_model('x-model_2026-03-29_05-52-40.bin', 'DP')
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
print('=' * 78)
print(f'{"Metric":<30} {"Pretrained":>14} {"Baseline":>14} {"DP (e=8)":>14}')
print('-' * 78)
for metric, label in [
    ('top1', 'Hit@1 (%)'),
    ('top3', 'Hit@3 (%)'),
    ('top5', 'Hit@5 (%)'),
    ('top10', 'Hit@10 (%)'),
    ('f1_1', 'F1@1 (%)'),
    ('f1_3', 'F1@3 (%)'),
    ('f1_5', 'F1@5 (%)'),
    ('f1_10', 'F1@10 (%)'),
    ('p1', 'Precision@1 (%)'),
    ('p3', 'Precision@3 (%)'),
    ('p5', 'Precision@5 (%)'),
    ('p10', 'Precision@10 (%)'),
    ('r1', 'Recall@1 (%)'),
    ('r3', 'Recall@3 (%)'),
    ('r5', 'Recall@5 (%)'),
    ('r10', 'Recall@10 (%)'),
    ('ndcg1', 'NDCG@1'),
    ('ndcg3', 'NDCG@3'),
    ('ndcg5', 'NDCG@5'),
    ('ndcg10', 'NDCG@10'),
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
    if '%' in label:
        print(f'{label:<30} {p:>14.2f} {b:>14.2f} {d:>14.2f}')
    else:
        print(f'{label:<30} {p:>14.4f} {b:>14.4f} {d:>14.4f}')

print('-' * 78)
# Utility retention
print(f'\n{"UTILITY RETENTION (DP vs Baseline):":}')
for metric, label in [('top1', 'Hit@1'), ('f1_3', 'F1@3'), ('f1_5', 'F1@5'), ('mrr', 'MRR'), ('ndcg5', 'NDCG@5')]:
    b = results['baseline'][metric]
    d = results['dp'][metric]
    if b > 0:
        retention = d / b * 100
        drop = b - d
        print(f'  {label:<12} Baseline={b:>8.2f}  DP={d:>8.2f}  Retention={retention:>6.1f}%  Drop={drop:>+7.2f}')

# Privacy-utility summary
print(f'\n{"="*78}')
print(f'CONCLUSION:')
print(f'  Baseline: NO privacy guarantee (eps ~ infinity)')
print(f'  DP model: (eps={8.0}, delta={1e-5})-DP formal guarantee')
b_f1 = results['baseline']['f1_5']
d_f1 = results['dp']['f1_5']
if b_f1 > 0:
    ret = d_f1 / b_f1 * 100
    print(f'  F1@5 utility retention: {ret:.1f}%')
    print(f'  Privacy-utility trade-off: {8.0:.1f}-DP for {100-ret:.1f}% F1 drop')
print('=' * 78)

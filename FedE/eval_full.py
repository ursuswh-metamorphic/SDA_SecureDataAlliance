import torch, torch.nn.functional as F, json, numpy as np, sys, io
from transformers import BertModel, BertTokenizer
from collections import defaultdict
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

with open('pubmed_train.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
np.random.seed(42)
indices = np.random.choice(len(data), size=min(200, len(data)), replace=False)
test_data = [data[i] for i in indices]
questions = [d['question'] for d in test_data]
references = [d['reference'] for d in test_data]
companies = [d['company'] for d in test_data]
print(f'Eval set: {len(test_data)} samples, {len(set(companies))} companies')

tokenizer = BertTokenizer.from_pretrained('BAAI/bge-base-en')
max_length = tokenizer.model_max_length

def evaluate_model(model, model_name):
    model.eval(); model.to(device)
    all_q, all_r = [], []
    for i in range(0, len(questions), 32):
        inp = tokenizer(questions[i:i+32], return_tensors='pt', padding=True, truncation=True, max_length=max_length)
        inp = {k: v.to(device) for k, v in inp.items()}
        with torch.no_grad(): all_q.append(model(**inp).last_hidden_state.mean(dim=1).cpu())
    for i in range(0, len(references), 32):
        inp = tokenizer(references[i:i+32], return_tensors='pt', padding=True, truncation=True, max_length=max_length)
        inp = {k: v.to(device) for k, v in inp.items()}
        with torch.no_grad(): all_r.append(model(**inp).last_hidden_state.mean(dim=1).cpu())
    sim = torch.mm(F.normalize(torch.cat(all_q), dim=-1), F.normalize(torch.cat(all_r), dim=-1).t()).numpy()
    N = len(sim)
    co = defaultdict(set)
    for idx, c in enumerate(companies): co[c].add(idx)
    ks = [1, 3, 5, 10]
    hit = {k: 0 for k in ks}; f1k = {k: [] for k in ks}; ndcgk = {k: [] for k in ks}; mrr = 0.0
    for i in range(N):
        si = np.argsort(-sim[i]); rel = co[companies[i]]
        mrr += 1.0 / (np.where(si == i)[0][0] + 1)
        for k in ks:
            tk = set(si[:k].tolist())
            if i in tk: hit[k] += 1
            ri = len(tk & rel); p = ri/k; r = ri/len(rel) if rel else 0
            f1k[k].append(2*p*r/(p+r) if (p+r) > 0 else 0)
            dcg = sum(1/np.log2(rp+2) for rp, di in enumerate(si[:k]) if di in rel)
            idcg = sum(1/np.log2(rp+2) for rp in range(min(k, len(rel))))
            ndcgk[k].append(dcg/idcg if idcg > 0 else 0)
    mrr /= N; diag = np.diag(sim); gap = diag.mean() - sim[~np.eye(N, dtype=bool)].mean()
    res = {'mrr': mrr, 'gap': gap, 'sim_correct': float(diag.mean()), 'sim_incorrect': float(sim[~np.eye(N, dtype=bool)].mean())}
    for k in ks:
        res[f'hit{k}'] = hit[k]/N*100; res[f'f1_{k}'] = np.mean(f1k[k])*100; res[f'ndcg{k}'] = np.mean(ndcgk[k])
    print(f'  {model_name}: Hit@1={res["hit1"]:.1f}% Hit@5={res["hit5"]:.1f}% MRR={mrr:.4f} NDCG@5={res["ndcg5"]:.4f} Gap={gap:.4f}')
    return res

def load_model(path, name):
    base = BertModel.from_pretrained('BAAI/bge-base-en')
    st = torch.load(path, map_location='cpu', weights_only=True)
    cl = {(k.replace('module.', '').replace('model.', '', 1) if 'model.' in k else k): v for k, v in st.items()}
    base.load_state_dict(cl, strict=False)
    return base

print('Loading models...')
pre = BertModel.from_pretrained('BAAI/bge-base-en')
base = load_model('x-model_2026-04-10_05-07-12.bin', 'Baseline')
dp20 = load_model('x-model_2026-04-10_03-58-24.bin', 'DP-eps20')
dp8 = load_model('x-model_2026-04-10_07-44-18.bin', 'DP-eps8')
print()

r = {}
r['pre'] = evaluate_model(pre, 'Pretrained')
r['base'] = evaluate_model(base, 'Baseline (no DP)')
r['dp20'] = evaluate_model(dp20, 'DP eps=20')
r['dp8'] = evaluate_model(dp8, 'DP eps=8')
print()

header = '{:<20} {:>12} {:>12} {:>12} {:>12}'.format('Metric', 'Pretrained', 'Baseline', 'DP_eps20', 'DP_eps8')
print('=' * 82)
print(header)
print('-' * 82)
for key, label in [('hit1','Hit@1 (%)'),('hit3','Hit@3 (%)'),('hit5','Hit@5 (%)'),('hit10','Hit@10 (%)'),
    ('f1_1','F1@1 (%)'),('f1_3','F1@3 (%)'),('f1_5','F1@5 (%)'),('f1_10','F1@10 (%)'),
    ('ndcg5','NDCG@5'),('ndcg10','NDCG@10'),('mrr','MRR'),
    ('sim_correct','Sim correct'),('sim_incorrect','Sim incorrect'),('gap','Sim Gap')]:
    p, b, d20, d8 = r['pre'][key], r['base'][key], r['dp20'][key], r['dp8'][key]
    if '%' in label:
        print('{:<20} {:>12.2f} {:>12.2f} {:>12.2f} {:>12.2f}'.format(label, p, b, d20, d8))
    else:
        print('{:<20} {:>12.4f} {:>12.4f} {:>12.4f} {:>12.4f}'.format(label, p, b, d20, d8))
print('=' * 82)
print()
print('Retention vs Baseline:')
for key, label in [('hit1','Hit@1'),('hit5','Hit@5'),('mrr','MRR'),('ndcg5','NDCG@5'),('gap','Sim Gap')]:
    b = r['base'][key]
    d20 = r['dp20'][key]
    d8 = r['dp8'][key]
    if b > 0:
        print('  {:<12} Baseline={:>8.2f}  eps20={:>8.2f} ({:.1f}%)  eps8={:>8.2f} ({:.1f}%)'.format(
            label, b, d20, d20/b*100, d8, d8/b*100))
print()
print('Training time: Baseline=766s (12.8min), DP eps=20=5378s (89.6min), DP eps=8=5369s (89.5min)')
print('Privacy: eps=20 sigma=0.8118, eps=8 sigma=1.6701')

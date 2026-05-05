"""Evaluate Phase 2 LoRA-only DP checkpoint vs pretrained BGE-base.

Uses 200 random samples from selected_data.json (financial RAG corpus).
Reports retrieval metrics: Hit@1/3/5/10, F1@k, MRR, NDCG@k.
"""
import torch, torch.nn.functional as F, json, numpy as np
from collections import defaultdict
from transformers import BertModel, BertTokenizer
import sys, os
sys.path.insert(0, '/root/sda/FedE')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

with open('selected_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
np.random.seed(42)
indices = np.random.choice(len(data), size=min(200, len(data)), replace=False)
test = [data[i] for i in indices]
questions = [d['question'] for d in test]
references = [d['reference'] for d in test]
companies = [d['company'] for d in test]
print(f'Eval set: {len(test)} samples, {len(set(companies))} unique companies')

tokenizer = BertTokenizer.from_pretrained('BAAI/bge-base-en')
max_length = tokenizer.model_max_length

def evaluate(model, name):
    model.eval().to(device)
    qe, re = [], []
    bs = 32
    for i in range(0, len(questions), bs):
        inp = tokenizer(questions[i:i+bs], return_tensors='pt', padding=True, truncation=True, max_length=max_length)
        inp = {k: v.to(device) for k, v in inp.items()}
        with torch.no_grad():
            qe.append(model(**inp).last_hidden_state.mean(dim=1).cpu())
    for i in range(0, len(references), bs):
        inp = tokenizer(references[i:i+bs], return_tensors='pt', padding=True, truncation=True, max_length=max_length)
        inp = {k: v.to(device) for k, v in inp.items()}
        with torch.no_grad():
            re.append(model(**inp).last_hidden_state.mean(dim=1).cpu())
    Q = F.normalize(torch.cat(qe), dim=-1)
    R = F.normalize(torch.cat(re), dim=-1)
    sim = (Q @ R.t()).numpy()
    N = len(sim)
    c2i = defaultdict(set)
    for i, c in enumerate(companies): c2i[c].add(i)
    ks = [1,3,5,10]
    hit = {k:0 for k in ks}; p = {k:[] for k in ks}; r = {k:[] for k in ks}; f = {k:[] for k in ks}; nd = {k:[] for k in ks}
    mrr = 0
    for i in range(N):
        sorted_i = np.argsort(-sim[i])
        rel = c2i[companies[i]]
        rank = np.where(sorted_i == i)[0][0] + 1
        mrr += 1/rank
        for k in ks:
            top = set(sorted_i[:k].tolist())
            if i in top: hit[k]+=1
            rik = len(top & rel)
            pk = rik/k; rk = rik/len(rel) if rel else 0
            p[k].append(pk); r[k].append(rk)
            f[k].append(2*pk*rk/(pk+rk) if pk+rk>0 else 0)
            dcg = sum(1/np.log2(j+2) for j, di in enumerate(sorted_i[:k]) if di in rel)
            idcg = sum(1/np.log2(j+2) for j in range(min(k, len(rel))))
            nd[k].append(dcg/idcg if idcg>0 else 0)
    print(f'\n=== {name} ===')
    print(f'  Hit@1={hit[1]/N*100:6.2f}%   Hit@3={hit[3]/N*100:6.2f}%   Hit@5={hit[5]/N*100:6.2f}%   Hit@10={hit[10]/N*100:6.2f}%')
    print(f'  F1@1 ={np.mean(f[1])*100:6.2f}%   F1@3 ={np.mean(f[3])*100:6.2f}%   F1@5 ={np.mean(f[5])*100:6.2f}%   F1@10 ={np.mean(f[10])*100:6.2f}%')
    print(f'  NDCG@1={np.mean(nd[1]):.4f}  NDCG@3={np.mean(nd[3]):.4f}  NDCG@5={np.mean(nd[5]):.4f}  NDCG@10={np.mean(nd[10]):.4f}')
    print(f'  MRR  ={mrr/N:.4f}')
    return {'hit1':hit[1]/N*100, 'hit3':hit[3]/N*100, 'hit5':hit[5]/N*100, 'hit10':hit[10]/N*100,
            'f1':np.mean(f[1])*100, 'f3':np.mean(f[3])*100, 'f5':np.mean(f[5])*100, 'f10':np.mean(f[10])*100,
            'ndcg1':np.mean(nd[1]), 'ndcg5':np.mean(nd[5]), 'mrr':mrr/N}

print('Loading pretrained BGE-base...')
pre = BertModel.from_pretrained('BAAI/bge-base-en')
r_pre = evaluate(pre, 'Pretrained BGE-base (zero-shot)')

import argparse
parser = argparse.ArgumentParser(description='Eval a LoRA checkpoint vs pretrained BGE-base.')
parser.add_argument('checkpoint', nargs='?', default='x-lora_latest.bin',
                    help='LoRA-only state_dict .bin to evaluate (default: x-lora_latest.bin)')
parser.add_argument('--name', default=None, help='Display name for the run (default: filename)')
_args = parser.parse_args()
ckpt_path = _args.checkpoint
ckpt_name = _args.name or f'LoRA checkpoint ({ckpt_path})'

print(f'\nLoading LoRA checkpoint and merging: {ckpt_path}')
from peft import LoraConfig, get_peft_model
from flgo.benchmark.fedrag_classification.config import LORA_R, LORA_ALPHA, LORA_TARGETS, LORA_DROPOUT
base = BertModel.from_pretrained('BAAI/bge-base-en')
cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=LORA_TARGETS, lora_dropout=LORA_DROPOUT, bias='none')
wrapped = get_peft_model(base, cfg)
state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
miss, unexp = wrapped.load_state_dict(state, strict=False)
print(f'  loaded {len(state)} LoRA tensors, {len(miss)} missing (expected: base keys), {len(unexp)} unexpected')
merged = wrapped.merge_and_unload()
r_dp = evaluate(merged, ckpt_name)

# Retention
print('\n' + '='*60)
print('UTILITY RETENTION (DP / Pretrained)')
print('='*60)
for label, k in [('Hit@1','hit1'),('Hit@3','hit3'),('Hit@5','hit5'),('Hit@10','hit10'),('F1@5','f5'),('NDCG@5','ndcg5'),('MRR','mrr')]:
    if r_pre[k] > 0:
        ret = r_dp[k] / r_pre[k] * 100
        delta = r_dp[k] - r_pre[k]
        print(f'  {label:<10}: pretrained={r_pre[k]:7.4f}  DP-LoRA={r_dp[k]:7.4f}  retention={ret:6.1f}%  delta={delta:+.4f}')

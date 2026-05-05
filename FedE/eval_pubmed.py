"""Eval Phase 2 LoRA-only DP checkpoint vs pretrained on PubMed data.

PubMed has all records labeled 'company=pubmed' (single label), so per-class
metrics (P/R/F1 with company-level relevance) collapse. Hit@k and MRR remain
valid because they use diagonal exact match.
"""
import torch, torch.nn.functional as F, json, numpy as np
from transformers import BertModel, BertTokenizer
import sys
sys.path.insert(0, '/root/sda/FedE')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

with open('selected_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
np.random.seed(42)
indices = np.random.choice(len(data), size=min(200, len(data)), replace=False)
test = [data[i] for i in indices]
questions = [d['question'] for d in test]
references = [d['reference'] for d in test]
print(f'Eval set: {len(test)} samples (PubMed)')

tokenizer = BertTokenizer.from_pretrained('BAAI/bge-base-en')
max_length = tokenizer.model_max_length

def evaluate(model, name):
    model.eval().to(device)
    qe, re = [], []
    bs = 32
    for i in range(0, len(questions), bs):
        inp = tokenizer(questions[i:i+bs], return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            qe.append(model(**inp).last_hidden_state.mean(dim=1).cpu())
    for i in range(0, len(references), bs):
        inp = tokenizer(references[i:i+bs], return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            re.append(model(**inp).last_hidden_state.mean(dim=1).cpu())
    Q = F.normalize(torch.cat(qe), dim=-1)
    R = F.normalize(torch.cat(re), dim=-1)
    sim = (Q @ R.t()).numpy()
    N = len(sim)
    ks = [1,3,5,10]
    hit = {k:0 for k in ks}
    mrr = 0
    nd = {k:[] for k in ks}
    for i in range(N):
        sorted_i = np.argsort(-sim[i])
        rank = np.where(sorted_i == i)[0][0] + 1
        mrr += 1/rank
        for k in ks:
            if i in set(sorted_i[:k].tolist()): hit[k] += 1
            # NDCG with single-relevant (target=i): DCG = 1/log2(rank+1) if i in top-k else 0
            if i in sorted_i[:k]:
                pos = np.where(sorted_i[:k] == i)[0][0]
                nd[k].append(1.0 / np.log2(pos + 2))
            else:
                nd[k].append(0)
    print(f'\n=== {name} ===')
    print(f'  Hit@1={hit[1]/N*100:6.2f}%   Hit@3={hit[3]/N*100:6.2f}%   Hit@5={hit[5]/N*100:6.2f}%   Hit@10={hit[10]/N*100:6.2f}%')
    print(f'  NDCG@1={np.mean(nd[1]):.4f}  NDCG@3={np.mean(nd[3]):.4f}  NDCG@5={np.mean(nd[5]):.4f}  NDCG@10={np.mean(nd[10]):.4f}')
    print(f'  MRR  ={mrr/N:.4f}')
    return {'hit1':hit[1]/N*100, 'hit3':hit[3]/N*100, 'hit5':hit[5]/N*100, 'hit10':hit[10]/N*100,
            'ndcg5':np.mean(nd[5]), 'mrr':mrr/N}

print('Loading pretrained...')
pre = BertModel.from_pretrained('BAAI/bge-base-en')
r_pre = evaluate(pre, 'Pretrained BGE-base (zero-shot, PubMed eval)')

import argparse
parser = argparse.ArgumentParser(description='Eval a LoRA checkpoint on PubMed data vs pretrained.')
parser.add_argument('checkpoint', nargs='?', default='x-lora_latest.bin',
                    help='LoRA-only state_dict .bin to evaluate')
parser.add_argument('--name', default=None, help='Display name for the run (default: filename)')
_args = parser.parse_args()
ckpt_path = _args.checkpoint
ckpt_name = _args.name or f'LoRA on PubMed ({ckpt_path})'

print(f'\nLoading LoRA checkpoint: {ckpt_path}')
from peft import LoraConfig, get_peft_model
from flgo.benchmark.fedrag_classification.config import LORA_R, LORA_ALPHA, LORA_TARGETS, LORA_DROPOUT
base = BertModel.from_pretrained('BAAI/bge-base-en')
cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=LORA_TARGETS, lora_dropout=LORA_DROPOUT, bias='none')
wrapped = get_peft_model(base, cfg)
state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
wrapped.load_state_dict(state, strict=False)
merged = wrapped.merge_and_unload()
r_dp = evaluate(merged, ckpt_name)

print('\n' + '='*60)
print('UTILITY: PubMed DP-LoRA / Pretrained zero-shot')
print('='*60)
for label, k in [('Hit@1','hit1'),('Hit@3','hit3'),('Hit@5','hit5'),('Hit@10','hit10'),('NDCG@5','ndcg5'),('MRR','mrr')]:
    delta = r_dp[k] - r_pre[k]
    if r_pre[k] > 0:
        ret = r_dp[k] / r_pre[k] * 100
    else:
        ret = float('nan')
    print(f'  {label:<8}: pretrained={r_pre[k]:7.4f}  DP-LoRA={r_dp[k]:7.4f}  delta={delta:+7.4f}  retention={ret:6.2f}%')

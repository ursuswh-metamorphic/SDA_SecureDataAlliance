"""
DP-LoRA FedRAG Training
========================
Instead of fine-tuning all 109M params with DP noise,
only fine-tune LoRA adapters (295K params = 0.27%).

Same sigma -> noise spread over 371x fewer params -> much better utility.

Key difference from full DP:
  Full DP:  noise on 109M params -> signal destroyed
  DP-LoRA:  noise on 295K params -> signal preserved in frozen base
"""
import os, sys
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn.functional as F
import numpy as np
import json
from datetime import datetime
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model

# Upstream embedding backbone: MedCPT Article Encoder (PubMed biomedical)
# https://huggingface.co/ncbi/MedCPT-Article-Encoder
EMBEDDING_MODEL_NAME = "ncbi/MedCPT-Article-Encoder"

sys.path.insert(0, os.path.dirname(__file__))
from privacy.rdp_accountant import compute_epsilon, find_noise_multiplier, RDPAccountant

# ── Config ─────────────────────────────────────────────────────────────────
TARGET_EPSILON    = 20.0
TARGET_DELTA      = 1e-5
NUM_ROUNDS        = 25
NUM_CLIENTS       = 5
BATCH_SIZE        = 8
LEARNING_RATE     = 1e-5  # Conservative LR to prevent gradient explosion with DP noise
LORA_R            = 8
LORA_ALPHA        = 16
LORA_DROPOUT      = 0.05
LORA_TARGETS      = ['query', 'value']

sampling_rate = NUM_CLIENTS / NUM_CLIENTS  # All clients

# Calibrate sigma
calibrated_sigma = find_noise_multiplier(TARGET_EPSILON, NUM_ROUNDS, sampling_rate, TARGET_DELTA)
print(f'Calibrated sigma = {calibrated_sigma:.4f} for eps={TARGET_EPSILON}')
eps_check, _ = compute_epsilon(NUM_ROUNDS, calibrated_sigma, sampling_rate, TARGET_DELTA)
print(f'Verification: eps={eps_check:.4f}')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ── Load data ──────────────────────────────────────────────────────────────
with open('pubmed_train.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

client_data = [[] for _ in range(NUM_CLIENTS)]
for i, d in enumerate(data):
    client_data[i % NUM_CLIENTS].append(d)
print(f'Data: {len(data)} total, {[len(c) for c in client_data]} per client')

tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
max_length = tokenizer.model_max_length


def create_lora_model():
    base = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)
    config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGETS,
        lora_dropout=LORA_DROPOUT, bias='none',
    )
    lora_model = get_peft_model(base, config)
    trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in lora_model.parameters())
    print(f'LoRA model: {trainable:,} trainable / {total:,} total ({trainable/total*100:.2f}%)')
    return lora_model


def train_client(client_id, global_state, client_dataset, sigma, clip_norm=0.1):
    """Train one client with DP-SGD on LoRA params only."""
    local_model = create_lora_model().to(device)
    local_model.load_state_dict(global_state, strict=False)
    local_model.train()

    lora_params = [p for p in local_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=LEARNING_RATE, weight_decay=0.01)

    np.random.shuffle(client_dataset)
    num_steps = min(50, max(1, len(client_dataset) // BATCH_SIZE))

    for step in range(num_steps):
        batch = client_dataset[step*BATCH_SIZE : (step+1)*BATCH_SIZE]
        if len(batch) == 0:
            continue

        questions = [b['question'] for b in batch]
        references = [b['reference'] for b in batch]
        bs = len(questions)

        accumulated = [torch.zeros_like(p.data) for p in lora_params]

        for i in range(bs):
            optimizer.zero_grad()

            q_inp = tokenizer([questions[i]], return_tensors='pt', padding=True,
                            truncation=True, max_length=max_length).to(device)
            # MedCPT uses [CLS] pooling (see HF model card)
            q_out = local_model(**q_inp).last_hidden_state[:, 0, :]

            r_inp = tokenizer([references[i]], return_tensors='pt', padding=True,
                            truncation=True, max_length=max_length).to(device)
            r_out = local_model(**r_inp).last_hidden_state[:, 0, :]

            sim = F.cosine_similarity(q_out, r_out)
            loss = 1.0 - sim.mean()
            loss.backward()

            grad_norm = torch.sqrt(sum(
                p.grad.detach().norm()**2 for p in lora_params if p.grad is not None
            )).item()

            clip_coef = min(1.0, clip_norm / (grad_norm + 1e-8))
            for j, p in enumerate(lora_params):
                if p.grad is not None:
                    accumulated[j] += p.grad.detach() * clip_coef

        # Scale noise by clip_norm (not sigma * clip_norm directly on accumulated)
        for j, p in enumerate(lora_params):
            noise = torch.randn_like(accumulated[j]) * (sigma * clip_norm / float(bs))
            p.grad = accumulated[j] / float(bs) + noise

        # Clip final gradient to prevent explosion
        torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
        optimizer.step()

        if step % max(1, num_steps//3) == 0:
            print(f'  [Client {client_id}] step {step}/{num_steps}, loss={loss.item():.4f}, grad_norm={grad_norm:.6f}')

    lora_state = {k: v.cpu() for k, v in local_model.state_dict().items()
                  if 'lora' in k.lower()}
    del local_model
    torch.cuda.empty_cache()
    return lora_state


def aggregate_lora(global_state, client_states, sigma, clip_norm=0.1):
    """Average LoRA params with server-level DP noise and update clipping."""
    K = len(client_states)
    lora_keys = list(client_states[0].keys())
    avg_state = {}

    # Compute client updates (delta from global) and clip each
    clipped_deltas = []
    for cs in client_states:
        delta = {}
        flat = []
        for key in lora_keys:
            d = cs[key].float() - global_state[key].float() if key in global_state else cs[key].float()
            delta[key] = d
            flat.append(d.flatten())
        update_norm = torch.cat(flat).norm().item()
        clip_coef = min(1.0, clip_norm / (update_norm + 1e-8))
        clipped_deltas.append({k: v * clip_coef for k, v in delta.items()})

    for key in lora_keys:
        avg_delta = torch.stack([cd[key] for cd in clipped_deltas]).mean(dim=0)
        noise_std = sigma * clip_norm / K
        noise = torch.randn_like(avg_delta) * noise_std
        avg_state[key] = global_state[key].float() + avg_delta + noise if key in global_state else avg_delta + noise

    new_state = {k: v.clone() for k, v in global_state.items()}
    for key in avg_state:
        if key in new_state:
            new_state[key] = avg_state[key]
    return new_state


# ── Create global model ──────────────────────────────────────────────────
global_model = create_lora_model().to(device)

accountant = RDPAccountant(
    noise_multiplier=calibrated_sigma,
    sample_rate=sampling_rate,
    delta=TARGET_DELTA,
)

print(f'\n{"="*60}')
print(f'[DP-LoRA] Starting training')
print(f'  eps={TARGET_EPSILON}, sigma={calibrated_sigma:.4f}')
print(f'  LoRA r={LORA_R}, alpha={LORA_ALPHA}, targets={LORA_TARGETS}')
print(f'  Trainable: 295K / 109M (0.27%)')
print(f'{"="*60}\n')

os.makedirs('checkpoints', exist_ok=True)

for round_num in range(1, NUM_ROUNDS + 1):
    print(f'--- Round {round_num}/{NUM_ROUNDS} ---')
    global_state = {k: v.cpu() for k, v in global_model.state_dict().items()}

    client_states = []
    for cid in range(NUM_CLIENTS):
        cs = train_client(cid, global_state, client_data[cid],
                         sigma=calibrated_sigma, clip_norm=0.1)
        client_states.append(cs)

    new_state = aggregate_lora(global_state, client_states,
                               sigma=calibrated_sigma, clip_norm=0.1)
    global_model.load_state_dict(new_state, strict=False)

    accountant.step()
    eps_spent = accountant.get_epsilon()
    print(f'  [Privacy] eps_spent={eps_spent:.4f} / {TARGET_EPSILON}')

    if round_num % 5 == 0 or round_num == NUM_ROUNDS:
        ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        path = f'checkpoints/lora_round{round_num}_{ts}.pt'
        torch.save(global_model.state_dict(), path)
        print(f'  Saved: {path}')

# ── Save final model ──────────────────────────────────────────────────────
ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# Merge LoRA into base and save
merged = global_model.merge_and_unload()
merged_path = f'x-model_lora_merged_{ts}.bin'
torch.save({'model.' + k: v for k, v in merged.state_dict().items()}, merged_path)

print(f'\n[DP-LoRA] Training complete!')
print(f'  Merged model: {merged_path}')
print(f'  Final eps: {accountant.get_epsilon():.4f}')

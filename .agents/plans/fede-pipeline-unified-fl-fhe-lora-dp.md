# Feature: Unified FedE Pipeline — FL + FHE + LoRA/qLoRA + DP

The following plan should be complete, but it is important that you validate documentation and codebase patterns and task sanity before you start implementing.

Pay special attention to naming of existing utils, types and models. Import from the right files etc.

## Feature Description

Refactor the FedE training stack so that **one cohesive pipeline** implements the 5-step upstream privacy flow: Federated Learning + LoRA/qLoRA + Differential Privacy + true Homomorphic-Encryption aggregation. Today the four components exist but in **disconnected scripts** ([main_dp_lora_eps20.py](FedE/main_dp_lora_eps20.py) bypasses the flgo framework; [fedrag_CKKS.py](FedE/flgo/algorithm/fedrag_CKKS.py) decrypts before aggregating; [fedrag_dp.py](FedE/flgo/algorithm/fedrag_dp.py) wraps the full 109M-param model). This feature wires them into a single algorithm class that ships only ~295K LoRA-adapter parameters, encrypts them end-to-end with CKKS, and performs the homomorphic average without ever decrypting at the server.

## User Story

As a **federated-learning researcher** working on medical-RAG privacy guarantees,
I want to **run one entrypoint that trains the BGE-base embedding model with FL + FHE + LoRA + DP simultaneously**,
So that **I can publish a result demonstrating the three-layer protection (no data sharing, no plaintext gradients on the wire, no reverse-engineering even after decrypt) without manually stitching four scripts together each experiment**.

## Problem Statement

The existing codebase contains four valuable but disjoint implementations:

1. [main_dp_lora_eps20.py](FedE/main_dp_lora_eps20.py) — correct per-sample DP-SGD on LoRA-only params, validated at ε=20 with 98.4 % retention on PubMed, but **bypasses flgo** entirely.
2. [fedrag_dp.py](FedE/flgo/algorithm/fedrag_dp.py) — RDP-correct user-level DP at the server, but on **full-model state**.
3. [fedrag_CKKS.py](FedE/flgo/algorithm/fedrag_CKKS.py) — CKKS context plumbing, but the server **decrypts before averaging** ([fedrag_CKKS.py:99-104](FedE/flgo/algorithm/fedrag_CKKS.py#L99-L104), [:181-190](FedE/flgo/algorithm/fedrag_CKKS.py#L181-L190)), defeating FHE.
4. [main.py](FedE/main.py) — vanilla flgo baseline, ships full-model state ([fedrag.py:80-85](FedE/flgo/algorithm/fedrag.py#L80-L85)).

Because each script owns its own `main`, no experiment ever combines all three privacy layers. The 5-step upstream architecture (server-encrypt → client-decrypt → DP-train → client-encrypt → server-aggregate-on-ciphertext) is therefore **impossible to demonstrate** with the current code.

## Solution Statement

Introduce two new flgo algorithm classes — `fedrag_lora` (Phase 1+2) and `fedrag_lora_ckks` (Phase 4) — plus modifications to the benchmark calculator (Phase 3) and the `get_model()` factory (Phase 1+5). Both classes subclass the existing `BasicServer`/`BasicClient` and override `pack()` / `unpack()` / `aggregate()` so that **only LoRA-adapter state_dicts** are ever serialised, and in the CKKS variant they are **encrypted client-side and summed homomorphically server-side**. The DP step from the standalone LoRA script is moved into `Client.train()` so per-sample clipping + Gaussian noise apply only to `requires_grad=True` parameters (the LoRA adapters). qLoRA (4-bit base via `bitsandbytes`) is gated on Linux to keep the Windows dev path working.

## Feature Metadata

**Feature Type**: Refactor (consolidates four scripts into one pipeline)
**Estimated Complexity**: High (Phase 4 CKKS aggregation is research-grade; rest is medium)
**Primary Systems Affected**: `FedE/flgo/algorithm/`, `FedE/flgo/benchmark/fedrag_classification/`, top-level `FedE/main_*.py`, `FedE/requirements.txt`, `FedE/TRAIN_GUIDE.md`
**Dependencies**:
- `peft >= 0.7.0` (already used in [main_dp_lora_eps20.py:23](FedE/main_dp_lora_eps20.py#L23))
- `tenseal >= 0.3.14` (already used in [fedrag_CKKS.py:12](FedE/flgo/algorithm/fedrag_CKKS.py#L12))
- `bitsandbytes >= 0.43` (Linux/Vast.ai only — new)
- existing `transformers`, `torch >= 2.1`, in-tree `privacy/rdp_accountant.py`

---

## CONTEXT REFERENCES

### Relevant Codebase Files — IMPORTANT: YOU MUST READ THESE BEFORE IMPLEMENTING!

- [FedE/flgo/algorithm/fedrag.py](FedE/flgo/algorithm/fedrag.py) (lines 9-85, 88-139) — Why: Canonical `Server.aggregate` and `Client.train` — your new classes must subclass these.
- [FedE/flgo/algorithm/fedbase.py](FedE/flgo/algorithm/fedbase.py) (lines 393-407, 825-851) — Why: Defines the `pack`/`unpack` contract you will override. `Server.pack` returns `{"model": copy.deepcopy(self.model)}` by default; `Client.pack` returns `{"model": model}`.
- [FedE/flgo/algorithm/fedrag_dp.py](FedE/flgo/algorithm/fedrag_dp.py) (lines 36-100, 244-356, 400-490, 521-590) — Why: **Reference implementation** of per-sample DP. Reuse helpers `_get_single_sample`, `_get_batch_size`, `_compute_grad_norm`, `_state_dict_diff_flat` ([fedrag_dp.py:521-590](FedE/flgo/algorithm/fedrag_dp.py#L521-L590)). Do NOT modify this file.
- [FedE/flgo/algorithm/fedrag_CKKS.py](FedE/flgo/algorithm/fedrag_CKKS.py) (lines 15-119, 121-204, 235-355) — Why: **Anti-pattern** — server holds the secret key and decrypts before aggregating. Your Phase 4 implementation must invert the key flow (clients hold secret key) and aggregate **on ciphertexts**. Keep this file untouched as historical reference.
- [FedE/main_dp_lora_eps20.py](FedE/main_dp_lora_eps20.py) (entire file, especially lines 28-46, 63-74, 77-140, 143-172, 178-227) — Why: **Canonical DP-LoRA recipe** — calibrated σ for ε=20 (line 43), `create_lora_model()` factory (line 63-74), per-sample clip+noise loop (line 100-127), LoRA-only state_dict filter (line 136-137), `merge_and_unload()` for eval checkpoint (line 221-223). The new flgo client must produce numerically equivalent updates.
- [FedE/flgo/benchmark/fedrag_classification/core.py](FedE/flgo/benchmark/fedrag_classification/core.py) (lines 128-188) — Why: `TaskCalculator` whose `compute_client_loss` returns `loss_1 + 100 * loss_2` (CrossEntropy on cosine + MSE distillation). Phase 3 replaces this with InfoNCE + KL.
- [FedE/flgo/benchmark/fedrag_classification/config.py](FedE/flgo/benchmark/fedrag_classification/config.py) (entire file, 19 lines) — Why: `get_model()` is the single point that loads BGE-base. Phase 1 wraps it with PEFT; Phase 5 adds optional 4-bit quantisation here.
- [FedE/privacy/rdp_accountant.py](FedE/privacy/rdp_accountant.py) (lines 1-50, plus exports `RDPAccountant`, `compute_epsilon`, `find_noise_multiplier`) — Why: σ calibration and ε-tracking. Same module already used by [main_dp_lora_eps20.py:26](FedE/main_dp_lora_eps20.py#L26) and [fedrag_dp.py:51](FedE/flgo/algorithm/fedrag_dp.py#L51).
- [FedE/main.py](FedE/main.py) (entire file, 19 lines) — Why: minimal flgo entrypoint shape. New `main_lora.py` and `main_full.py` mirror this pattern.
- [FedE/main_dp.py](FedE/main_dp.py) — Why: Shows how to plumb DP options through `flgo.init(..., option={'dp_enabled': True, 'target_epsilon': ...})`.
- [FedE/eval_compare.py](FedE/eval_compare.py) (line 23) — Why: Loads `pubmed_train.json` for downstream RAG eval. Will be extended in Phase 6.

### New Files to Create

- `FedE/flgo/algorithm/fedrag_lora.py` — `Server` + `Client` classes that ship LoRA-only `state_dict` and apply per-sample DP on adapter params (Phase 1 + 2).
- `FedE/flgo/algorithm/fedrag_lora_ckks.py` — Subclass of `fedrag_lora.Server` / `Client` performing **homomorphic** aggregation on CKKS ciphertexts (Phase 4).
- `FedE/main_lora.py` — Entrypoint for non-encrypted LoRA+DP run (regression target ≡ ε=20 baseline).
- `FedE/main_full.py` — Entrypoint for full pipeline (FHE + DP-LoRA + qLoRA gate).
- `FedE/tests/test_lora_filter.py` — Unit test asserting LoRA-key filter ships exactly 295 K params.
- `FedE/tests/test_ckks_correctness.py` — Cryptographic correctness test: encrypted-vs-plaintext aggregate ≤ 1e-3 elementwise.

### Relevant Documentation — YOU SHOULD READ THESE BEFORE IMPLEMENTING!

- [PEFT — LoRA configuration](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora) — Specific section: *target_modules, r, lora_alpha*. Why: Confirms `target_modules=['query','value']` produces ~295 K trainable params on `BAAI/bge-base-en` (matches [main_dp_lora_eps20.py:38](FedE/main_dp_lora_eps20.py#L38)).
- [PEFT — `prepare_model_for_kbit_training`](https://huggingface.co/docs/peft/main/en/package_reference/peft_model#peft.prepare_model_for_kbit_training) — Why: Required wrapper before `get_peft_model` when base is 4-bit (Phase 5).
- [TenSEAL — CKKS vector ops](https://github.com/OpenMined/TenSEAL/blob/main/tutorials/Tutorial%200%20-%20Getting%20Started.ipynb) — Specific section: *ckks_vector + scalar mul, vector + vector*. Why: Confirms `ts.ckks_vector(ctx, x) + ts.ckks_vector(ctx, y)` returns ciphertext (Phase 4 aggregation).
- [TenSEAL — context serialisation](https://github.com/OpenMined/TenSEAL#tenseal-context) — Specific section: *save_secret_key=False*. Why: Required to ship public-only context to server.
- [Mironov 2017 — Rényi DP](https://arxiv.org/abs/1702.07476) — Why: Already implemented in [privacy/rdp_accountant.py](FedE/privacy/rdp_accountant.py); only need to confirm the `find_noise_multiplier(target_epsilon=20, ...)` call signature.
- [BitsAndBytes — `BitsAndBytesConfig`](https://huggingface.co/docs/transformers/main/en/quantization#4-bit) — Specific section: *load_in_4bit + bnb_4bit_quant_type='nf4'*. Why: Phase 5 4-bit quantisation kwargs.
- [Andrew et al. 2021 — Adaptive DP clipping](https://arxiv.org/abs/1905.03871) — Why: Already implemented at [fedrag_dp.py:432-444](FedE/flgo/algorithm/fedrag_dp.py#L432-L444); we **inherit**, do not re-derive.

### Patterns to Follow

**Naming Conventions**
- Algorithm files: `fedrag_<variant>.py` (e.g. `fedrag.py`, `fedrag_dp.py`, `fedrag_CKKS.py`). Phase-1 file is therefore `fedrag_lora.py`; Phase-4 is `fedrag_lora_ckks.py`.
- Entrypoints: `main_<variant>.py` at `FedE/` root.
- Constants in entrypoints: ALL_CAPS (`LORA_R`, `BATCH_SIZE`, `TARGET_EPSILON` — see [main_dp_lora_eps20.py:29-38](FedE/main_dp_lora_eps20.py#L29-L38)).
- LoRA filter helper: literal predicate `'lora' in k.lower()` (matches [main_dp_lora_eps20.py:137](FedE/main_dp_lora_eps20.py#L137)).

**flgo Subclass Pattern** — mirror [fedrag_dp.py:61](FedE/flgo/algorithm/fedrag_dp.py#L61):
```
from .fedbase import BasicServer, BasicClient
class Server(BasicServer):
    def run(self): ...
    def aggregate(self, model_old, models, *args, **kwargs): ...
class Client(BasicClient):
    def train(self, model, local_model): ...
    def reply(self, svr_pkg): ...
```
**Do NOT** subclass `fedrag.Server`/`Client` directly — always go through `BasicServer`/`BasicClient` (this is what `fedrag.py`, `fedrag_dp.py`, `fedrag_CKKS.py` all do; flgo's algorithm dispatch resolves the module's top-level `Server`/`Client` symbols).

**`pack` / `unpack` Contract** — follow [fedbase.py:393-407, 825-851](FedE/flgo/algorithm/fedbase.py#L393-L851):
- Server returns a **dict** with at minimum a `"model"` key (or in our case a `"lora_state"` key plus manifest).
- Client returns a **dict** with the same keys it expects to round-trip.

**Per-sample DP Pattern** — mirror [fedrag_dp.py:447-478](FedE/flgo/algorithm/fedrag_dp.py#L447-L478):
```
params = [p for p in local_model.parameters() if p.requires_grad]   # ← LoRA-only
accumulated = [torch.zeros_like(p.data) for p in params]
for i in range(batch_size):
    single = _get_single_sample(batch_data, i)
    local_model.zero_grad()
    loss = ...
    loss.backward()
    norm = _compute_grad_norm(params)
    coef = min(1.0, clip_norm / (norm + 1e-8))
    for j, p in enumerate(params):
        accumulated[j] += p.grad.detach() * coef
for j, p in enumerate(params):
    noise = torch.randn_like(accumulated[j]) * (sigma * clip_norm)
    p.grad = (accumulated[j] + noise) / float(batch_size)
optimizer.step()
```

**LoRA-Adapter State_Dict Filter** — mirror [main_dp_lora_eps20.py:136-137](FedE/main_dp_lora_eps20.py#L136-L137):
```
lora_state = {k: v.cpu() for k, v in model.state_dict().items() if 'lora' in k.lower()}
```

**CKKS Context** — mirror [fedrag_CKKS.py:106-119](FedE/flgo/algorithm/fedrag_CKKS.py#L106-L119) but **invert key holder**:
```
ctx = ts.context(ts.SCHEME_TYPE.CKKS,
                 poly_modulus_degree=8192,
                 coeff_mod_bit_sizes=[60, 40, 40, 40, 60])  # ← extra 40 vs current
ctx.global_scale = 2**40
ctx.generate_galois_keys()
```
**Do NOT** call `ctx.make_context_public()` on the client — clients keep secret key. **DO** call it on the serialized copy sent to server (`ctx.serialize(save_secret_key=False)`).

**RDP Calibration** — mirror [main_dp_lora_eps20.py:43-46](FedE/main_dp_lora_eps20.py#L43-L46):
```
sampling_rate = clients_per_round / total_clients      # 5/5 = 1.0 in the validated baseline
calibrated_sigma = find_noise_multiplier(target_epsilon, num_rounds, sampling_rate, target_delta)
eps_check, _ = compute_epsilon(num_rounds, calibrated_sigma, sampling_rate, target_delta)
```

**Logging Style** — mirror [main_dp_lora_eps20.py:133](FedE/main_dp_lora_eps20.py#L133), [fedrag_dp.py:485-489](FedE/flgo/algorithm/fedrag_dp.py#L485-L489): print rank-style `[DP-Client {id}] step {s}/{n}, loss=...`.

**Anti-Patterns to Avoid**
- ❌ Decrypting at server before aggregation — see [fedrag_CKKS.py:181-190](FedE/flgo/algorithm/fedrag_CKKS.py#L181-L190).
- ❌ Sending the full model state when only LoRA is trainable — see [fedrag.py:81](FedE/flgo/algorithm/fedrag.py#L81).
- ❌ Per-coordinate noise without per-sample clipping — see [fedrag.py:115-127](FedE/flgo/algorithm/fedrag.py#L115-L127) (the `dp_enabled` branch in baseline `Client.train` does batch-level clipping; this is the very bug `fedrag_dp.py` was created to fix).
- ❌ Importing `peft.LoraConfig` inside `get_model` without freezing the base — `get_peft_model` already freezes non-LoRA params, but explicit assertion is required (see Validate step under Task 1.2).

---

## IMPLEMENTATION PLAN

### Phase 1: Foundation — LoRA-Only State Transport

Wrap `BertModel` with PEFT in the model factory; introduce a new flgo algorithm whose server↔client `pack`/`unpack` ship only `'lora.*'` keys; baseline `Client.train` reuses the existing non-DP path. Reduces wire payload from ~430 MB / round to ~1.2 MB / round.

**Tasks:**
- Edit `config.py:get_model` to apply `LoraConfig(r=8, lora_alpha=16, target_modules=['query','value'], lora_dropout=0.05, bias='none')` then `get_peft_model`.
- Create `fedrag_lora.py` with `Server`/`Client` overriding `pack` / `unpack` / `aggregate` to handle LoRA-only state.
- Create `main_lora.py` mirroring [main.py](FedE/main.py) with `algorithm=fedrag_lora`.
- Add `tests/test_lora_filter.py` asserting param count.

### Phase 2: Per-Sample DP on LoRA Adapters

Move the DP-SGD loop from `main_dp_lora_eps20.py:100-127` into `fedrag_lora.Client.train`, gated on `option['dp_enabled']`. Keep server aggregation as plain mean (no server-side DP) so the regression baseline is reproducible byte-for-byte.

**Tasks:**
- Add DP branch in `fedrag_lora.Client.train` reusing `_get_single_sample`, `_compute_grad_norm` from [fedrag_dp.py](FedE/flgo/algorithm/fedrag_dp.py).
- Plumb `target_epsilon`, `target_delta`, `dp_clip_norm`, `dp_enabled` from the `option` dict.
- Calibrate σ once in `Server.__init__` via `find_noise_multiplier` and broadcast to clients (or compute it client-side from the same params).
- Tick the `RDPAccountant` once per round in `Server.aggregate`, log `eps_spent`.

### Phase 3: Formal InfoNCE + KL Distillation

Replace the informal `CrossEntropyLoss(cos_sim, diag) + 100*MSE` in `core.py:compute_client_loss` with temperature-scaled InfoNCE and KL-divergence KD. Land **after** Phase 2 baseline is locked.

**Tasks:**
- Add `temperature` and `kd_weight` params to `TaskCalculator.__init__`.
- Rewrite `compute_client_loss` to use `criterion(logits / temperature, label)` (InfoNCE) + `KLDivLoss(F.log_softmax(student/τ, -1), F.softmax(teacher/τ, -1)) * τ**2`.
- Plumb `temperature`, `kd_weight` through `option` dict.

### Phase 4: True Homomorphic Aggregation

Subclass `fedrag_lora.Server`/`Client` into `fedrag_lora_ckks.py`. Invert key-holder (clients hold secret key, server only public/eval). Server aggregates **on ciphertexts** without ever calling `.decrypt()`. Bandwidth ≈ 1.1 MB/client/round; tractable because LoRA is small.

**Tasks:**
- Build `_chunked_encrypt(state_dict, ctx)` and `_chunked_decrypt(ctx_list, manifest, secret_ctx)` helpers.
- `Client.pack`: pre-divide `delta_i / K` (avoids server-side plaintext-mul level cost), encrypt, attach manifest.
- `Server.aggregate`: sum ciphertexts per (key, chunk_idx); never call `.decrypt`; assert `self.context.has_secret_key() is False`.
- `Client.unpack`: decrypt, reshape via manifest, `load_state_dict(strict=False)`.
- Add `tests/test_ckks_correctness.py` comparing encrypted vs plain aggregate within 1e-3.

### Phase 5: qLoRA (Linux/Vast.ai only)

Add 4-bit NF4 quantisation to the frozen base, gated on `option['use_qlora']`. Skip on Windows (bitsandbytes wheels are flaky). Allows batch size 8 → 16 on the same VRAM budget.

**Tasks:**
- Extend `config.py:get_model` to accept `quantize: bool` kwarg; add `BitsAndBytesConfig` import.
- Wrap with `prepare_model_for_kbit_training` before `get_peft_model`.
- Add `bitsandbytes>=0.43; sys_platform == 'linux'` to `requirements.txt`.
- Add platform guard in `main_full.py`.

### Phase 6: End-to-End Integration & Eval

Single entrypoint `main_full.py` exercising algorithm=`fedrag_lora_ckks` with `dp_enabled=True, use_qlora=<gated>, target_epsilon=20, temperature=0.05`. Extend `eval_compare.py` to load merged checkpoints from any phase. Update `TRAIN_GUIDE.md` with phase runbook.

**Tasks:**
- Write `main_full.py`.
- Extend `eval_compare.py` to optionally call `model.merge_and_unload()` if a PEFT model is detected.
- Append Phase-by-phase runbook section to `TRAIN_GUIDE.md`.

---

## STEP-BY-STEP TASKS

IMPORTANT: Execute every task in order, top to bottom. Each task is atomic and independently testable.

### Task Format Guidelines
- **CREATE**: New files / components
- **UPDATE**: Modify existing files
- **ADD**: Insert new functionality into existing code
- **REMOVE**: Delete deprecated code (none expected here)
- **REFACTOR**: Restructure without changing behavior
- **MIRROR**: Copy pattern from elsewhere in codebase

---

### Task 1.1 — UPDATE `FedE/flgo/benchmark/fedrag_classification/config.py`

- **IMPLEMENT**: Wrap `BertModel.from_pretrained('BAAI/bge-base-en')` with `peft.get_peft_model(LoraConfig(...))`. Accept optional `quantize: bool=False` kwarg (Phase 5 will fill it in; Phase 1 simply ignores when `False`). Print trainable-param ratio identical to [main_dp_lora_eps20.py:71-73](FedE/main_dp_lora_eps20.py#L71-L73).
- **PATTERN**: [main_dp_lora_eps20.py:63-74](FedE/main_dp_lora_eps20.py#L63-L74) (`create_lora_model`).
- **IMPORTS**: `from peft import LoraConfig, get_peft_model`.
- **GOTCHA**: `get_peft_model` returns a `PeftModel` whose `forward` signature differs from `BertModel`. The flgo training loop calls `model(**inputs)` (see [core.py:146](FedE/flgo/benchmark/fedrag_classification/core.py#L146)), and `PeftModel.forward` forwards kwargs to the wrapped base, so this works. **Do NOT** unwrap with `.base_model` or `.merge_and_unload()` here — that breaks LoRA training.
- **VALIDATE**: `python -c "import sys; sys.path.insert(0,'FedE'); from flgo.benchmark.fedrag_classification.config import get_model; m=get_model(); t=sum(p.numel() for p in m.parameters() if p.requires_grad); assert 290000 < t < 300000, t; print(f'OK trainable={t}')"`

---

### Task 1.2 — CREATE `FedE/flgo/algorithm/fedrag_lora.py`

- **IMPLEMENT**: `Server(BasicServer)` with overrides `pack(client_id, mtype=0)`, `aggregate(model_old, models)`, `run` (copy from [fedrag.py:9-85](FedE/flgo/algorithm/fedrag.py#L9-L85) verbatim then replace `self.model.model.state_dict()` saves with the LoRA-filtered dict). `Client(BasicClient)` with overrides `unpack(received_pkg)`, `pack(model)`, `train(model, local_model)`, `reply(svr_pkg)`.
- **PATTERN**: Subclass shape from [fedrag.py:9, 88](FedE/flgo/algorithm/fedrag.py#L9), [fedrag_dp.py:61](FedE/flgo/algorithm/fedrag_dp.py#L61).
- **IMPORTS**:
  ```
  import copy, os, torch
  from datetime import datetime
  from .fedbase import BasicServer, BasicClient
  ```
- **SERVER.PACK** — must return `{"lora_state": <dict>}`:
  ```
  def pack(self, client_id, mtype=0, *args, **kwargs):
      sd = self.model.state_dict()
      lora_state = {k: v.detach().cpu() for k, v in sd.items() if 'lora' in k.lower()}
      return {"lora_state": lora_state}
  ```
- **CLIENT.UNPACK** — must accept and apply LoRA-only state:
  ```
  def unpack(self, received_pkg):
      lora_state = received_pkg["lora_state"]
      self.model.load_state_dict(lora_state, strict=False)   # base weights untouched
      return self.model
  ```
- **CLIENT.PACK** — symmetric:
  ```
  def pack(self, model, *args, **kwargs):
      sd = model.state_dict()
      lora_state = {k: v.detach().cpu() for k, v in sd.items() if 'lora' in k.lower()}
      return {"lora_state": lora_state}
  ```
- **SERVER.AGGREGATE** — operate on the list of LoRA dicts:
  ```
  def aggregate(self, model_old, models, *args, **kwargs):
      # `models` already comes from BasicServer.unpack, but our pack returns dicts
      # so models will be a list of {'lora_state': dict}. Use _unpack_helpers.
      lora_dicts = [c["lora_state"] for c in models]
      keys = list(lora_dicts[0].keys())
      avg = {k: torch.stack([d[k].float() for d in lora_dicts]).mean(0) for k in keys}
      sd = model_old.state_dict()
      for k, v in avg.items():
          sd[k] = v.to(sd[k].device).to(sd[k].dtype)
      model_old.load_state_dict(sd, strict=False)
      return model_old
  ```
  **GOTCHA**: `BasicServer.iterate` calls `self.communicate(...)` then `self.aggregate(self.model, models)` where `models` is the **list of unpacked dicts**, not torch modules. Verify by reading the `iterate` body in `fedbase.py` and adjusting the unpack contract; if the framework's `unpack` treats `lora_state` as a generic key, it will accumulate it as `models = {"lora_state": [...]}` (list-of-dicts inside a dict). **Test on a 1-round dry run before continuing**.
- **CLIENT.TRAIN** (Phase-1 scope: non-DP, identical to baseline) — copy [fedrag.py:89-133](FedE/flgo/algorithm/fedrag.py#L89-L133) but **do not** include the DP branch (Phase 2 adds it). Keep `compute_server_loss` + `compute_client_loss` calls unchanged.
- **CLIENT.REPLY**: identical to [fedrag.py:135-139](FedE/flgo/algorithm/fedrag.py#L135-L139).
- **VALIDATE** (1-round smoke): `python FedE/main_lora.py 2>&1 | grep -E "(trainable=|Round 1|aggregate)"` — should show `trainable=295688` (or similar) and one round complete without exceptions.

---

### Task 1.3 — CREATE `FedE/main_lora.py`

- **IMPLEMENT**: Minimal entrypoint exactly like [main.py](FedE/main.py) but with `algorithm=fedrag_lora`, `task='./num5_alpha05_lora'` (separate task path so existing baseline is untouched).
- **PATTERN**: [main.py](FedE/main.py) (entire file).
- **IMPORTS**:
  ```
  import os
  os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
  os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
  import flgo
  import flgo.algorithm.fedrag_lora as fedrag_lora
  ```
- **OPTION DICT**:
  ```
  option = {
      'num_rounds': 25, 'num_epochs': 1, 'gpu': 0, 'batch_size': 8,
      'learning_rate': 1e-5,
      'dp_enabled': False,           # Phase 2 will flip this to True
  }
  ```
- **GOTCHA**: Re-use `flgo.benchmark.fedrag_classification` as the benchmark; **do NOT** create a new benchmark — the LoRA wrapping happens at `get_model` time, transparent to flgo.
- **VALIDATE**: `python FedE/main_lora.py` runs to completion (25 rounds) without crash; final `x-model_*.bin` checkpoint is created.

---

### Task 1.4 — CREATE `FedE/tests/test_lora_filter.py`

- **IMPLEMENT**: Standalone pytest-style script that constructs the model via `config.get_model()`, asserts trainable param count is in `[290_000, 300_000]`, asserts `'lora' in k.lower()` filter selects the same number of state_dict keys as `requires_grad=True` keys, and asserts ship-payload < 2 MB when `torch.save`'d.
- **PATTERN**: Inline pytest-free assertions for portability (Windows dev has no pytest installed by default).
- **IMPORTS**:
  ```
  import io, sys, torch
  sys.path.insert(0, 'FedE')
  from flgo.benchmark.fedrag_classification.config import get_model
  ```
- **GOTCHA**: state_dict includes `lora_A.default.weight` AND `lora_B.default.weight` AND `base_model.model.<layer>.lora_A.default.weight` etc. — the `'lora' in k.lower()` predicate matches all of them. **Verify** by printing the matched keys.
- **VALIDATE**: `python FedE/tests/test_lora_filter.py` exits 0.

---

### Task 2.1 — UPDATE `FedE/flgo/algorithm/fedrag_lora.py` — add DP branch in `Client.train`

- **IMPLEMENT**: Inside `Client.train`, branch on `self.option.get('dp_enabled', False)`. Non-DP path = current Phase-1 logic. DP path mirrors [fedrag_dp.py:447-480](FedE/flgo/algorithm/fedrag_dp.py#L447-L480) but with `params = [p for p in local_model.parameters() if p.requires_grad]` (LoRA-only). σ comes from `self.option['dp_noise_multiplier']` (calibrated by Server in Task 2.3).
- **PATTERN**: [fedrag_dp.py:402-490](FedE/flgo/algorithm/fedrag_dp.py#L402-L490).
- **IMPORTS** (append):
  ```
  from .fedrag_dp import _get_single_sample, _get_batch_size, _compute_grad_norm
  ```
- **PSEUDO-CODE** (mirrors [fedrag_dp.py:447-480](FedE/flgo/algorithm/fedrag_dp.py#L447-L480)):
  ```
  if self.option.get('dp_enabled', False):
      params = [p for p in local_model.parameters() if p.requires_grad]
      clip_norm = self.option.get('dp_clip_norm', 0.1)
      sigma = self.option['dp_noise_multiplier']
      for step in range(self.num_steps):
          batch = self.get_batch_data()
          bs = _get_batch_size(batch)
          if bs == 0: continue
          accumulated = [torch.zeros_like(p.data) for p in params]
          for i in range(bs):
              single = _get_single_sample(batch, i)
              local_model.zero_grad()
              srv = self.calculator.compute_server_loss(model, single)
              loss, _, _ = self.calculator.compute_client_loss(srv, local_model, single)
              loss.backward()
              norm = _compute_grad_norm(params)
              coef = min(1.0, clip_norm / (norm + 1e-8))
              for j, p in enumerate(params):
                  if p.grad is not None:
                      accumulated[j] += p.grad.detach() * coef
          for j, p in enumerate(params):
              noise = torch.randn_like(accumulated[j]) * (sigma * clip_norm)
              p.grad = (accumulated[j] + noise) / float(bs)
          torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)  # post-noise stability, mirrors main_dp_lora_eps20.py:130
          optimizer.step()
  else:
      # Phase-1 non-DP path (unchanged)
      ...
  ```
- **GOTCHA**:
  - `clip_norm=0.1` is the **validated** default from [main_dp_lora_eps20.py:200](FedE/main_dp_lora_eps20.py#L200) — do NOT default to 1.0 (that was the buggy fedrag.py value).
  - The post-aggregation `clip_grad_norm_(params, max_norm=1.0)` at [main_dp_lora_eps20.py:130](FedE/main_dp_lora_eps20.py#L130) is a stability hack — keep it, document it.
  - Per-sample backward over CE+100·MSE loss (the Phase-1 loss) is **slow** (≈ 8 × per batch). Acceptable for the validated baseline; Phase 3 makes it cheaper.
- **VALIDATE**: `python FedE/main_lora.py` with `dp_enabled=True` in the option dict — completes 25 rounds, prints `[Privacy] eps_spent=X.XX / 20.0` per round, final eps ≤ 20.

---

### Task 2.2 — UPDATE `FedE/flgo/algorithm/fedrag_lora.py` — calibrate σ in `Server.__init__`

- **IMPLEMENT**: Override `Server.__init__` (or use the `run` head, mirror [fedrag_dp.py:90-100](FedE/flgo/algorithm/fedrag_dp.py#L90-L100)) to call `find_noise_multiplier(target_epsilon, num_rounds, sampling_rate, target_delta)` and stash σ. Inject σ into the option dict before clients are created (or pass via package).
- **PATTERN**: [main_dp_lora_eps20.py:43-46](FedE/main_dp_lora_eps20.py#L43-L46).
- **IMPORTS**:
  ```
  import sys, os
  sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
  from privacy.rdp_accountant import RDPAccountant, find_noise_multiplier, compute_epsilon
  ```
- **GOTCHA**: flgo creates clients **before** `Server.__init__` runs in some paths; the safest place to inject σ is `Server.run`'s preamble (before the `while True` loop) writing into `self.option` and propagating via `Server.pack`.
- **VALIDATE**: First-round log shows `Calibrated sigma = X.XXXX for eps=20`. Subsequent rounds show monotonically increasing `eps_spent`.

---

### Task 2.3 — UPDATE `FedE/flgo/algorithm/fedrag_lora.py` — RDP accountant in `Server.aggregate`

- **IMPLEMENT**: Instantiate `RDPAccountant(noise_multiplier=σ, sample_rate=q, delta=target_delta)` once in `Server.run`. Call `accountant.step()` and log `accountant.get_epsilon()` after each `aggregate`.
- **PATTERN**: [main_dp_lora_eps20.py:178-209](FedE/main_dp_lora_eps20.py#L178-L209).
- **GOTCHA**: `sample_rate` = `clients_per_round / num_clients`. With the validated baseline (5/5 = 1.0), there is **no** subsampling amplification — σ must be calibrated against `q=1.0`.
- **VALIDATE**: After 25 rounds, log line `Final eps: X.XXXX` matches eps=20 ± 0.1.

---

### Task 3.1 — UPDATE `FedE/flgo/benchmark/fedrag_classification/core.py:128-188`

- **IMPLEMENT**: Modify `TaskCalculator.__init__` to accept `temperature: float = 0.05` and `kd_weight: float = 1.0` from `**kwargs`. Rewrite `compute_client_loss` to apply temperature scaling on InfoNCE logits and switch the distillation term to KL.
- **PATTERN**: Standard temperature-scaled KD (Hinton 2015) — multiply by `τ²` to keep gradient magnitude.
- **IMPORTS**:
  ```
  import torch.nn.functional as F   # already imported as nn? — check core.py:24, add F import
  ```
- **REPLACE** [core.py:155-164](FedE/flgo/benchmark/fedrag_classification/core.py#L155-L164):
  ```
  logits = cos_sim(question_pooled_tensors, reference_pooled_tensors)
  logits = logits.to(self.device)
  label = torch.arange(len(logits)).to(self.device)
  τ = self.temperature
  loss_1 = F.cross_entropy(logits / τ, label)                              # InfoNCE
  loss_2 = F.kl_div(
      F.log_softmax(logits / τ, dim=-1),
      F.softmax(server_logits / τ, dim=-1),
      reduction='batchmean'
  ) * (τ ** 2)                                                              # KD-GLE
  return loss_1 + self.kd_weight * loss_2, loss_1, loss_2
  ```
- **GOTCHA**:
  - `kd_weight=1.0` default is **intentionally different** from the legacy `100×` (which was a magnitude artefact of MSE). If retention regresses by > 0.5 % vs Phase-2 baseline, sweep `kd_weight ∈ {1, 5, 10}` and `learning_rate ∈ {1e-5, 5e-5}`.
  - `temperature=0.05` is the standard for retrieval contrastive (mirrors SimCSE / E5 / BGE training); do not start at 1.0.
  - The `server_logits` argument shape is `(B, B)` (cosine matrix); softmax along `dim=-1` is correct.
- **VALIDATE**:
  ```
  python -c "
  import sys, torch, torch.nn.functional as F
  sys.path.insert(0,'FedE')
  from flgo.benchmark.fedrag_classification.core import TaskCalculator
  # synthetic Q==R sanity: logits ~ I, server_logits ~ I -> InfoNCE -> 0, KL -> 0
  "
  ```
  Plus full DP run: `python FedE/main_lora.py` (with `dp_enabled=True`) → final eps=20 retention within ±0.5 % of Phase 2 baseline.

---

### Task 4.1 — CREATE `FedE/flgo/algorithm/fedrag_lora_ckks.py`

- **IMPLEMENT**: New module with helpers `_chunked_encrypt`, `_chunked_decrypt`, and subclasses `Server(fedrag_lora.Server)` / `Client(fedrag_lora.Client)`.
- **PATTERN**: CKKS plumbing from [fedrag_CKKS.py:106-119](FedE/flgo/algorithm/fedrag_CKKS.py#L106-L119) — but **invert** key holder.
- **IMPORTS**:
  ```
  import tenseal as ts
  import numpy as np, torch
  from . import fedrag_lora
  ```
- **CKKS CONTEXT FACTORY** (extra coeff modulus to allow one extra mul-depth for safety):
  ```
  def _make_ckks_context():
      ctx = ts.context(
          ts.SCHEME_TYPE.CKKS,
          poly_modulus_degree=8192,
          coeff_mod_bit_sizes=[60, 40, 40, 40, 60],
      )
      ctx.global_scale = 2 ** 40
      ctx.generate_galois_keys()
      return ctx
  ```
- **CLIENT.PACK** — divide-by-K *before* encryption to avoid level-cost on the server:
  ```
  def pack(self, model, *args, **kwargs):
      sd = model.state_dict()
      lora_state = {k: v.detach().cpu().float() for k,v in sd.items() if 'lora' in k.lower()}
      K = self.option.get('num_clients', 5)
      cipher_chunks, manifest = _chunked_encrypt(
          {k: v / float(K) for k,v in lora_state.items()},  # pre-divide
          self.public_ctx_for_server,
          chunk_size=4096,
      )
      return {"cipher": cipher_chunks, "manifest": manifest}
  ```
- **SERVER.AGGREGATE** — sum ciphertexts, no decrypt:
  ```
  def aggregate(self, model_old, models, *args, **kwargs):
      assert not self.public_ctx.has_secret_key(), "Server must NOT hold the secret key"
      ciphers = [c["cipher"] for c in models]   # list of list[bytes]
      manifest = models[0]["manifest"]
      summed = []
      for chunk_idx in range(len(ciphers[0])):
          acc = ts.ckks_vector_from(self.public_ctx, ciphers[0][chunk_idx])
          for k in range(1, len(ciphers)):
              acc = acc + ts.ckks_vector_from(self.public_ctx, ciphers[k][chunk_idx])
          summed.append(acc.serialize())
      # Server stores summed ciphertexts; pack() will ship them next round
      self._last_aggregated_cipher = summed
      self._last_manifest = manifest
      return model_old   # base unchanged on server side
  ```
- **CLIENT.UNPACK** — receive aggregated ciphertext (server's pack ships it back):
  ```
  def unpack(self, received_pkg):
      ciphers = received_pkg["cipher"]
      manifest = received_pkg["manifest"]
      lora_state = _chunked_decrypt(ciphers, manifest, self.secret_ctx)
      self.model.load_state_dict(lora_state, strict=False)
      return self.model
  ```
- **GOTCHA**:
  - `_chunked_encrypt` must record `(key, original_shape, num_chunks, last_chunk_len)` for each LoRA tensor; otherwise reshape on decrypt is lossy at chunk boundaries.
  - Slot capacity `= poly_modulus_degree // 2 = 4096`.
  - `ts.ckks_vector + ts.ckks_vector` is supported; `ts.ckks_vector + scalar` is supported and scalar-mul is also supported (used in alternative aggregation strategy).
  - **Key flow** is exactly inverted vs [fedrag_CKKS.py:23-24](FedE/flgo/algorithm/fedrag_CKKS.py#L23-L24): clients hold the secret key, server only the public/eval. The `assert` at the top of `Server.aggregate` is the cryptographic guard.
  - Determinism: TenSEAL adds CKKS noise; the correctness test must use a tolerance ≥ `1e-3` per element.
  - **Bandwidth budget**: 35 chunks × 32 KB ≈ 1.1 MB / client / round. Acceptable; document in PR.
- **VALIDATE**: Run Task 4.2 correctness test; assertion must pass.

---

### Task 4.2 — CREATE `FedE/tests/test_ckks_correctness.py`

- **IMPLEMENT**: Standalone script that (a) initialises 5 fake LoRA states with known random tensors, (b) computes the plaintext mean, (c) runs the same through the encrypt/aggregate/decrypt pipeline, (d) asserts elementwise abs-diff ≤ 1e-3.
- **PATTERN**: TenSEAL tutorial 0 quickstart.
- **IMPORTS**:
  ```
  import torch, tenseal as ts, sys
  sys.path.insert(0,'FedE')
  from flgo.algorithm.fedrag_lora_ckks import _make_ckks_context, _chunked_encrypt, _chunked_decrypt
  ```
- **GOTCHA**: The 1e-3 tolerance is loose because CKKS scale=2^40 + 5-mod chain; **do not** loosen further without offline numerical analysis.
- **VALIDATE**: `python FedE/tests/test_ckks_correctness.py` exits 0 with `Max abs diff: <small>`.

---

### Task 5.1 — UPDATE `FedE/flgo/benchmark/fedrag_classification/config.py` — add qLoRA branch

- **IMPLEMENT**: Accept `quantize: bool = False` kwarg. When `True`, load with `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16)` and call `prepare_model_for_kbit_training` before `get_peft_model`.
- **PATTERN**: HuggingFace Transformers 4-bit recipe (linked in docs section).
- **IMPORTS** (guarded):
  ```
  import platform
  if platform.system() == 'Linux':
      try:
          from transformers import BitsAndBytesConfig
          from peft import prepare_model_for_kbit_training
          _QLORA_AVAILABLE = True
      except ImportError:
          _QLORA_AVAILABLE = False
  else:
      _QLORA_AVAILABLE = False
  ```
- **GOTCHA**: `BertModel.from_pretrained(..., quantization_config=...)` works but PEFT's `prepare_model_for_kbit_training` was originally written for causal LMs; for `BertModel` it just sets `requires_grad=False` on the base — that's exactly what we want but **verify on Vast.ai with a 1-step smoke run**.
- **VALIDATE** (Linux/Vast.ai only):
  ```
  python -c "from flgo.benchmark.fedrag_classification.config import get_model; m=get_model(quantize=True); print(sum(p.numel() for p in m.parameters() if p.requires_grad))"
  # Expect ~295K trainable; total VRAM use ≤ 1/3 of Phase-1
  ```

---

### Task 5.2 — UPDATE `FedE/requirements.txt`

- **IMPLEMENT**: Append:
  ```
  # ── Phase 5 qLoRA (Linux/Vast.ai only) ─────────────────────────────────
  bitsandbytes>=0.43; sys_platform == 'linux'
  peft>=0.7.0
  tenseal>=0.3.14
  ```
- **GOTCHA**: `peft` and `tenseal` are already implicitly required by [main_dp_lora_eps20.py](FedE/main_dp_lora_eps20.py) and [fedrag_CKKS.py](FedE/flgo/algorithm/fedrag_CKKS.py) but not pinned in `requirements.txt` — pin them now.
- **VALIDATE**: `pip install -r FedE/requirements.txt` on Vast.ai succeeds; on Windows succeeds (skipping bitsandbytes line).

---

### Task 6.1 — CREATE `FedE/main_full.py`

- **IMPLEMENT**: Single entrypoint exercising the full pipeline. Auto-disable qLoRA on Windows. Default to `target_epsilon=20, target_delta=1e-5, lora_r=8, temperature=0.05, num_rounds=25, num_clients=5`.
- **PATTERN**: [main_dp.py](FedE/main_dp.py) for option plumbing, [main.py](FedE/main.py) for skeleton.
- **IMPORTS**:
  ```
  import os, platform
  os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
  os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
  import flgo
  import flgo.algorithm.fedrag_lora_ckks as fedrag_lora_ckks
  ```
- **OPTION DICT**:
  ```
  use_qlora = (platform.system() == 'Linux')
  option = {
      'num_rounds': 25, 'num_epochs': 1, 'gpu': 0,
      'batch_size': 16 if use_qlora else 8,
      'learning_rate': 1e-5,
      'dp_enabled': True,
      'target_epsilon': 20.0, 'target_delta': 1e-5,
      'dp_clip_norm': 0.1,
      'use_qlora': use_qlora,
      'temperature': 0.05, 'kd_weight': 1.0,
      'num_clients': 5,
  }
  ```
- **GOTCHA**: If `use_qlora=True` but bitsandbytes import failed in `config.py`, log a warning and force `use_qlora=False` before calling `flgo.init`.
- **VALIDATE**: Vast.ai only — `python FedE/main_full.py` runs to completion; final retention within 2 % of Phase-2 plain baseline; CKKS no-decrypt assertion holds.

---

### Task 6.2 — UPDATE `FedE/eval_compare.py`

- **IMPLEMENT**: Detect PEFT checkpoints (state_dict has `'lora' in any key`); call `model.merge_and_unload()` before comparison. Otherwise leave existing logic intact.
- **PATTERN**: [main_dp_lora_eps20.py:221-223](FedE/main_dp_lora_eps20.py#L221-L223).
- **GOTCHA**: Currently `eval_compare.py:23` reads `pubmed_train.json` directly; do NOT change that path.
- **VALIDATE**: `python FedE/eval_compare.py` runs against any `x-model_lora_merged_*.bin` produced by Phase 6 without exception.

---

### Task 6.3 — UPDATE `FedE/TRAIN_GUIDE.md`

- **IMPLEMENT**: Append a "Pipeline Phase Runbook" section listing the 6 phases, their entrypoints (`main_lora.py`, `main_full.py`), platform gates (Windows vs Vast.ai), and per-phase validation checks (the `VALIDATE` lines from this plan).
- **PATTERN**: Existing `TRAIN_GUIDE.md` style.
- **VALIDATE**: A teammate reading this section can reproduce Phase 1 on Windows and Phase 6 on Vast.ai without re-asking.

---

## TESTING STRATEGY

The project has no formal pytest harness today (no `pytest.ini`, no CI). Tests are inline scripts that exit non-zero on failure — mirror that style.

### Unit Tests

- **`tests/test_lora_filter.py`** (Task 1.4) — confirms LoRA-key filter selects exactly the trainable params.
- **`tests/test_ckks_correctness.py`** (Task 4.2) — confirms encrypted aggregate matches plaintext within 1e-3.
- Loss sanity (inline in Task 3.1 VALIDATE) — synthetic Q==R inputs drive InfoNCE → 0 and KL → 0.

### Integration Tests

- **Phase 1 end-to-end**: `python FedE/main_lora.py` with `dp_enabled=False` for 5 rounds; produces a checkpoint loadable by `eval_compare.py`.
- **Phase 2 regression**: `python FedE/main_lora.py` with `dp_enabled=True, target_epsilon=20`; final retention within ±0.5 % of [main_dp_lora_eps20.py](FedE/main_dp_lora_eps20.py) baseline (98.4 %).
- **Phase 4 cryptographic correctness**: see Task 4.2.
- **Phase 6 full pipeline (Vast.ai only)**: `python FedE/main_full.py`; assert final ε ≤ 20 and `Server.aggregate` `assert` did not fire.

### Edge Cases

- **Empty client batch** (`bs == 0`): per-sample loop must skip without div-by-zero — already handled in [fedrag_dp.py:429-430](FedE/flgo/algorithm/fedrag_dp.py#L429-L430), copy that pattern.
- **Single client per round** (`q = 1/K`): RDP amplification reduces — verify σ is calibrated against the actual `sample_rate` (Task 2.3 GOTCHA).
- **CKKS scale exhaustion**: if any future change adds more multiplications, the 5-mod coeff chain `[60,40,40,40,60]` may run out — Task 4.1 GOTCHA documents the divide-by-K-pre-encrypt mitigation.
- **Windows + qLoRA**: must auto-disable, not crash (Task 6.1 GOTCHA).
- **Mixed-dtype state_dict** (some LoRA layers `float16`, others `float32`): aggregation must `.float()` before averaging then cast back (Task 1.2 `Server.aggregate`).

---

## VALIDATION COMMANDS

Execute every command to ensure zero regressions and 100% feature correctness. Commands assume `cwd = c:\Users\INNOTECH\Documents\SDA_SecureDataAlliance`.

### Level 1: Syntax & Style

PowerShell (Windows dev):
```
python -c "import ast; ast.parse(open('FedE/flgo/algorithm/fedrag_lora.py').read()); print('OK fedrag_lora.py')"
python -c "import ast; ast.parse(open('FedE/flgo/algorithm/fedrag_lora_ckks.py').read()); print('OK fedrag_lora_ckks.py')"
python -c "import ast; ast.parse(open('FedE/main_lora.py').read()); print('OK main_lora.py')"
python -c "import ast; ast.parse(open('FedE/main_full.py').read()); print('OK main_full.py')"
```

### Level 2: Unit Tests

```
python FedE/tests/test_lora_filter.py
python FedE/tests/test_ckks_correctness.py     # Vast.ai only (TenSEAL)
```

### Level 3: Integration Tests

Phase 1 (Windows OK):
```
python FedE/main_lora.py
```
Phase 2 regression (Vast.ai recommended for speed; works on Windows CPU):
```
$env:DP_ENABLED='1'; python FedE/main_lora.py    # ensure main_lora.py reads env or edit option dict
```
Phase 4 (Vast.ai only):
```
python FedE/main_full.py
```

### Level 4: Manual Validation

- Inspect `logs/` for `[Privacy] eps_spent=...` lines per round; final value ≤ 20.0.
- Open the produced `x-model_lora_merged_*.bin`; verify its size (~ full model 437 MB after merge — sanity check the merge worked).
- Run `python FedE/eval_compare.py` against the merged checkpoint; retention vs `main_dp_lora_eps20.py` baseline within ±0.5 % (Phase 2) or ±2 % (Phase 6).
- Confirm Phase 4 server log contains the line `assertion holds: server has no secret key`.

### Level 5: Additional Validation (Optional)

- Vast.ai GPU memory profile via `nvidia-smi --query-gpu=memory.used --format=csv -l 5`: Phase 5 with qLoRA shows ≈ 1/3 of Phase 1 footprint.
- Inspect ciphertext payload size on the wire: `python -c "from FedE.flgo.algorithm.fedrag_lora_ckks import ...; print(total_bytes)"` ≈ 1.1 MB / client / round.

---

## ACCEPTANCE CRITERIA

- [ ] Phase 1: `main_lora.py` runs 25 rounds; payload size per round < 2 MB; trainable param count = 295 K ± 5 K.
- [ ] Phase 2: DP-enabled run reports final ε within `[19.5, 20.0]`; PubMed retention within ±0.5 % of [main_dp_lora_eps20.py](FedE/main_dp_lora_eps20.py) baseline.
- [ ] Phase 3: Synthetic Q==R sanity test reports InfoNCE loss < 1e-4 and KL < 1e-4. Real-data retention not worse than Phase 2 by more than 0.5 %.
- [ ] Phase 4: `tests/test_ckks_correctness.py` passes (max abs diff ≤ 1e-3). `assert not has_secret_key()` fires on every aggregate without raising.
- [ ] Phase 5 (Linux only): `nvidia-smi` shows ≥ 60 % VRAM reduction vs Phase 1 at the same batch size; batch size 16 trains without OOM on a 24 GB GPU.
- [ ] Phase 6: `main_full.py` runs end-to-end on Vast.ai; final ε ≤ 20; retention within 2 % of Phase 2 baseline.
- [ ] Files [main.py](FedE/main.py), [main_dp.py](FedE/main_dp.py), [main_dp_lora_eps20.py](FedE/main_dp_lora_eps20.py), [fedrag_dp.py](FedE/flgo/algorithm/fedrag_dp.py), [fedrag_CKKS.py](FedE/flgo/algorithm/fedrag_CKKS.py), [fedrag.py](FedE/flgo/algorithm/fedrag.py) are byte-identical to their pre-refactor `git show HEAD:` versions.
- [ ] [TRAIN_GUIDE.md](FedE/TRAIN_GUIDE.md) updated with phase runbook.
- [ ] [requirements.txt](FedE/requirements.txt) pins `peft`, `tenseal`, and (Linux-gated) `bitsandbytes`.

---

## COMPLETION CHECKLIST

- [ ] Task 1.1 — config.py LoRA wrap, validation passes
- [ ] Task 1.2 — fedrag_lora.py (Server/Client + pack/unpack/aggregate), 1-round smoke run completes
- [ ] Task 1.3 — main_lora.py runs 25 rounds without DP
- [ ] Task 1.4 — test_lora_filter.py exits 0
- [ ] Task 2.1 — DP branch in Client.train (per-sample clip + noise on LoRA params)
- [ ] Task 2.2 — σ calibration in Server (`find_noise_multiplier`)
- [ ] Task 2.3 — RDPAccountant ticking in Server.aggregate; final ε ≤ 20
- [ ] Phase 2 regression — retention within ±0.5 % of `main_dp_lora_eps20.py` baseline
- [ ] Task 3.1 — InfoNCE + KL loss; synthetic sanity passes; real-data retention preserved
- [ ] Task 4.1 — fedrag_lora_ckks.py with no-decrypt aggregate; assertion fires every round
- [ ] Task 4.2 — test_ckks_correctness.py exits 0 with diff ≤ 1e-3
- [ ] Task 5.1 — config.py qLoRA branch; bitsandbytes import guarded; Linux-only smoke runs
- [ ] Task 5.2 — requirements.txt updated
- [ ] Task 6.1 — main_full.py runs end-to-end on Vast.ai
- [ ] Task 6.2 — eval_compare.py merges PEFT checkpoints transparently
- [ ] Task 6.3 — TRAIN_GUIDE.md runbook updated
- [ ] All 4 baseline scripts unchanged (git diff empty for `main.py`, `main_dp.py`, `main_dp_lora_eps20.py`, `fedrag_dp.py`, `fedrag_CKKS.py`, `fedrag.py`)

---

## NOTES

- **Why split Phase 1 + 2 vs Phase 4 vs Phase 5**: each phase is independently testable and rollback-able. Phase 1+2 reproduces the validated ε=20 baseline through the flgo framework — that is the **lock point** before introducing loss changes (Phase 3) or homomorphic encryption (Phase 4).
- **Why Phase 3 lands AFTER Phase 2**: changing the loss function changes effective LR; without the regression baseline locked, we cannot tell whether retention drift comes from the loss or from the framework move.
- **Why the CKKS server holds *only* the public/eval key**: this is the architectural bug fix vs [fedrag_CKKS.py:23-24](FedE/flgo/algorithm/fedrag_CKKS.py#L23-L24). With the inverted key flow, an honest-but-curious server **cannot** read individual client gradients — only the aggregated sum, and only after a client decrypts. Combined with DP noise injected before encryption, this delivers all three protection layers.
- **Why pre-divide by K on the client**: avoids consuming a CKKS multiplication level on the server's plaintext-mul `*(1/K)` step, which would otherwise force a deeper modulus chain (more ciphertext expansion).
- **Why qLoRA is gated on Linux**: bitsandbytes wheels exist for Windows but are flaky and require specific CUDA toolkit versions; the project's primary dev path is Windows + Vast.ai for training. Forcing qLoRA on Windows would block local iteration without scientific benefit.
- **Future work** (out of scope): client subsampling for amplification (would require server to decide who participates each round and broadcast σ accordingly); secure aggregation with verifiable computation (Pinocchio / Groth16) layered on top of CKKS.

**Confidence Score**: **7/10** — Phases 1, 2, 3, 5, 6 are mechanical refactors of existing validated patterns; Phase 4 (true homomorphic aggregation with the inverted key flow) is research-grade and has the highest residual risk: TenSEAL's API for `ts.ckks_vector_from(ctx, bytes)` arithmetic on arrays of ciphertexts must be re-validated on the actual Vast.ai TenSEAL version (we cite tutorial-level guarantees, not project-locked behaviour). Recommend running Task 4.2 (the correctness test) **before** Task 4.1 (production code) — write the test first, prove the math holds in TenSEAL, then build the algorithm class around the proven primitive.

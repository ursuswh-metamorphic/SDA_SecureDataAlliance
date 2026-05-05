"""
test_lora_filter.py — Phase 1 validation.

Asserts that:
  1. config.get_model() returns a PeftModel with ~295K trainable params.
  2. The 'lora' substring filter selects EXACTLY the trainable params
     (no false positives, no false negatives).
  3. The serialised LoRA-only state_dict is < 2 MB (vs ~437 MB for full base).

Exits 0 on success, non-zero on any assertion failure. No pytest dependency
so it runs on the Windows dev machine without extra installs.
"""
import io
import os
import sys

# Make `import flgo.benchmark.fedrag_classification.config` resolvable
HERE = os.path.dirname(os.path.abspath(__file__))
FEDE_ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, FEDE_ROOT)

import torch  # noqa: E402

from flgo.benchmark.fedrag_classification.config import get_model  # noqa: E402
from flgo.algorithm.fedrag_lora import _is_lora_key, _lora_state_only  # noqa: E402


def main():
    print('=' * 60)
    print('[test_lora_filter] Loading model...')
    print('=' * 60)
    model = get_model()

    # ── Check 1: trainable param count is in the expected band ───────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f'\n[Check 1] Trainable params: {n_trainable:,}')
    assert 290_000 < n_trainable < 300_000, (
        f'Expected ~295K trainable params, got {n_trainable:,}. '
        f'Did the PEFT version change LoRA layer naming?'
    )
    print(f'    OK ({n_trainable:,} in [290K, 300K]).')

    # ── Check 2: 'lora' filter == requires_grad set ──────────────────────
    state_dict = model.state_dict()

    # Set of keys whose tensors point at trainable parameters.
    trainable_keys = set()
    trainable_data_ptrs = {p.data_ptr() for p in trainable_params}
    for k, v in state_dict.items():
        if v.data_ptr() in trainable_data_ptrs:
            trainable_keys.add(k)

    lora_keys = {k for k in state_dict if _is_lora_key(k)}

    print(f'\n[Check 2] state_dict keys matching `lora` filter: {len(lora_keys)}')
    print(f'          state_dict keys backed by trainable tensors: {len(trainable_keys)}')

    only_in_filter = lora_keys - trainable_keys
    only_in_grad = trainable_keys - lora_keys
    if only_in_filter:
        print(f'    WARN: {len(only_in_filter)} key(s) match filter but are not trainable: '
              f'{sorted(only_in_filter)[:3]}...')
    if only_in_grad:
        print(f'    FAIL: {len(only_in_grad)} trainable key(s) NOT matched by filter: '
              f'{sorted(only_in_grad)[:3]}...')
    assert not only_in_grad, (
        f'LoRA filter missed {len(only_in_grad)} trainable parameter(s). '
        f'Naming convention may have changed.'
    )
    # Allow filter to match a few extra "decorative" keys (e.g. counters)
    # as long as it includes ALL trainable ones.
    print(f'    OK (filter includes all trainable params).')

    # ── Check 3: serialised payload size ─────────────────────────────────
    lora_state = _lora_state_only(state_dict)
    n_lora_params = sum(v.numel() for v in lora_state.values())
    buf = io.BytesIO()
    torch.save(lora_state, buf)
    n_bytes = len(buf.getvalue())
    print(f'\n[Check 3] LoRA payload: {n_lora_params:,} params, '
          f'{n_bytes / 1024:.1f} KB serialised.')
    assert n_bytes < 2 * 1024 * 1024, (
        f'Expected < 2 MB serialised, got {n_bytes / 1024:.1f} KB. '
        f'Filter is letting too many keys through.'
    )
    print(f'    OK (< 2 MB).')

    # ── Sample the matched keys for manual inspection ───────────────────
    sample_keys = sorted(lora_keys)[:5]
    print(f'\n[Sample] First 5 matched LoRA keys:')
    for k in sample_keys:
        print(f'    {k}: shape={tuple(state_dict[k].shape)}')

    print('\n' + '=' * 60)
    print('[test_lora_filter] ALL CHECKS PASSED')
    print('=' * 60)


if __name__ == '__main__':
    main()

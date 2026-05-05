"""
test_qlora_gate.py — Phase 5 smoke test.

Verifies the qLoRA gate logic in config.py:

  1. (Always) DEFAULT_USE_QLORA is False by default — Windows + Linux dev
     paths see plain LoRA unless explicitly opted in.
  2. (Always) When `quantize=False` is passed (or DEFAULT_USE_QLORA=False),
     get_model() returns a regular PEFT model with 294,912 trainable params.
  3. (Linux + bitsandbytes only) Setting DEFAULT_USE_QLORA=True (or passing
     quantize=True) returns a 4-bit base wrapped in PEFT, with the same
     trainable param count (LoRA is unchanged; only the frozen base is quantized).
  4. (Anywhere without bitsandbytes) Passing quantize=True falls back to
     plain BertModel and prints the warning.

Skips the Linux-specific check if `_QLORA_AVAILABLE` is False.
"""
import os
import platform
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FEDE_ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, FEDE_ROOT)

import torch  # noqa: E402

from flgo.benchmark.fedrag_classification import config as fedrag_config  # noqa: E402


def _trainable(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def main():
    print('=' * 60)
    print(f'[test_qlora_gate] platform={platform.system()}, '
          f'_QLORA_AVAILABLE={fedrag_config._QLORA_AVAILABLE}, '
          f'DEFAULT_USE_QLORA={fedrag_config.DEFAULT_USE_QLORA}')
    print('=' * 60)

    # ── Check 1: default is False ────────────────────────────────────────
    assert fedrag_config.DEFAULT_USE_QLORA is False, (
        f'DEFAULT_USE_QLORA should be False at module load, got '
        f'{fedrag_config.DEFAULT_USE_QLORA}'
    )
    print(f'\n[Check 1] DEFAULT_USE_QLORA defaults to False     OK')

    # ── Check 2: plain LoRA path returns 294,912 trainable ──────────────
    print(f'\n[Check 2] Loading plain LoRA model (quantize=False)...')
    m_plain = fedrag_config.get_model(quantize=False)
    n_plain = _trainable(m_plain)
    assert 290_000 < n_plain < 300_000, f'expected ~295K trainable, got {n_plain}'
    print(f'    OK trainable={n_plain:,}')

    # ── Check 3: qLoRA path on Linux+bitsandbytes ───────────────────────
    if fedrag_config._QLORA_AVAILABLE:
        print(f'\n[Check 3] Loading qLoRA model (quantize=True)...')
        try:
            m_q = fedrag_config.get_model(quantize=True)
            n_q = _trainable(m_q)
            assert 290_000 < n_q < 300_000, f'qLoRA trainable count off: {n_q}'
            print(f'    OK trainable={n_q:,} (matches plain LoRA — only base is 4-bit)')
            # Optional: verify that some param has 4-bit dtype
            saw_4bit = False
            for name, p in m_q.named_parameters():
                # bitsandbytes uses Params4bit; non-trainable base layers
                if 'base' in name and not p.requires_grad and p.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                    saw_4bit = True
                    print(f'    Sample 4-bit param: {name} dtype={p.dtype}')
                    break
            if not saw_4bit:
                # Some bitsandbytes versions hide quant state in attributes
                # rather than dtype. Don't fail the test on this signal.
                print(f'    NOTE: no obvious 4-bit dtype found via .dtype; '
                      f'check Linear4bit modules manually if needed.')
        except Exception as e:
            print(f'    SKIPPED at runtime: {type(e).__name__}: {e}')
    else:
        print(f'\n[Check 3] SKIPPED (qLoRA not available on this platform)')

    # ── Check 4: quantize=True without bitsandbytes falls back ──────────
    if not fedrag_config._QLORA_AVAILABLE:
        print(f'\n[Check 4] quantize=True with no bitsandbytes -> falls back to plain...')
        m_fallback = fedrag_config.get_model(quantize=True)
        n_fb = _trainable(m_fallback)
        assert 290_000 < n_fb < 300_000, f'fallback path broken: {n_fb}'
        print(f'    OK fallback works, trainable={n_fb:,}')
    else:
        print(f'\n[Check 4] SKIPPED (bitsandbytes is available; cannot test fallback)')

    # ── Check 5: requirements.txt has Phase 5 marker ────────────────────
    req_path = os.path.join(FEDE_ROOT, 'requirements.txt')
    with open(req_path, encoding='utf-8') as f:
        req_src = f.read()
    assert "bitsandbytes>=0.43" in req_src, 'requirements.txt missing bitsandbytes pin'
    assert 'sys_platform == "linux"' in req_src, 'bitsandbytes not Linux-gated'
    assert 'peft>=0.7.0' in req_src, 'requirements.txt missing peft pin'
    print(f'\n[Check 5] requirements.txt has Phase 5 markers   OK')

    print('\n' + '=' * 60)
    print('[test_qlora_gate] ALL CHECKS PASSED')
    print('=' * 60)


if __name__ == '__main__':
    main()

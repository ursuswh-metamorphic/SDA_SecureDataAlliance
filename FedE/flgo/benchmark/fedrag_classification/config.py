"""
train_data (torch.utils.data.Dataset),
test_data (torch.utils.data.Dataset),
and the model (torch.nn.Module) should be implemented here.

Phase 1: BGE-base wrapped with PEFT LoRA adapters.
  - Base BGE-base weights are FROZEN (requires_grad=False).
  - Only LoRA-A and LoRA-B matrices on `query` and `value` modules are trainable.
  - Trainable params: ~295K out of 109M (~0.27%).

Phase 5 hook: pass quantize=True to load base in 4-bit (Linux/Vast.ai only).
"""
import platform

import torch.nn
from transformers import BertModel
from peft import LoraConfig, get_peft_model

# Phase 5: optional 4-bit quantization (gated on Linux + working bitsandbytes)
_QLORA_AVAILABLE = False
if platform.system() == 'Linux':
    try:
        from transformers import BitsAndBytesConfig  # noqa: F401
        from peft import prepare_model_for_kbit_training  # noqa: F401
        _QLORA_AVAILABLE = True
    except ImportError:
        _QLORA_AVAILABLE = False

# ── LoRA hyperparameters (mirror main_dp_lora_eps20.py:35-38) ────────────────
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGETS = ['query', 'value']

# ── Phase 5 default for qLoRA gate ───────────────────────────────────────────
# Module-level so main_lora.py / main_full.py can flip it BEFORE flgo.init()
# instantiates the model. fedllm.py:7 calls get_model() with no kwargs, so the
# default reads from this constant.
DEFAULT_USE_QLORA = False

train_data = None
val_data = None
test_data = None
vocab = None
tokenizer = None


def get_model(*args, **kwargs) -> torch.nn.Module:
    """Load BGE-base wrapped with LoRA adapters; base weights are frozen.

    Args:
        quantize (bool): If True and Linux+bitsandbytes available, load base in 4-bit NF4.
            When omitted, falls back to module-level DEFAULT_USE_QLORA so entrypoints
            can flip qLoRA on without modifying fedllm.py's get_model() call site.

    Returns:
        peft.PeftModel whose .forward forwards kwargs to BertModel.forward.
    """
    quantize = kwargs.get('quantize', DEFAULT_USE_QLORA)

    # Phase 5 branch (gated)
    if quantize and _QLORA_AVAILABLE:
        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base = BertModel.from_pretrained(
            'BAAI/bge-base-en',
            quantization_config=bnb_config,
        )
        base = prepare_model_for_kbit_training(base)
    else:
        if quantize and not _QLORA_AVAILABLE:
            print(f'[config.get_model] quantize=True ignored: bitsandbytes unavailable on {platform.system()}.')
        base = BertModel.from_pretrained('BAAI/bge-base-en')

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGETS,
        lora_dropout=LORA_DROPOUT,
        bias='none',
    )
    model = get_peft_model(base, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f'[config.get_model] LoRA: {trainable:,} trainable / {total:,} total ({trainable/total*100:.2f}%)')

    return model

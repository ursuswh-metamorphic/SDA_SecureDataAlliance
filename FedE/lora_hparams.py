"""Pure helpers for resolving tunable LoRA hyperparameters."""


def resolve_lora_hparams(env):
    """Resolve and validate LoRA/optimizer values from an env-like mapping."""
    rank = int(env.get("LORA_R", "8"))
    alpha = int(env.get("LORA_ALPHA", str(2 * rank)))
    dropout = float(env.get("LORA_DROPOUT", "0.05"))
    learning_rate = float(env.get("LEARNING_RATE", "1e-5"))

    if rank <= 0:
        raise ValueError("LORA_R must be positive")
    if alpha <= 0:
        raise ValueError("LORA_ALPHA must be positive")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("LORA_DROPOUT must be in [0, 1)")
    if learning_rate <= 0.0:
        raise ValueError("LEARNING_RATE must be positive")

    return {
        "rank": rank,
        "alpha": alpha,
        "dropout": dropout,
        "learning_rate": learning_rate,
    }

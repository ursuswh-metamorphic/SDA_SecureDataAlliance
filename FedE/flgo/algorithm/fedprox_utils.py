"""Lightweight FedProx helpers with no PEFT/Transformers dependency."""
import torch


def proximal_penalty(model, reference):
    """Return 1/2 ||w - w_global||² over trainable named parameters."""
    terms = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if name not in reference:
                raise KeyError(f"missing FedProx reference parameter: {name}")
            terms.append((parameter - reference[name]).pow(2).sum())
    if not terms:
        raise ValueError("FedProx requires at least one trainable parameter")
    return 0.5 * torch.stack(terms).sum()

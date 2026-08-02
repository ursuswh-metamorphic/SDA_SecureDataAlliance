import pytest
import torch

from flgo.algorithm.fedprox_utils import proximal_penalty


def test_proximal_penalty_is_zero_at_global_and_has_expected_value():
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 2.0]]))
    reference = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    assert proximal_penalty(model, reference).item() == 0.0

    with torch.no_grad():
        model.weight.add_(torch.tensor([[3.0, 4.0]]))
    assert proximal_penalty(model, reference).item() == pytest.approx(12.5)


def test_proximal_penalty_ignores_frozen_parameters():
    model = torch.nn.Linear(2, 1)
    model.bias.requires_grad = False
    reference = {"weight": model.weight.detach().clone()}
    with torch.no_grad():
        model.bias.add_(100)
    assert proximal_penalty(model, reference).item() == 0.0


def test_proximal_penalty_fails_on_missing_reference():
    model = torch.nn.Linear(2, 1, bias=False)
    with pytest.raises(KeyError, match="missing FedProx reference"):
        proximal_penalty(model, {})

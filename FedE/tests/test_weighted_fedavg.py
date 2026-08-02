"""Numerical regression test for sample-weighted LoRA FedAvg."""

from types import SimpleNamespace

import torch

from flgo.algorithm.fedrag_lora import Server


class ToyModel(torch.nn.Module):
    def __init__(self, lora_value, base_value):
        super().__init__()
        self.lora_weight = torch.nn.Parameter(
            torch.tensor([float(lora_value)])
        )
        self.base_weight = torch.nn.Parameter(
            torch.tensor([float(base_value)])
        )


def test_weighted_fedavg_uses_flgo_datavol_and_preserves_base():
    server = Server.__new__(Server)
    server.clients = [
        SimpleNamespace(datavol=1),
        SimpleNamespace(datavol=3),
    ]
    server.received_clients = [0, 1]
    server._rdp_accountant = None

    global_model = ToyModel(lora_value=0, base_value=99)
    client_models = [
        ToyModel(lora_value=1, base_value=-1),
        ToyModel(lora_value=3, base_value=-2),
    ]

    result = Server.aggregate(server, global_model, client_models)

    # (1 * 1 + 3 * 3) / (1 + 3) = 2.5
    assert torch.allclose(
        result.lora_weight.detach(), torch.tensor([2.5])
    )
    assert torch.allclose(
        result.base_weight.detach(), torch.tensor([99.0])
    )

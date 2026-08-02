from types import SimpleNamespace

import pytest
import torch

from flgo.algorithm.fedrag_client_dp import Server


class ToyModel(torch.nn.Module):
    def __init__(self, lora_value, base_value=99.0):
        super().__init__()
        self.lora_weight = torch.nn.Parameter(
            torch.tensor([float(lora_value)])
        )
        self.base_weight = torch.nn.Parameter(
            torch.tensor([float(base_value)])
        )


class FakeAccountant:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1

    def get_epsilon(self):
        return float(self.steps)


def make_server(mode):
    server = Server.__new__(Server)
    server.option = {"client_dp_target_delta": 1e-5}
    server.clients = [
        SimpleNamespace(datavol=1),
        SimpleNamespace(datavol=3),
    ]
    server.received_clients = [0, 1]
    server._client_dp_initialized = True
    server.client_dp_mode = mode
    server.client_dp_clip_norm = 1.0
    server._client_dp_release_count = 0
    server._client_dp_round_reports = []
    server._rdp_accountant = None
    server._dp_sigma = None
    server._client_dp_noise_generator = None
    return server


def test_noise0_matches_weighted_fedavg_and_preserves_base():
    server = make_server("noise0")
    global_model = ToyModel(0)
    result = server.aggregate(
        global_model,
        [ToyModel(1, -1), ToyModel(3, -2)],
    )
    assert torch.allclose(result.lora_weight.detach(), torch.tensor([2.5]))
    assert torch.allclose(result.base_weight.detach(), torch.tensor([99.0]))
    assert server._client_dp_release_count == 0


def test_clip_only_clips_whole_client_delta_once():
    server = make_server("clip_only")
    result = server.aggregate(
        ToyModel(0),
        [ToyModel(3), ToyModel(4)],
    )
    # Scalar positive deltas 3 and 4 both clip to +1 before weighting.
    assert torch.allclose(result.lora_weight.detach(), torch.tensor([1.0]))
    report = server._client_dp_round_reports[-1]
    assert report["raw_update_norms"] == pytest.approx([3.0, 4.0])
    assert report["clip_coefficients"] == pytest.approx([1 / 3, 1 / 4])
    assert report["epsilon_spent"] is None


def test_dp_adds_one_noise_release_and_one_accountant_step():
    server = make_server("dp")
    accountant = FakeAccountant()
    server._rdp_accountant = accountant
    server._dp_sigma = 1.0
    server._client_dp_noise_generator = torch.Generator().manual_seed(7)

    result = server.aggregate(
        ToyModel(0),
        [ToyModel(3), ToyModel(4)],
    )
    report = server._client_dp_round_reports[-1]
    assert accountant.steps == 1
    assert server._client_dp_release_count == 1
    assert report["sensitivity"] == pytest.approx(1.5)
    assert report["noise_std"] == pytest.approx(1.5)
    assert report["epsilon_spent"] == pytest.approx(1.0)
    assert not torch.allclose(result.lora_weight.detach(), torch.tensor([1.0]))
    assert torch.allclose(result.base_weight.detach(), torch.tensor([99.0]))

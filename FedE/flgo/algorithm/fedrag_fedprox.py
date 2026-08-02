"""FedProx variant of the validated FedRAG/FedAvg training path.

Aggregation, LoRA transport, checkpointing, and optional DP remain inherited
from ``fedrag_lora``.  The only algorithmic change is the standard FedProx
local objective:

    L_local(w) + (mu / 2) * ||w - w_global||^2

Only trainable parameters participate, so the same implementation supports
both full fine-tuning and LoRA.
"""
from .fedrag_lora import Client as _FedAvgClient
from .fedrag_lora import Server as _FedAvgServer
from .fedprox_utils import proximal_penalty

import torch


class Server(_FedAvgServer):
    """FedProx uses the same datavol-weighted server aggregation as B3."""


class Client(_FedAvgClient):
    def _train_plain(self, model, local_model, optimizer):
        mu = float(self.option.get("fedprox_mu", 0.01))
        if mu < 0:
            raise ValueError(f"fedprox_mu must be non-negative, got {mu}")

        params = [p for p in local_model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params, lr=self.learning_rate, weight_decay=0.01
        )
        reference = {
            name: parameter.detach().clone()
            for name, parameter in local_model.named_parameters()
            if parameter.requires_grad
        }
        print(
            f"[fedrag_fedprox.Client] mu={mu}, "
            f"proximal_params={len(reference)}"
        )

        for step in range(self.num_steps):
            batch_data = self.get_batch_data()
            local_model.zero_grad()
            server_loss = self.calculator.compute_server_loss(model, batch_data)
            task_loss, client_only, server_only = (
                self.calculator.compute_client_loss(
                    server_loss, local_model, batch_data
                )
            )
            prox = proximal_penalty(local_model, reference)
            loss = task_loss + mu * prox
            print(
                f"client running:{step}/{self.num_steps}, loss:{loss}, "
                f"task:{task_loss}, prox:{prox}, mu:{mu}, "
                f"loss 1:{client_only}, loss 2:{server_only}"
            )
            loss.backward()
            optimizer.step()

            if step == self.num_steps - 1:
                print(f"server loss: {server_loss}")

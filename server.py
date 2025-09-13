"""Server-side strategy and models for the FCFM‑ALM demo.

Defines a discriminator over client embeddings and a tiny MLP (LambdaPolicy)
that maps client bias to an adaptive fairness weight λ. The custom strategy
collects client metrics, updates both networks, and injects λ into the next
round's fit config for each selected client.
"""
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from collections import OrderedDict
from typing import Dict, List, Tuple
from copy import deepcopy

class LambdaPolicy(nn.Module):
    """Small MLP producing λ in (0,1) from a scalar bias."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, b: torch.Tensor) -> torch.Tensor:
        return self.net(b.unsqueeze(-1))

class Discriminator(nn.Module):
    """Predicts fairness (1) vs bias (0) from a client embedding."""
    def __init__(self, embed_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FCFMStrategy(fl.server.strategy.FedAvg):
    """FedAvg strategy extended with fairness feedback and λ scheduling."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.device = torch.device("cpu")
        self.discriminator = Discriminator().to(self.device)
        self.disc_optim = optim.Adam(self.discriminator.parameters(), lr=1e-3)
        self.lambda_policy = LambdaPolicy().to(self.device)
        self.lambda_opt = optim.Adam(self.lambda_policy.parameters(), lr=1e-3)
        self.client_state: Dict[str, Dict] = {}

    def aggregate_fit(self,
                      rnd: int,
                      results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.server.client_proxy.FitRes]],
                      failures: List[BaseException]
                      ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Aggregate results, update models, and report server metrics."""

        params, metrics = super().aggregate_fit(rnd, results, failures)

        if len(results) == 0:
            return params, metrics if isinstance(metrics, dict) else {}

        biases = []
        ents   = []
        embeds = []
        lambdas = []

        raw_biases = []
        for (client, fit_res) in results:
            m = fit_res.metrics
            biases.append(float(m.get("bias_noisy", m.get("bias", 0.0))))
            ents.append(float(m["uncert"]))
            embeds.append(np.array(json.loads(m["embed"]), dtype=np.float32))
            lambdas.append(float(m.get("lambda", 0.0)))
            if "bias_raw" in m:
                try:
                    raw_biases.append(float(m["bias_raw"]))
                except Exception:
                    pass

        mean_bias = float(np.mean(biases)) if len(biases) > 0 else 0.0
        msg = f"[Server] Round {rnd}: mean bias noised = {mean_bias:.4f}"
        if len(raw_biases) > 0:
            mean_bias_raw = float(np.mean(raw_biases))
            msg += f" | raw = {mean_bias_raw:.4f}"
        print(msg)

        emb_tensor = torch.tensor(np.stack(embeds), dtype=torch.float32)
        bias = torch.tensor(biases, dtype=torch.float32)
        fair_targets = (bias < 0.1).float().unsqueeze(-1)
        disc_pred = self.discriminator(emb_tensor)
        disc_loss = nn.BCELoss()(disc_pred, fair_targets)

        self.disc_optim.zero_grad()
        disc_loss.backward()
        self.disc_optim.step()

        target_lam = (1.0 / (1.0 + bias.abs())).unsqueeze(-1)
        pred_lam   = self.lambda_policy(bias)
        lam_loss   = nn.MSELoss()(pred_lam, target_lam)

        self.lambda_opt.zero_grad()
        lam_loss.backward()
        self.lambda_opt.step()

        lam_pred_all = self.lambda_policy(bias).detach().cpu().numpy().squeeze()

        # store last seen state per client id
        for (client, fit_res), b, u, e, l in zip(results, biases, ents, embeds, lambdas):
            cid = getattr(client, "cid", None)
            if cid is None:
                continue
            self.client_state[cid] = {
                "bias": float(b),
                "uncert": float(u),
                "embed": np.asarray(e, dtype=np.float32),
                "lambda": float(l),
            }

        print(f"[Server] Round {rnd}: Discriminator loss={disc_loss.item():.4f}, "
              f"Lambda loss={lam_loss.item():.4f}")

        out_metrics = {"lambda_policy_loss": lam_loss.item(),
                        "disc_loss": disc_loss.item(),
                        "mean_bias_noisy": mean_bias}
        if len(raw_biases) > 0:
            out_metrics["mean_bias_raw"] = float(np.mean(raw_biases))
        return params, out_metrics

    def configure_fit(self,
                      server_round: int,
                      parameters: fl.common.Parameters,
                      client_manager: fl.server.client_manager.ClientManager):
        """Attach the current λ estimate to each client's fit config."""
        cfg = super().configure_fit(server_round, parameters, client_manager)
        out = []
        for client, fit_ins in cfg:
            cid = getattr(client, "cid", None)
            last_bias = 0.0
            if cid is not None and cid in self.client_state:
                last_bias = float(self.client_state[cid].get("bias", 0.0))
            with torch.no_grad():
                lam = float(self.lambda_policy(torch.tensor([last_bias], device=self.device)).item())
            new_conf = dict(fit_ins.config)
            new_conf["lambda"] = lam
            out.append((client, fl.common.FitIns(fit_ins.parameters, new_conf)))
        return out

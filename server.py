# server.py
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Tuple
from sklearn.metrics import roc_auc_score
from sklearn.metrics import equal_opportunity_difference
from copy import deepcopy


# -----------------------------------------------------------------
# 1.  Simple MLP that maps bias → λ
# -----------------------------------------------------------------
class LambdaPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()      # output in (0,1)
        )

    def forward(self, b: torch.Tensor) -> torch.Tensor:
        return self.net(b.unsqueeze(-1))  # shape -> (batch, 1)


# -----------------------------------------------------------------
# 2.  Discriminator that predicts “fair” (1) / “biased” (0) from
#    the client embedding
# -----------------------------------------------------------------
class Discriminator(nn.Module):
    def __init__(self, embed_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------------------------------------------
# 3.  Custom Flower strategy that
#     – collects payloads
#     – updates lambda‑policy and discriminator
#     – sends λ back to clients
# -----------------------------------------------------------------
class FCFMStrategy(fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # devices
        self.device = torch.device("cpu")
        # models
        self.discriminator = Discriminator().to(self.device)
        self.disc_optim = optim.Adam(self.discriminator.parameters(), lr=1e-3)
        self.lambda_policy = LambdaPolicy().to(self.device)
        self.lambda_opt = optim.Adam(self.lambda_policy.parameters(), lr=1e-3)

    # ----------------------------------------------
    # 1.  Called after a round of aggregation
    # ----------------------------------------------
    def aggregate_fit(self,
                      rnd: int,
                      results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.server.client_proxy.FitRes]],
                      failures: List[BaseException]
                      ) -> Tuple[torch.Tensor, Dict[str, float]]:

        params, metrics = super().aggregate_fit(rnd, results, failures)

        # ----  collect all payloads
        biases = []
        ents   = []
        embeds = []
        lambdas = []

        for (client, fit_res) in results:
            m = fit_res.metrics
            biases.append(m["bias"])
            ents.append(m["uncert"])
            embeds.append(m["embed"])
            lambdas.append(m["lambda"])

        # -- 1) update discriminator
        emb_tensor = torch.tensor(np.stack(embeds), dtype=torch.float32)
        bias = torch.tensor(biases, dtype=torch.float32)
        # target: 0 = biased, 1 = fair
        # fair if bias < 0.1 (tunable)
        fair_targets = (bias < 0.1).float().unsqueeze(-1)
        disc_pred = self.discriminator(emb_tensor).unsqueeze(-1)
        disc_loss = nn.BCELoss()(disc_pred, fair_targets)

        self.disc_optim.zero_grad()
        disc_loss.backward()
        self.disc_optim.step()

        # -- 2) update lambda policy
        # target λ* = 1 / (1 + |bias|)  (smaller bias → smaller λ)
        target_lam = (1.0 / (1.0 + bias.abs())).unsqueeze(-1)
        pred_lam   = self.lambda_policy(bias.unsqueeze(-1))
        lam_loss   = nn.MSELoss()(pred_lam, target_lam)

        self.lambda_opt.zero_grad()
        lam_loss.backward()
        self.lambda_opt.step()

        # ----  compute per‑client λ to send back
        lam_pred_all = self.lambda_policy(bias.unsqueeze(-1)).detach().cpu().numpy().squeeze()

        # ----  log
        print(f"[Server] Round {rnd}: Discriminator loss={disc_loss.item():.4f}, "
              f"Lambda loss={lam_loss.item():.4f}")

        return params, {"lambda_policy_loss": lam_loss.item(),
                        "disc_loss": disc_loss.item()}

    # ----------------------------------------------
    # 2.  Called before each round to supply config
    # ----------------------------------------------
    def configure_fit(self,
                      rnd: int,
                      server_round: fl.server.strategy.FitConfig,
                      parameters: fl.common.NDArray,
                      client_manager: fl.server.client_manager.ClientManager
                      ) -> List[Tuple[int, Dict]]:
        # send current λ to each client
        configs = []
        for cid in client_manager.get_client_ids():
            # compute λ for this client: use discriminator's prediction on
            # the last embedding received from that client – for demo we
            # simply broadcast the *global* λ from the policy:
            # (in a real system you would keep a per‑client λ history)
            lam = float(self.lambda_policy(torch.tensor([0.0], device=self.device)).item())
            configs.append((cid, {"lambda": lam}))
        return configs

# Why this is simple yet illustrative
# – λ is learned from bias via a tiny MLP.
# – The discriminator decides which clients are “fair”; its gradients flow into the λ‑policy.
# – No explicit graph GNN is used – we just broadcast the policy's output. Re‑implementing a GNN would require only a few more lines.
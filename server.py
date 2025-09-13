# server.py
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from collections import OrderedDict
from typing import Dict, List, Tuple
# Note: Metrics imports removed; not used and can cause import issues.
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
        # track last per-client stats
        self.client_state: Dict[str, Dict] = {}

    # ----------------------------------------------
    # 1.  Called after a round of aggregation
    # ----------------------------------------------
    def aggregate_fit(self,
                      rnd: int,
                      results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.server.client_proxy.FitRes]],
                      failures: List[BaseException]
                      ) -> Tuple[torch.Tensor, Dict[str, float]]:

        params, metrics = super().aggregate_fit(rnd, results, failures)

        # If no successful client results, skip custom updates
        if len(results) == 0:
            return params, metrics if isinstance(metrics, dict) else {}

        # ----  collect all payloads
        biases = []
        ents   = []
        embeds = []
        lambdas = []

        raw_biases = []
        for (client, fit_res) in results:
            m = fit_res.metrics
            # Metrics come back as scalars/JSON strings
            biases.append(float(m.get("bias_noisy", m.get("bias", 0.0))))
            ents.append(float(m["uncert"]))
            embeds.append(np.array(json.loads(m["embed"]), dtype=np.float32))
            lambdas.append(float(m.get("lambda", 0.0)))
            if "bias_raw" in m:
                try:
                    raw_biases.append(float(m["bias_raw"]))
                except Exception:
                    pass

        # ----  compute mean bias and log it
        mean_bias = float(np.mean(biases)) if len(biases) > 0 else 0.0
        msg = f"[Server] Round {rnd}: mean bias noised = {mean_bias:.4f}"
        if len(raw_biases) > 0:
            mean_bias_raw = float(np.mean(raw_biases))
            msg += f" | raw = {mean_bias_raw:.4f}"
        print(msg)

        # -- 1) update discriminator
        emb_tensor = torch.tensor(np.stack(embeds), dtype=torch.float32)
        bias = torch.tensor(biases, dtype=torch.float32)
        # target: 0 = biased, 1 = fair
        # fair if bias < 0.1 (tunable)
        fair_targets = (bias < 0.1).float().unsqueeze(-1)
        disc_pred = self.discriminator(emb_tensor)
        disc_loss = nn.BCELoss()(disc_pred, fair_targets)

        self.disc_optim.zero_grad()
        disc_loss.backward()
        self.disc_optim.step()

        # -- 2) update lambda policy
        # target λ* = 1 / (1 + |bias|)  (smaller bias → smaller λ)
        target_lam = (1.0 / (1.0 + bias.abs())).unsqueeze(-1)
        pred_lam   = self.lambda_policy(bias)
        lam_loss   = nn.MSELoss()(pred_lam, target_lam)

        self.lambda_opt.zero_grad()
        lam_loss.backward()
        self.lambda_opt.step()

        # ----  compute per‑client λ to send back
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

        # ----  log
        print(f"[Server] Round {rnd}: Discriminator loss={disc_loss.item():.4f}, "
              f"Lambda loss={lam_loss.item():.4f}")

        out_metrics = {"lambda_policy_loss": lam_loss.item(),
                        "disc_loss": disc_loss.item(),
                        "mean_bias_noisy": mean_bias}
        if len(raw_biases) > 0:
            out_metrics["mean_bias_raw"] = float(np.mean(raw_biases))
        return params, out_metrics

    # ----------------------------------------------
    # 2.  Called before each round to supply config
    # ----------------------------------------------
    def configure_fit(self,
                      server_round: int,
                      parameters: fl.common.Parameters,
                      client_manager: fl.server.client_manager.ClientManager):
        # start from FedAvg defaults (sampling, same parameters for all)
        cfg = super().configure_fit(server_round, parameters, client_manager)
        out = []
        for client, fit_ins in cfg:
            cid = getattr(client, "cid", None)
            last_bias = 0.0
            if cid is not None and cid in self.client_state:
                last_bias = float(self.client_state[cid].get("bias", 0.0))
            # compute lambda from last bias
            with torch.no_grad():
                lam = float(self.lambda_policy(torch.tensor([last_bias], device=self.device)).item())
            # extend config
            new_conf = dict(fit_ins.config)
            new_conf["lambda"] = lam
            out.append((client, fl.common.FitIns(fit_ins.parameters, new_conf)))
        return out

# Why this is simple yet illustrative
# – λ is learned from bias via a tiny MLP.
# – The discriminator decides which clients are “fair”; its gradients flow into the λ‑policy.
# – No explicit graph GNN is used – we just broadcast the policy's output. Re‑implementing a GNN would require only a few more lines.

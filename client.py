# client.py
import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict
from typing import Dict, List, Tuple
import numpy as np

from utils import (SimpleCounterfactual, BiasEstimator,
                   compute_embedding, dp_clip_and_add_noise)
import json


# ---------------------------  Basic CNN  ---------------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool  = nn.AdaptiveAvgPool2d(1)
        self.fc    = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ---------------------------  Flower client --------------------------------
class FCFMClient(fl.client.NumPyClient):
    """
    A client that:
      – trains a local CNN
      – computes a DP‑protected bias estimate
      – returns the weight delta together with the bias payload
      – receives lambda from the server for the next round
    """
    def __init__(self,
                 model: nn.Module,
                 train_loader,
                 valid_loader,
                 device,
                 test_loader=None,
                 lambda_init: float = 1.0,
                 clip_norm_bias: float = 0.5,
                 noise_std_bias: float = 0.1,
                 clip_norm_emb: float = 10.0,
                 noise_std_emb: float = 1.0,
                 client_id: int = 0):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.device = device
        self.lambda_i = lambda_init
        self.clip_norm_bias = clip_norm_bias
        self.noise_std_bias = noise_std_bias
        self.clip_norm_emb = clip_norm_emb
        self.noise_std_emb = noise_std_emb
        self.client_id = client_id
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.BCEWithLogitsLoss()
        self.cf_gen = SimpleCounterfactual(device)
        self.bias_estimator = BiasEstimator(self.model, device)

    # ---------------------------  internal training  --------------------------------
    def _train_one_round(self, epochs: int = 1):
        self.model.train()
        for _ in range(epochs):
            for X, y, a in self.train_loader:
                X, y, a = X.to(self.device), y.to(self.device).float(), a.to(self.device).float()
                logits = self.model(X).squeeze()
                loss_pred = self.criterion(logits, y)
                # compute bias and uncertainty
                bias, uncert = self.bias_estimator.compute_bias(X, y, a,
                                                               self.cf_gen,
                                                               num_ensembles=3)
                loss = loss_pred + self.lambda_i * bias  # λ‑weighted
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    # ---------------------------  bias estimate  --------------------------------
    def _compute_bias_payload(self) -> dict:
        self.model.eval()
        biases = []
        uncertainties = []
        sum_emb = None
        total_emb = 0
        with torch.no_grad():
            for X, y, a in self.valid_loader:
                X, y, a = X.to(self.device), y.to(self.device).float(), a.to(self.device).float()
                bias, uncert = self.bias_estimator.compute_bias(X, y, a,
                                                               self.cf_gen,
                                                               num_ensembles=3)
                biases.append(bias)
                uncertainties.append(uncert)
                emb = compute_embedding(X)  # shape (batch, 10)
                # accumulate sum over batch for a stable mean across variable batch sizes
                batch_sum = emb.sum(axis=0)
                if sum_emb is None:
                    sum_emb = batch_sum.astype(np.float32)
                else:
                    sum_emb = sum_emb + batch_sum.astype(np.float32)
                total_emb += emb.shape[0]
        mean_bias = float(np.mean(biases)) if len(biases) else 0.0
        mean_uncert = float(np.mean(uncertainties)) if len(uncertainties) else 0.0
        if total_emb > 0 and sum_emb is not None:
            mean_emb = (sum_emb / float(total_emb)).astype(np.float32)
        else:
            mean_emb = np.zeros(10, dtype=np.float32)

        # DP clip + noise
        dp_bias = dp_clip_and_add_noise(
            torch.tensor(mean_bias, dtype=torch.float32),
            self.clip_norm_bias, self.noise_std_bias, eps=1.0,
            key=self.client_id
        )
        dp_uncert = dp_clip_and_add_noise(
            torch.tensor(mean_uncert, dtype=torch.float32),
            self.clip_norm_bias, self.noise_std_bias, eps=1.0,
            key=self.client_id + 100
        )
        dp_embed = dp_clip_and_add_noise(
            torch.tensor(mean_emb, dtype=torch.float32),
            self.clip_norm_emb, self.noise_std_emb, eps=1.0,
            key=self.client_id + 200
        )

        payload = {
            "bias_noisy": float(dp_bias),
            "bias_raw": float(mean_bias),
            "uncert": float(dp_uncert),
            # Flower metrics must be scalars/strings; encode array as JSON
            "embed":  json.dumps(np.asarray(dp_embed, dtype=np.float32).tolist()),
        }
        return payload

    # ---------------------------  Flower API --------------------------------
    def get_properties(self, ins) -> Dict[str, str]:
        return {"num_clients": str(1), "client_id": str(self.client_id)}

    def get_parameters(self, config=None):
        return [v.cpu().numpy() for v in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = OrderedDict(
            zip(self.model.state_dict().keys(), [torch.tensor(x) for x in parameters])
        )
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.lambda_i = config.get("lambda", self.lambda_i)  # updated λ from server
        round_epochs = config.get("local_epochs", 1)
        self._train_one_round(round_epochs)

        payload = self._compute_bias_payload()
        metrics = {
            "lambda": float(self.lambda_i),
            "bias_noisy": payload.get("bias_noisy"),
            "bias_raw": payload.get("bias_raw"),
            "uncert": payload.get("uncert"),
            "embed": payload.get("embed"),
        }
        return self.get_parameters(), len(self.train_loader.dataset), metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loader = self.test_loader if self.test_loader is not None else self.valid_loader
        total_loss = 0.0
        total_correct = 0
        total = 0
        with torch.no_grad():
            for X, y, _a in loader:
                X = X.to(self.device)
                y = y.to(self.device).float()
                logits = self.model(X).squeeze()
                loss = self.criterion(logits, y)
                total_loss += float(loss.item()) * X.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                total_correct += int((preds == y).sum().item())
                total += int(X.size(0))
        avg_loss = float(total_loss / total) if total > 0 else 0.0
        acc = float(total_correct / total) if total > 0 else 0.0
        return avg_loss, total, {"accuracy": acc}

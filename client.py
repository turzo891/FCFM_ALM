# client.py
import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict
from typing import Dict, List, Tuple

from utils import (SimpleCounterfactual, BiasEstimator,
                   compute_embedding, dp_clip_and_add_noise)


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
                 lambda_init: float = 1.0,
                 clip_norm_bias: float = 0.5,
                 noise_std_bias: float = 0.1,
                 clip_norm_emb: float = 10.0,
                 noise_std_emb: float = 1.0,
                 client_id: int = 0):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
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
        embeddings = []
        with torch.no_grad():
            for X, y, a in self.valid_loader:
                X, y, a = X.to(self.device), y.to(self.device).float(), a.to(self.device).float()
                bias, uncert = self.bias_estimator.compute_bias(X, y, a,
                                                               self.cf_gen,
                                                               num_ensembles=3)
                biases.append(bias)
                uncertainties.append(uncert)
                emb = compute_embedding(X)
                embeddings.append(emb)
        mean_bias = np.mean(biases)
        mean_uncert = np.mean(uncertainties)
        mean_emb = np.mean(embeddings, axis=0)  # 10 dims

        # DP clip + noise
        payload = {
            "bias":   dp_clip_and_add_noise(
                torch.tensor(mean_bias, dtype=torch.float32),
                self.clip_norm_bias, self.noise_std_bias, eps=1.0,
                key=self.client_id
            ),
            "uncert": dp_clip_and_add_noise(
                torch.tensor(mean_uncert, dtype=torch.float32),
                self.clip_norm_bias, self.noise_std_bias, eps=1.0,
                key=self.client_id + 100
            ),
            "embed":  dp_clip_and_add_noise(
                torch.tensor(mean_emb, dtype=torch.float32),
                self.clip_norm_emb, self.noise_std_emb, eps=1.0,
                key=self.client_id + 200
            ),
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
        return self.get_parameters(), len(self.train_loader.dataset), {
            "lambda": float(self.lambda_i),
            "bias":   payload["bias"],
            "uncert": payload["uncert"],
            "embed":  payload["embed"]
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        return 0.0, 0, {}

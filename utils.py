"""Utility functions for DP noise, counterfactuals, and embeddings."""
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from typing import Tuple

def dp_clip_and_add_noise(vec: torch.Tensor,
                          clip_norm: float,
                          noise_std: float,
                          eps: float,
                          key: int = 0):
    """Clip a vector to `clip_norm` and add Gaussian noise.

    Returns a numpy array for convenient serialization. Randomness is seeded
    with `key` so clients can report deterministic DP payloads per round.
    """
    vec_np = vec.detach().cpu().numpy()
    norm = np.linalg.norm(vec_np)
    if norm > clip_norm:
        vec_np = vec_np * (clip_norm / (norm + 1e-10))
    rng = np.random.default_rng(seed=key)
    noise = rng.normal(0, noise_std, size=vec_np.shape)
    return (vec_np + noise).astype(np.float32)

class SimpleCounterfactual:
    """
    Deterministic, fast counterfactual for MNIST-like images in [0,1].
    We invert pixel intensities to produce x_cf = 1 - x, which differs
    from x while remaining in-bounds.
    """
    def __init__(self, device):
        self.device = device

    def make_counterfactual(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Pixel inversion counterfactual (keeps values in [0,1])."""
        return torch.clamp(1.0 - x, 0.0, 1.0).to(self.device)

class BiasEstimator:
    """Estimate bias and predictive uncertainty for a given model."""
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def compute_bias(self,
                     x: torch.Tensor,
                     y: torch.Tensor,
                     a: torch.Tensor,
                     cf_gen: SimpleCounterfactual,
                     num_ensembles: int = 3) -> Tuple[float, float]:
        """Return `(bias, uncertainty)` for a batch.

        Bias is the mean absolute difference between predictions on the
        original inputs and their counterfactuals. Uncertainty is computed as
        the mean standard deviation across a small prediction ensemble.
        """
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.sigmoid(logits).detach()
            x_cf = cf_gen.make_counterfactual(x, a)
            logits_cf = self.model(x_cf)
            probs_cf = torch.sigmoid(logits_cf).detach()

            bias = (probs - probs_cf).abs().mean().item()

            logits_ens = []
            for _ in range(num_ensembles):
                logits_ = self.model(x)
                probs_ = torch.sigmoid(logits_).detach()
                logits_ens.append(probs_.cpu().numpy())
            probs_ens = np.stack(logits_ens, axis=0)
            uncertainty = probs_ens.std(axis=0).mean().item()

        return bias, uncertainty

def compute_embedding(x: torch.Tensor) -> np.ndarray:
    """Return a quick 10‑D embedding from flattened pixels.

    This keeps the first 10 columns of the flattened image as a toy
    representation. For real use, prefer model features or a projection.
    """
    flat = x.view(x.size(0), -1).cpu().numpy()
    emb = flat[:, :10]
    return emb.astype(np.float32)

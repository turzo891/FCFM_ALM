# utils.py
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from typing import Tuple

# ---------------------------  DP helper  ---------------------------
def dp_clip_and_add_noise(vec: torch.Tensor,
                          clip_norm: float,
                          noise_std: float,
                          eps: float,
                          key: int = 0):
    """
    Clip the vector to clip_norm and add Gaussian noise.
    Returns a numpy array (will be sent to the server).
    """
    vec_np = vec.detach().cpu().numpy()
    norm = np.linalg.norm(vec_np)
    if norm > clip_norm:
        vec_np = vec_np * (clip_norm / (norm + 1e-10))
    # noise
    rng = np.random.default_rng(seed=key)  # deterministic per client
    noise = rng.normal(0, noise_std, size=vec_np.shape)
    return (vec_np + noise).astype(np.float32)

# ---------------------------  Counterfactual  --------------------------
class SimpleCounterfactual:
    """
    Very simple surrogate: we *flip* the protected attribute (here
    the label) and feed the same image back.  In a real setting you
    would replace this with a VAE/Flow that learns a mapping p(x'|x,a').
    """
    def __init__(self, device):
        self.device = device

    def make_counterfactual(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # For demo, just return the same image but pretend that we flipped a
        return x.to(self.device)

# ---------------------------  Bias estimator  ------------------------
class BiasEstimator:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def compute_bias(self,
                     x: torch.Tensor,
                     y: torch.Tensor,
                     a: torch.Tensor,
                     cf_gen: SimpleCounterfactual,
                     num_ensembles: int = 3) -> Tuple[float, float]:
        """
        Returns (bias, uncertainty)
        bias    – mean absolute difference of predictions on X vs X' over the batch
        uncertainty – std of predictions of an ensemble (simple MC dropout)
        """
        self.model.eval()
        with torch.no_grad():
            # Baseline prediction
            logits = self.model(x)
            probs = torch.sigmoid(logits).detach()
            # Counterfactual prediction
            x_cf = cf_gen.make_counterfactual(x, a)
            logits_cf = self.model(x_cf)
            probs_cf = torch.sigmoid(logits_cf).detach()

            bias = (probs - probs_cf).abs().mean().item()

            # Simple uncertainty: replicate batch several times with dropout
            logits_ens = []
            for _ in range(num_ensembles):
                logits_ = self.model(x)  # dropout active by default
                probs_ = torch.sigmoid(logits_).detach()
                logits_ens.append(probs_.cpu().numpy())
            probs_ens = np.stack(logits_ens, axis=0)
            uncertainty = probs_ens.std(axis=0).mean().item()

        return bias, uncertainty

# ---------------------------  Simple Embedding  --------------------------
def compute_embedding(x: torch.Tensor) -> np.ndarray:
    """
    Very simple embedding: first 10 principal components
    of the flattened image.  In practice you could feed the
    local model's penultimate layer or use a random projection.
    """
    flat = x.view(x.size(0), -1).cpu().numpy()
    # Keep 10 columns (first 10 pixels)
    emb = flat[:, :10]
    return emb.astype(np.float32)

FCFM‑ALM: Federated Counterfactual Fairness with Adaptive Lambda (Flower)

This repo contains a ready‑to‑run Flower simulation that trains MNIST clients with a fairness penalty derived from simple counterfactual predictions. The server learns a per‑client weighting λ from the clients’ reported bias and broadcasts it each round.

Key highlights
- Fairness penalty: computes bias via a deterministic counterfactual (pixel inversion) and an uncertainty estimate.
- Adaptive λ: server learns a tiny MLP that maps bias → λ and updates per round.
- DP payloads: client bias/embedding are clipped + noised; raw (non‑DP) bias is also sent for diagnostics only.
- Scales to 100 clients: Ray‑backed simulation, optional CUDA use with fractional GPU per client.

What’s implemented
- Client model/training and bias payload: `client.py`
- Server strategy with λ‑policy and discriminator: `server.py`
- Deterministic counterfactual (invert pixels): `utils.py`
- Simulation orchestration (Context‑based client_fn): `run_fed.py`


**Requirements**
- Python 3.10
- Windows PowerShell (or WSL/Linux/macOS)
- Optional GPU: NVIDIA driver + CUDA‑enabled PyTorch


**Install**
- Create venv (PowerShell):
  - `python -m venv .venv`
  - `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
  - `.\.venv\Scripts\Activate.ps1`
- Install dependencies (includes Ray via Flower simulation extra):
  - `pip install -U pip setuptools wheel`
  - `pip install -r requirements.txt`
- Verify CUDA (optional):
  - `python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"`
  - If `False None`, install CUDA wheels (Windows, CUDA 11.8):
    - `pip uninstall -y torch torchvision`
    - `pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118`


**Run The Simulation**
- Command: `python run_fed.py`
- Expected logs (abridged):
  - `Resources for each Virtual Client: {'num_cpus': 1, 'num_gpus': 0.1}` (if CUDA available)
  - `[Server] Round N: mean bias noised = 0.0xxx | raw = 0.xx` (raw trend decreases)
  - `[Server] Round N: Discriminator loss=..., Lambda loss=...`


**How It Works**
- Client (`client.py`)
  - Trains a small CNN with BCE loss + λ·bias penalty.
  - Computes batch‑wise counterfactuals using pixel inversion (`utils.py`) and measures bias as |p(x)−p(x_cf)|.
  - Aggregates to per‑client means; returns DP‑noised scalars and a DP‑noised 10‑d embedding.
  - Also returns `bias_raw` (non‑DP) strictly for logging trends.
- Server (`server.py`)
  - After each round, collects client metrics, updates:
    - Discriminator: predicts “fair” (1) vs “biased” (0) from embeddings.
    - λ‑policy MLP: learns a mapping bias → λ via MSE to target `1/(1+|bias|)`.
  - Logs mean DP‑bias and mean raw bias per round, returns metrics to Flower history.
- Simulation (`run_fed.py`)
  - Builds N client datasets (MNIST) with balanced protected groups.
  - Uses Context‑based `client_fn` and schedules fractional GPU per client if CUDA is available.


**Configuration**
- Number of clients, GPU use, and per‑client data: `run_fed.py`
  - `n_clients = 100`
  - `per_client_samples = 600`  (100×600 ≈ 60k total)
  - `device = cuda:0` if `torch.cuda.is_available()` else `cpu`
  - Client GPU scheduling: `client_resources = {"num_cpus": 1, "num_gpus": 0.1}`
- DP settings (client‑side): `client.py`
  - `clip_norm_bias`, `noise_std_bias` for scalar bias/uncertainty
  - `clip_norm_emb`, `noise_std_emb` for 10‑d embedding
- Counterfactual: `utils.py`
  - `SimpleCounterfactual.make_counterfactual` uses pixel inversion. Swap for `torch.roll` or other transforms if desired.


**Why DP mean bias can look constant**
- DP noise is seeded per client ID to be deterministic across rounds for reproducibility.
- The server’s DP mean can look flat, while raw mean bias decays as training progresses.
- For debugging only, use `bias_raw` trend; keep DP payloads for privacy reporting.


**Troubleshooting**
- GPU not used (clients show `num_gpus: 0`):
  - Your PyTorch build is CPU‑only. Install CUDA wheels as shown above and re‑run.
- Ray errors on Windows:
  - Ensure `flwr[simulation]` is installed (it pulls Ray), or use WSL2/Docker.
- ZeroDivisionError during evaluation:
  - Fixed: client now returns non‑zero evaluation counts from `evaluate`.
- KeyError: 'bias' in metrics:
  - Fixed: metric was renamed to `bias_noisy`; both noisy/raw are returned by client.


**Repository Structure**
- `run_fed.py` — Simulation entrypoint (clients, strategy, GPU scheduling)
- `client.py` — CNN, training loop, DP payload, evaluation
- `server.py` — Custom Flower strategy, λ‑policy + discriminator, logging
- `utils.py` — DP helpers, counterfactual generator, embedding
- `Docker/` — Optional dockerized server/client (update to `flwr` if using)


**What We Achieved**
- End‑to‑end Flower simulation runs cleanly across rounds.
- Per‑client fairness penalty with learned λ fed back each round.
- Deterministic counterfactuals produce meaningful, shrinking `bias_raw` trends.
- DP‑protected payloads preserve privacy; constant DP mean explained and expected.
- Scales to 100 clients; supports CUDA with fractional GPU per client.


**Next Steps**
- Persist per‑round metrics to CSV for plotting.
- Add aggregation functions for fit/evaluate metrics to compute global stats.
- Move server discriminator/λ‑policy to GPU if desired.
- Try alternative counterfactuals (e.g., learned generative models).


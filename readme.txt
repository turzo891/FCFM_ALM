How to run
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the test
python run_fed.py
The console will print the server‑side losses and the λ value that is sent back to each client every round. After 20 rounds you should see the bias (AUC‑DP, EO) dropping visibly if you instrument the code for that evaluation (see the BiasEstimator in utils.py). The communication per round is minimal: each client transmits

a DP‑clipped 1‑dim bias,
a 1‑dim uncertainty,
and a 10‑dim embedding (≈ 40 bytes) – well below 1 kB.
Extending the test
Real counterfactuals – replace SimpleCounterfactual with a VAE/Flow that learns (x′∣x,a′).
Graph propagation – attach a small GCN (torch_geometric or pytorch_geometric) that takes the DP‑protected embeddings as nodes and propagates a fairness‑confidence score.
DP‑calculus – use the moments accountant or RDP to sum privacy loss over rounds.
Happy coding – this scaffold brings the FCFM‑AML pipeline from concept to a running prototype in under 200 lines of code. It is fully ready to be turned into a reproducible experiment for the paper!


Next Step Checklist
1.0Run the demo – confirm you see the server logs and that the client payloads have sizes < 1 kB.
2.Add evaluation – modify client.py to return the raw prediction probabilities; let the server compute fairness metrics.
3.Replace the surrogate generator – train a simple VAE and use it inside SimpleCounterfactual for realistic counterfactuals.
4.Add a GNN – optional but demonstrates how similarity between clients can improve λ propagation.
5.Integrate DP accounting – to certify privacy (ε, δ) across all rounds.
Once the above additions are in place, you will have a research‑grade federated pipeline that satisfies the three bullets you specified: real‑time bias discovery, dynamic λ adjustment, and privacy‑preserving knowledge sharing.



python -m venv .venv
.\.venv\Scripts\Activate.ps1


PowerShell: .\.venv\Scripts\Activate.ps1
CMD: .\.venv\Scripts\activate.bat
Git Bash/MSYS: source .venv/Scripts/activate

Upgrade tools: python -m pip install -U pip setuptools wheel
Install: python -m pip install -r requirements.txt
If needed for simulation: python -m pip install "flwr[simulation]==1.11.0"

Run: python run_fed.py
Common Pitfall (Execution Policy)

If you see “running scripts is disabled”, run:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
Then: .\.venv\Scripts\Activate.ps1
Or: Unblock-File .\.venv\Scripts\Activate.ps1 once
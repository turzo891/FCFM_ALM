"""Simulation entrypoint for the FCFM‑ALM Flower demo.

Builds MNIST‑based client datasets, instantiates Flower clients, and runs a
Ray‑backed simulation with optional CUDA. Clients report DP fairness signals
and the server adapts a per‑client λ each round.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
import torchvision.transforms as T
import numpy as np
import flwr as fl
from typing import List
from flwr.common import Context

from client import FCFMClient, SimpleCNN
from server import FCFMStrategy

def load_mnist_clients(n_clients: int,
                       per_client_samples: int = 1000) -> List[Dataset]:
    """Build `n_clients` subsets with balanced groups per client."""
    transform = T.Compose([T.ToTensor()])
    full_ds = torchvision.datasets.MNIST(root='./data',
                                         train=True,
                                         download=True,
                                         transform=transform)
    labels = (full_ds.targets != 0).int()
    idx_prot  = (labels == 0).nonzero(as_tuple=False).view(-1)
    idx_nonp  = (labels == 1).nonzero(as_tuple=False).view(-1)

    rng = np.random.default_rng(42)
    clients = []
    samples_per_group = per_client_samples // 2
    for _ in range(n_clients):
        sel0 = rng.choice(idx_prot.numpy(), samples_per_group, replace=False)
        sel1 = rng.choice(idx_nonp.numpy(), samples_per_group, replace=False)
        sel  = np.concatenate([sel0, sel1])
        rng.shuffle(sel)
        subset = Subset(full_ds, sel)
        clients.append((subset, labels[sel]))
    return clients


class ClientDataset(Dataset):
    """Dataset wrapper exposing image, label, and protected attribute."""
    def __init__(self, base_ds, labels):
        self.base_ds = base_ds
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img, _ = self.base_ds[idx]
        y = self.labels[idx].item()
        a = y
        return img, torch.tensor(y, dtype=torch.long), torch.tensor(a, dtype=torch.long)


def build_clients(n_clients=5,
                  per_client_samples=600,
                  batch_size=32,
                  device=None):
    """Instantiate clients and their data loaders."""
    raw_clients = load_mnist_clients(n_clients, per_client_samples=per_client_samples)
    client_objs = []

    for i, (subset, labels) in enumerate(raw_clients):
        ds = ClientDataset(subset, labels)
        train_ds, val_ds, test_ds = torch.utils.data.random_split(
            ds, [int(0.6*len(ds)), int(0.2*len(ds)), int(0.2*len(ds))])

        pin_mem = bool(device and device.type == "cuda")
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=pin_mem, num_workers=0)
        val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=pin_mem, num_workers=0)
        test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=pin_mem, num_workers=0)

        model = SimpleCNN()
        client = FCFMClient(model=model,
                            train_loader=train_loader,
                            valid_loader=val_loader,
                            device=device or torch.device("cpu"),
                            test_loader=test_loader,
                            lambda_init=1.0,
                            client_id=i)
        client_objs.append(client)

    return client_objs

def main():
    """Configure and run the Flower simulation."""
    n_clients = 10
    n_rounds = 20

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0") if use_cuda else torch.device("cpu")
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    per_client_samples = 600
    batch_size = 32

    clients = build_clients(n_clients,
                            per_client_samples=per_client_samples,
                            batch_size=batch_size,
                            device=device)

    strategy = FCFMStrategy(
        fraction_fit=1.0,
        min_fit_clients=n_clients,
        min_available_clients=n_clients
    )

    def client_wrapper(context: Context):
        cid = int(getattr(context, "cid", 0))
        return clients[cid].to_client()

    client_resources = {"num_cpus": 1, "num_gpus": (0.1 if use_cuda else 0)}

    fl.simulation.start_simulation(
        client_fn=client_wrapper,
        num_clients=n_clients,
        config=fl.server.ServerConfig(num_rounds=n_rounds),
        strategy=strategy,
        client_resources=client_resources,
    )


if __name__ == "__main__":
    main()

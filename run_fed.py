# run_fed.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
import torchvision.transforms as T
import numpy as np
import flwr as fl

from client import FCFMClient
from server import FCFMStrategy

# -----------------------------------------------------------------
# 1.  Build a toy dataset: MNIST digits 0 (label 0) vs 1 (label 1).
#    We create a *protected attribute* a = label (0 for “protected”).
# -----------------------------------------------------------------
def load_mnist_clients(n_clients: int,
                       per_client_samples: int = 1000) -> List[Dataset]:
    transform = T.Compose([T.ToTensor()])
    full_ds = torchvision.datasets.MNIST(root='./data',
                                         train=True,
                                         download=True,
                                         transform=transform)
    labels = (full_ds.targets != 0).int()   # 0 ↦ protected, 1 ↦ non‑protected
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
    def __init__(self, base_ds, labels):
        self.base_ds = base_ds
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img, _ = self.base_ds[idx]
        y = self.labels[idx].item()
        a = y  # protected attribute == label for this toy demo
        return img, torch.tensor(y, dtype=torch.long), torch.tensor(a, dtype=torch.long)


# -----------------------------------------------------------------
# 2.  Build Flower client objects
# -----------------------------------------------------------------
def build_clients(n_clients=5):
    raw_clients = load_mnist_clients(n_clients, per_client_samples=2000)
    client_objs = []

    for i, (subset, labels) in enumerate(raw_clients):
        ds = ClientDataset(subset, labels)
        train_ds, val_ds, test_ds = torch.utils.data.random_split(
            ds, [int(0.6*len(ds)), int(0.2*len(ds)), int(0.2*len(ds))])

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

        model = SimpleCNN()
        client = FCFMClient(model=model,
                            train_loader=train_loader,
                            valid_loader=val_loader,
                            device=torch.device("cpu"),
                            lambda_init=1.0,
                            client_id=i)
        # Wrap to Flower client
        flower_client = fl.client.NumPyClient(
            get_parameters=client.get_parameters,
            set_parameters=client.set_parameters,
            fit=client.fit,
            evaluate=client.evaluate
        )

        # Keep a reference so the lambda policy can be accessed later
        flower_client._internal_client = client
        client_objs.append(flower_client)

    return client_objs


# -----------------------------------------------------------------
# 3.  Launch the simulation
# -----------------------------------------------------------------
def main():
    n_clients = 5
    n_rounds = 20
    clients = build_clients(n_clients)

    strategy = FCFMStrategy(
        # usual FedAvg parameters
        fraction_fit=1.0,     # use all clients
        min_fit_clients=n_clients,
        min_available_clients=n_clients
    )

    # Flower simulation
    fl.simulation.start_simulation(
        client_fn=lambda cid: clients[int(cid)],
        num_clients=n_clients,
        config=fl.server.ServerConfig(num_rounds=n_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0},
    )


if __name__ == "__main__":
    main()

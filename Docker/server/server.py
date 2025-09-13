import flwr as fl
from server import FCFMStrategy  # the custom strategy you already wrote

if __name__ == "__main__":
    strategy = FCFMStrategy(
        fraction_fit=1.0,
        min_fit_clients=1,
        min_available_clients=1,
    )

    # 80 % of the port will be used for the gRPC server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=20),
    )

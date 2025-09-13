server/server.py
Start the server

cd FL‑BC‑Demo/server
python server.py
You should see a log line like:
[*] (Server [id: 1]) listening on 0.0.0.0:8080









client/entrypoint.sh
Make the script executable

chmod +x client/entrypoint.sh
client/requirements.txt (optional, but included for clarity)
Same as the server’s requirements – you can keep one copy in the root directory.









docker-compose.yml
How it works

host.docker.internal resolves to the host’s IP from inside the container (works on macOS/Windows ≥ 20.10).
On Linux you can simply use the host’s IP address (e.g. 192.168.1.10).
The server container exposes port 8080 so that all clients can reach 0.0.0.0:8080 from inside the network.
Run it

docker compose up -d    # Start everything in the background
You will see logs similar to:

[+] Running 3/3
 ✔ Container fcfm_server   Started
 ✔ Container fcfm_client1  Started
 ✔ Container fcfm_client2  Started
The clients connect to the server over the Docker network and begin federated rounds immediately.













Working with a dedicated host (real network)
If your server sits on a remote VM (e.g. AWS EC2, GCP Compute Engine) simply:

Open port 8080 in the VM’s firewall.

In each Docker client container, set the environment variable:

SERVER_ADDRESS=203.0.113.42:8080   # <-- your server IP
You can keep this in docker-compose.yml or set it at runtime:

docker run -e SERVER_ADDRESS=203.0.113.42:8080 --network host client_image
Launch the server on the VM with:

python server.py
Clients across the Internet will now talk to the same FL server.

6. Security / Enhancements
Feature	How to enable
TLS / Mutual‑TLS	Pass --tls and key/cert paths to fl.server.start_server and fl.client.start_numpy_client.
Authentication	Use Flower’s built‑in authentication or wrap the gRPC connections.
Docker secrets	If you have any secrets (e.g. API keys) use Docker Swarm’s secret management or a cloud secret manager.
Resource limits	Add deploy.resources.limits in docker-compose.yml (CPU, memory) to cap client resources.
7. Quick sanity check
# On the desktop – run server
python server/server.py

# On any machine – run a single client (no Docker)
python client/client.py

# Or use the Docker client
docker compose up -d
You should see the server printing round numbers and the clients writing “Client <id> starting…” logs.
If the server never receives any payload, double‑check:

SERVER_ADDRESS is reachable from the container (ping).
The Flower port is open (nc -zv <IP> 8080).
All containers share the same Docker network (i.e. Compose’s default network).
Bottom line – the desktop can act as the federated learning server; each client runs inside its own Docker container, connects over gRPC, and participates in the FCFM‑AML round. This architecture is production‑ready, scales linearly with the number of containers, and preserves the privacy guarantees that you built into the code.
#!/usr/bin/env bash
# 1. Warm‑up (optional)
echo "Client $(hostname) starting…"

# 2. Point to the FL server (replace 192.168.1.10 with your desktop IP)
#    If you keep the server on 127.0.0.1 (same machine) change accordingly.
SERVER_ADDRESS=${SERVER_ADDRESS:-0.0.0.0:8080}

# 3. Run the Flower client
fl.client.start_numpy_client(server_address=$SERVER_ADDRESS)

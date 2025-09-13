# Base image
FROM python:3.10-slim

# Install dependencies
RUN pip install --no-cache-dir \
    torch==2.0.1 \
    torchvision==0.15.1 \
    flower==1.11.0

# Copy the client code
COPY client.py   /app/
COPY utils.py    /app/
COPY entrypoint.sh /app/

WORKDIR /app

# Default command – will be overwritten by docker‑compose
ENTRYPOINT ["bash", "entrypoint.sh"]

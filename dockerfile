# Dockerfile — stable: use nvidia/cuda base and install torch+cu126
FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

# Install Python & system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev python3-venv build-essential ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first for caching
COPY requirements.txt /app/requirements.txt

# Upgrade pip and install torch (cu126 wheel) BEFORE other deps to avoid conflicts
RUN python3 -m pip install --upgrade pip setuptools wheel
RUN python3 -m pip install --no-cache-dir "torch==2.6.0+cu126" --index-url https://download.pytorch.org/whl/cu126

# Install the rest of the python requirements (ensure requirements.txt does NOT pin torch/torchvision)
RUN python3 -m pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY . /app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Inference server for the merged Qwen fine-tuned model.
# The model weights are NOT baked into the image — they are mounted as a volume
# at runtime (see docker-compose.yml). This keeps the image portable.

FROM python:3.11-slim

WORKDIR /app

# Install curl for health checks inside the container
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (saves ~1.5 GB vs the default CUDA build)
RUN pip install --no-cache-dir \
    torch==2.6.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the inference server script
COPY serve.py .

# Model weights are mounted at runtime — this path must match docker-compose.yml
ENV MODEL_PATH=/app/models/merged

# Suppress the HuggingFace tokenizer fork-safety warning (single-worker server)
ENV TOKENIZERS_PARALLELISM=false

# Unbuffered stdout so logs appear immediately in `docker compose logs`
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Single worker — the model is not safe to share across forked processes
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

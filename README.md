# LLM Fine-Tuning Pipeline on Cloud Infrastructure

**CSL-520 Course Project** — End-to-end pipeline that takes a raw dataset, fine-tunes an open-source LLM using LoRA, stores the adapter in a model registry, and evaluates the result against the base model.

---

## Architecture

```
Raw Dataset
    │
    ▼
[1] preprocess.py       ─── Converts to ChatML JSONL (train / val / test splits)
    │
    ▼
[2] finetune_training.py ── QLoRA fine-tuning on Qwen2.5-1.5B-Instruct (Kaggle GPU)
                            https://www.kaggle.com/code/haaziq/cloud-qwen-finetune
    │
    ▼
[3] upload_s3.py         ── Uploads LoRA adapter to AWS S3
    register_model.py    ── Registers adapter version in MLflow
    │
    ▼
[4] merge_adapter.py     ── Downloads adapter from S3, merges into base model
    serve.py + Docker    ── FastAPI inference server (Docker container)
    evaluate.py          ── ROUGE evaluation: base model vs fine-tuned model
```

**Model:** `Qwen/Qwen2.5-1.5B-Instruct`  
**Fine-tuning method:** QLoRA (4-bit quantization during training, LoRA r=16, α=32)  
**Deployment:** FastAPI + HuggingFace Transformers + Docker (CPU inference, no GPU required)  
**Model registry:** AWS S3 + MLflow  

---

## Repository Structure

| File | Stage | Description |
|------|-------|-------------|
| `download_data.py` | 1 | Downloads the Databricks Dolly 15k dataset |
| `preprocess.py` | 1 | Converts raw data to ChatML JSONL; splits into train/val/test |
| `finetune_training.py` | 2 | QLoRA fine-tuning with TRL/PEFT; logs metrics to MLflow |
| `upload_s3.py` | 3 | Uploads LoRA adapter directory to S3 with timestamp versioning |
| `register_model.py` | 3 | Registers the S3 adapter as a versioned model in MLflow |
| `merge_adapter.py` | 4 | Downloads adapter from S3, merges into base model, saves locally |
| `serve.py` | 4 | FastAPI inference server (`/health`, `/generate` endpoints) |
| `Dockerfile` | 4 | Docker image for the inference server (CPU PyTorch) |
| `docker-compose.yml` | 4 | Orchestrates the inference service |
| `evaluate.py` | 4 | ROUGE evaluation comparing base vs fine-tuned model |
| `requirements.txt` | — | Python dependencies |

---

## Setup

### 1. Clone and configure environment

```bash
git clone <repo-url>
cd Qwen-Fine-Tuning-Pipeline-on-Cloud-Infrastructure

cp .env.example .env
# Fill in your AWS credentials in .env
```

### 2. Install dependencies

```bash
# Install PyTorch (CPU build — works without a GPU)
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu

# Install remaining packages
pip install -r requirements.txt
```

---

## Stage 1 — Data Preparation

```bash
# Download Databricks Dolly 15k
python download_data.py --output data/raw_data.jsonl

# Convert to ChatML format and split into train/val/test (80/10/10)
python preprocess.py \
    --input data/raw_data.jsonl \
    --output data/final_data.jsonl \
    --format chatml
# Outputs: data/final_data_train.jsonl, data/final_data_val.jsonl, data/final_data_test.jsonl
```

---

## Stage 2 — QLoRA Fine-Tuning

Fine-tuning was run on **Kaggle** (free T4 GPU). The script is `finetune_training.py`.

**Kaggle Notebook:** [cloud-qwen-finetune](https://www.kaggle.com/code/haaziq/cloud-qwen-finetune)

Key configuration:
- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Quantization: 4-bit NF4 (BitsAndBytes)
- LoRA: r=16, α=32, dropout=0.05, 7 target modules
- Training: 3 epochs, batch size 2, gradient accumulation 8, lr=2e-4
- MLflow tracking enabled

The fine-tuned adapter is saved to `final_adapter/` on the training machine (Kaggle output).

---

## Stage 3 — Model Registry

```bash
# Upload the LoRA adapter to S3
python upload_s3.py --model-dir ./final_adapter

# Register the uploaded version in MLflow (start MLflow server first)
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow-artifacts &
python register_model.py
```

The adapter is stored at:
```
s3://cloud-project-model-registery/models/qwen-1.5b-finetuned/v20260406_095050/
```

Files in S3: `adapter_config.json`, `adapter_model.safetensors`, `chat_template.jinja`, `tokenizer_config.json`, `tokenizer.json`

---

## Stage 4 — Deployment & Evaluation

### Why FastAPI + Transformers instead of vLLM or Ollama?

- **vLLM** requires a dedicated GPU instance — too expensive for a course project.
- **Ollama** requires converting the LoRA adapter to GGUF format first (merge → llama.cpp convert → quantize → Modelfile), adding unnecessary complexity.
- **FastAPI + Transformers** loads the merged model directly from safetensors, consistent with the training stack, and runs on CPU for free.

### Step 4a — Merge the LoRA adapter

Downloads the adapter from S3, loads the base model in full precision, merges the weights, and saves the result locally. Run this once.

```bash
python merge_adapter.py
# Saves merged model to ./models/merged/
# Takes ~5-10 minutes on CPU (downloads base model on first run)
```

### Step 4b — Start the inference server

```bash
docker compose up -d

# Check health
curl http://localhost:8000/health
# {"status":"ok","model":"qwen-finetuned","model_path":"/app/models/merged"}

# Test inference
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain what machine learning is in simple terms.", "max_new_tokens": 150}'
```

The server starts in ~30-90 seconds (loading a 1.5B float32 model on CPU). Check status with:

```bash
docker compose ps
docker compose logs -f
```

### Step 4c — Run evaluation

Compares base model vs fine-tuned model on the test split using ROUGE scores.
The base model is loaded in-process (then unloaded), and the fine-tuned model is queried via the running Docker server — to avoid holding two 6 GB models in RAM at once.

```bash
python evaluate.py \
    --test-file data/final_data_test.jsonl \
    --max-samples 50

# Results are saved to evaluation_results.json
```

Example output:
```
┌────────────────────────────────────────────────────────────┐
│          Evaluation: Base vs Fine-Tuned (ROUGE F1)         │
├─────┬──────────────────┬──────────┬──────────┬────────────┤
│  #  │ Instruction      │ Base R-L │ FT R-L   │ Base R-1   │
├─────┼──────────────────┼──────────┼──────────┼────────────┤
│  1  │ What is...       │ 0.312    │ 0.447    │ 0.401      │
└─────┴──────────────────┴──────────┴──────────┴────────────┘

Summary (mean over all examples):
  Base model    — ROUGE-1: 0.3812  ROUGE-2: 0.2104  ROUGE-L: 0.3156
  Fine-tuned    — ROUGE-1: 0.4921  ROUGE-2: 0.2897  ROUGE-L: 0.4103
  Delta         — ROUGE-1: +0.1109  ROUGE-2: +0.0793  ROUGE-L: +0.0947
```

### Stop the server

```bash
docker compose down
```

---

## API Reference

### `POST /generate`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt` | string | required | Raw instruction text |
| `max_new_tokens` | int | 256 | Maximum tokens to generate |
| `temperature` | float | 0.1 | Sampling temperature (only used when `do_sample=true`) |
| `do_sample` | bool | false | Use sampling (false = greedy decode) |

**Response:**
```json
{
  "response": "Machine learning is...",
  "prompt_tokens": 42,
  "generated_tokens": 128
}
```

---

## Tools Used

| Tool | Purpose |
|------|---------|
| Hugging Face Transformers | Model loading, tokenization, inference |
| PEFT | LoRA adapter management |
| TRL | SFTTrainer for fine-tuning |
| BitsAndBytes | 4-bit quantization during training |
| MLflow | Experiment tracking and model registry |
| AWS S3 (boto3) | Adapter storage |
| FastAPI + Uvicorn | Inference server |
| Docker | Containerized deployment |
| rouge-score | Evaluation metrics |

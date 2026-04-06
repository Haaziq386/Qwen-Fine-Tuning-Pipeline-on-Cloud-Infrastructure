"""
serve.py
--------
FastAPI inference server for the merged Qwen fine-tuned model.

Usage (directly):
    MODEL_PATH=./models/merged uvicorn serve:app --host 0.0.0.0 --port 8000 --workers 1

Usage (via Docker):
    docker compose up -d

Endpoints:
    GET  /health   → {"status": "ok", "model": "qwen-finetuned"}
    POST /generate → {"prompt": "...", "max_new_tokens": 256} → {"response": "..."}
"""

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = os.getenv("MODEL_PATH", "./models/merged")
# Requests are appended here as JSON Lines — file lives on a host bind mount
# so logs survive container restarts and `docker compose down`.
LOG_PATH = os.getenv("LOG_PATH", "./logs/requests.jsonl")

# Global model/tokenizer — loaded once at startup
_tokenizer = None
_model = None


def _append_log(entry: dict) -> None:
    """Append a JSON record to the persistent log file."""
    try:
        Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass  # Never let logging crash the server


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _tokenizer, _model
    print(f"Loading model from {MODEL_PATH} ...")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    _tokenizer.pad_token = _tokenizer.eos_token  # Qwen has no pad_token by default

    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    _model.eval()
    print("Model loaded. Server ready.")
    _append_log({"event": "startup", "model_path": MODEL_PATH, "ts": time.time()})
    yield
    # Cleanup on shutdown
    _append_log({"event": "shutdown", "ts": time.time()})
    del _model, _tokenizer


app = FastAPI(title="Qwen Fine-Tuned Inference Server", lifespan=lifespan)


# ---------- Request / Response schemas ----------


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 0.1
    do_sample: bool = False


class GenerateResponse(BaseModel):
    response: str
    prompt_tokens: int
    generated_tokens: int


# ---------- Endpoints ----------


@app.get("/health")
def health():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return {"status": "ok", "model": "qwen-finetuned", "model_path": MODEL_PATH}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if _model is None or _tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    # Wrap the raw instruction in Qwen's chat template so the format matches training
    messages = [{"role": "user", "content": req.prompt}]
    text = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = _tokenizer(text, return_tensors="pt")
    prompt_len = inputs["input_ids"].shape[1]

    generate_kwargs = dict(
        **inputs,
        max_new_tokens=req.max_new_tokens,
        do_sample=req.do_sample,
        pad_token_id=_tokenizer.eos_token_id,
    )
    # Only pass temperature when sampling; greedy decode ignores it
    if req.do_sample:
        generate_kwargs["temperature"] = req.temperature

    t0 = time.time()
    with torch.no_grad():
        outputs = _model.generate(**generate_kwargs)
    latency = round(time.time() - t0, 3)

    # Slice off the prompt tokens — keep only generated portion
    new_tokens = outputs[0][prompt_len:]
    response_text = _tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Persist request log to the host-mounted ./logs/ directory
    _append_log(
        {
            "ts": time.time(),
            "prompt_tokens": prompt_len,
            "generated_tokens": len(new_tokens),
            "latency_s": latency,
            "max_new_tokens": req.max_new_tokens,
            "do_sample": req.do_sample,
        }
    )

    return GenerateResponse(
        response=response_text,
        prompt_tokens=prompt_len,
        generated_tokens=len(new_tokens),
    )

"""
evaluate.py
-----------
Compares the fine-tuned Qwen model against the base model on the held-out
test set, computing ROUGE scores and printing a side-by-side comparison.

Prerequisites:
    1. Run `python merge_adapter.py` to create ./models/merged/
    2. Start the inference server: `docker compose up -d`
    3. Run this script: `python evaluate.py`

The fine-tuned model is queried via the running Docker inference server.
The base model is loaded in-process (sequentially, BEFORE querying the server)
to avoid holding two 6 GB models in RAM simultaneously.

Results are saved to evaluation_results.json.
"""

import argparse
import gc
import json
import time
from pathlib import Path

import requests
import torch
from rich.console import Console
from rich.table import Table
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
console = Console()


# ---------- Data loading ----------

def load_test_examples(test_file: str, max_samples: int) -> list[dict]:
    """
    Load examples from a ChatML-format JSONL test file.
    Each record: {"messages": [{"role": "user"|"assistant"|"system", "content": "..."}]}
    Returns a list of {"instruction": str, "reference": str}.
    """
    examples = []
    path = Path(test_file)
    if not path.exists():
        raise FileNotFoundError(
            f"Test file not found: {test_file}\n"
            "Run `python preprocess.py` first to generate the test split."
        )

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            messages = record.get("messages", [])

            try:
                instruction = next(
                    m["content"] for m in messages if m["role"] == "user"
                )
                reference = next(
                    m["content"] for m in reversed(messages) if m["role"] == "assistant"
                )
            except StopIteration:
                continue  # Skip malformed records

            examples.append({"instruction": instruction, "reference": reference})
            if len(examples) >= max_samples:
                break

    return examples


# ---------- Base model inference (in-process) ----------

def run_base_model(examples: list[dict], max_new_tokens: int) -> list[str]:
    """Load the base Qwen model, run inference on all examples, then unload it."""
    console.print(f"\n[bold cyan]Phase A:[/] Loading base model ({BASE_MODEL_ID}) in-process...")
    console.print("  (This downloads the base model on first run and may take a few minutes)")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    model.eval()

    predictions = []
    for i, ex in enumerate(examples, 1):
        console.print(f"  [{i}/{len(examples)}] Generating (base model)...", end="\r")
        messages = [{"role": "user", "content": ex["instruction"]}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt")
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][prompt_len:]
        predictions.append(tokenizer.decode(new_tokens, skip_special_tokens=True))

    console.print(f"\n  Done. Generated {len(predictions)} predictions.")

    # Explicitly free memory before loading the fine-tuned model via Docker
    del model
    del tokenizer
    gc.collect()
    console.print("  Base model unloaded from memory.")

    return predictions


# ---------- Fine-tuned model inference (via HTTP) ----------

def wait_for_server(serve_url: str, timeout: int = 180) -> None:
    """Poll /health until the inference server is ready."""
    health_url = serve_url.rstrip("/") + "/health"
    deadline = time.time() + timeout
    console.print(f"\n[bold cyan]Phase B:[/] Waiting for inference server at {health_url}")
    while time.time() < deadline:
        try:
            r = requests.get(health_url, timeout=5)
            if r.status_code == 200:
                console.print("  Server is ready.")
                return
        except requests.exceptions.ConnectionError:
            pass
        console.print("  Server not ready yet, retrying in 10s...", end="\r")
        time.sleep(10)
    raise TimeoutError(
        f"Inference server did not become ready within {timeout}s.\n"
        "Make sure you ran: docker compose up -d"
    )


def run_finetuned_model(
    examples: list[dict], serve_url: str, max_new_tokens: int
) -> list[str]:
    """Query the running Docker inference server for all examples."""
    generate_url = serve_url.rstrip("/") + "/generate"
    predictions = []

    for i, ex in enumerate(examples, 1):
        console.print(f"  [{i}/{len(examples)}] Querying fine-tuned model...", end="\r")
        payload = {
            "prompt": ex["instruction"],
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
        }
        try:
            r = requests.post(generate_url, json=payload, timeout=300)
            r.raise_for_status()
            predictions.append(r.json()["response"])
        except requests.exceptions.RequestException as e:
            console.print(f"\n  [red]Error on example {i}:[/] {e}")
            predictions.append("")

    console.print(f"\n  Done. Generated {len(predictions)} predictions.")
    return predictions


# ---------- Scoring ----------

def compute_rouge(reference: str, prediction: str, scorer_obj) -> dict:
    if not prediction.strip():
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    scores = scorer_obj.score(reference, prediction)
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }


# ---------- Output ----------

def print_results_table(results: list[dict]) -> None:
    table = Table(title="Evaluation: Base vs Fine-Tuned (ROUGE F1)", show_lines=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Instruction", max_width=30)
    table.add_column("Base R-L", justify="right")
    table.add_column("FT R-L", justify="right")
    table.add_column("Base R-1", justify="right")
    table.add_column("FT R-1", justify="right")

    for r in results[:20]:  # Show first 20 rows to keep output manageable
        rl_diff = r["finetuned"]["rougeL"] - r["base"]["rougeL"]
        rl_color = "green" if rl_diff >= 0 else "red"
        table.add_row(
            str(r["index"]),
            r["instruction"][:50] + ("..." if len(r["instruction"]) > 50 else ""),
            f"{r['base']['rougeL']:.3f}",
            f"[{rl_color}]{r['finetuned']['rougeL']:.3f}[/{rl_color}]",
            f"{r['base']['rouge1']:.3f}",
            f"{r['finetuned']['rouge1']:.3f}",
        )

    console.print(table)


def print_summary(results: list[dict]) -> None:
    def avg(key, model):
        return sum(r[model][key] for r in results) / len(results)

    console.print("\n[bold]Summary (mean over all examples)[/]")
    console.print(
        f"  Base model    — ROUGE-1: {avg('rouge1', 'base'):.4f}  "
        f"ROUGE-2: {avg('rouge2', 'base'):.4f}  "
        f"ROUGE-L: {avg('rougeL', 'base'):.4f}"
    )
    console.print(
        f"  Fine-tuned    — ROUGE-1: {avg('rouge1', 'finetuned'):.4f}  "
        f"ROUGE-2: {avg('rouge2', 'finetuned'):.4f}  "
        f"ROUGE-L: {avg('rougeL', 'finetuned'):.4f}"
    )
    delta_r1 = avg('rouge1', 'finetuned') - avg('rouge1', 'base')
    delta_r2 = avg('rouge2', 'finetuned') - avg('rouge2', 'base')
    delta_rl = avg('rougeL', 'finetuned') - avg('rougeL', 'base')
    color = "green" if delta_rl >= 0 else "red"
    console.print(
        f"  [bold {color}]Delta         — ROUGE-1: {delta_r1:+.4f}  "
        f"ROUGE-2: {delta_r2:+.4f}  "
        f"ROUGE-L: {delta_rl:+.4f}[/]"
    )


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned Qwen model vs base model on test JSONL"
    )
    parser.add_argument(
        "--test-file",
        default="data/final_data_test.jsonl",
        help="Path to test JSONL file (default: data/final_data_test.jsonl)",
    )
    parser.add_argument(
        "--serve-url",
        default="http://localhost:8000",
        help="Base URL of the fine-tuned model inference server (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50,
        help="Number of test examples to evaluate (default: 50)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max tokens to generate per example (default: 256)",
    )
    parser.add_argument(
        "--output",
        default="evaluation_results.json",
        help="Path to save JSON results (default: evaluation_results.json)",
    )
    args = parser.parse_args()

    console.rule("[bold]Qwen Fine-Tuning Pipeline — Evaluation")

    # Load test data
    console.print(f"\nLoading test examples from {args.test_file}")
    examples = load_test_examples(args.test_file, args.max_samples)
    console.print(f"  Loaded {len(examples)} examples.")

    # Phase A: Base model inference (in-process, then unloaded)
    base_predictions = run_base_model(examples, args.max_new_tokens)

    # Phase B: Fine-tuned model inference (via Docker HTTP API)
    wait_for_server(args.serve_url)
    ft_predictions = run_finetuned_model(examples, args.serve_url, args.max_new_tokens)

    # Phase C: ROUGE scoring
    console.print("\n[bold cyan]Phase C:[/] Computing ROUGE scores...")
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    results = []
    for i, (ex, base_pred, ft_pred) in enumerate(
        zip(examples, base_predictions, ft_predictions), 1
    ):
        results.append({
            "index": i,
            "instruction": ex["instruction"],
            "reference": ex["reference"],
            "base_prediction": base_pred,
            "finetuned_prediction": ft_pred,
            "base": compute_rouge(ex["reference"], base_pred, scorer),
            "finetuned": compute_rouge(ex["reference"], ft_pred, scorer),
        })

    # Print table and summary
    print_results_table(results)
    print_summary(results)

    # Save full results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\nFull results saved to [bold]{args.output}[/]")


if __name__ == "__main__":
    main()

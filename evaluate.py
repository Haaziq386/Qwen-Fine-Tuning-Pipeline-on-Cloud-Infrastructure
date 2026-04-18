"""
evaluate.py
-----------
Compares the fine-tuned Qwen model against the base model on the held-out
test set, computing BERTScore (default) or ROUGE and printing a side-by-side
comparison.

Prerequisites:
    1. Run `python merge_adapter.py` to create ./models/merged/
    2. Start the inference server: `docker compose up -d`
    3. Run this script: `python evaluate.py`

The fine-tuned model is queried via the running Docker inference server.
The base model is loaded in-process (sequentially, BEFORE querying the server)
to avoid holding two 6 GB models in RAM simultaneously.

Results are saved to evaluation_results.json.

Use command:
python evaluate.py --test-file data/final_data_test.jsonl --max-samples 10 --max-new-tokens 128 --metric bertscore --repetition-penalty 1.2

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
        raise FileNotFoundError(f"Test file not found: {test_file}\n" "Run `python preprocess.py` first to generate the test split.")

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            messages = record.get("messages", [])

            try:
                instruction = next(m["content"] for m in messages if m["role"] == "user")
                reference = next(m["content"] for m in reversed(messages) if m["role"] == "assistant")
            except StopIteration:
                continue  # Skip malformed records

            examples.append({"instruction": instruction, "reference": reference})
            if len(examples) >= max_samples:
                break

    return examples


# ---------- Base model inference (in-process) ----------


def run_base_model(examples: list[dict], max_new_tokens: int, repetition_penalty: float) -> list[str]:
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
                repetition_penalty=repetition_penalty,
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
    raise TimeoutError(f"Inference server did not become ready within {timeout}s.\n" "Make sure you ran: docker compose up -d")


def run_finetuned_model(examples: list[dict], serve_url: str, max_new_tokens: int, repetition_penalty: float) -> list[str]:
    """Query the running Docker inference server for all examples."""
    generate_url = serve_url.rstrip("/") + "/generate"
    predictions = []

    for i, ex in enumerate(examples, 1):
        console.print(f"  [{i}/{len(examples)}] Querying fine-tuned model...", end="\r")
        payload = {
            "prompt": ex["instruction"],
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "repetition_penalty": repetition_penalty,
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


def compute_bertscore_batch(references: list[str], predictions: list[str], model_type: str) -> list[dict]:
    try:
        from bert_score import score as bertscore_score
    except ImportError as exc:
        raise ImportError(
            "BERTScore is not installed. Run: pip install bert-score==0.3.13 "
            "or pip install -r requirements.txt"
        ) from exc

    # bert-score has issues with fully empty hypotheses; substitute placeholders and then zero them out.
    safe_predictions = [p if p.strip() else " " for p in predictions]
    p_scores, r_scores, f1_scores = bertscore_score(
        safe_predictions,
        references,
        lang="en",
        model_type=model_type,
        device="cpu",
        verbose=False,
        rescale_with_baseline=True,
    )

    rows = []
    for pred, p_val, r_val, f1_val in zip(predictions, p_scores, r_scores, f1_scores):
        if not pred.strip():
            rows.append({"precision": 0.0, "recall": 0.0, "f1": 0.0})
            continue
        rows.append(
            {
                "precision": round(float(p_val), 4),
                "recall": round(float(r_val), 4),
                "f1": round(float(f1_val), 4),
            }
        )
    return rows


# ---------- Output ----------


def print_results_table(results: list[dict], metric: str) -> None:
    metric_label = "BERTScore" if metric == "bertscore" else "ROUGE"
    primary_key = "f1" if metric == "bertscore" else "rougeL"
    secondary_key = "precision" if metric == "bertscore" else "rouge1"
    secondary_label = "P" if metric == "bertscore" else "R-1"

    table = Table(title=f"Evaluation: Base vs Fine-Tuned ({metric_label})", show_lines=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Instruction", max_width=30)
    table.add_column("Base Main", justify="right")
    table.add_column("FT Main", justify="right")
    table.add_column(f"Base {secondary_label}", justify="right")
    table.add_column(f"FT {secondary_label}", justify="right")

    for r in results[:20]:  # Show first 20 rows to keep output manageable
        main_diff = r["finetuned"][primary_key] - r["base"][primary_key]
        main_color = "green" if main_diff >= 0 else "red"
        table.add_row(
            str(r["index"]),
            r["instruction"][:50] + ("..." if len(r["instruction"]) > 50 else ""),
            f"{r['base'][primary_key]:.3f}",
            f"[{main_color}]{r['finetuned'][primary_key]:.3f}[/{main_color}]",
            f"{r['base'][secondary_key]:.3f}",
            f"{r['finetuned'][secondary_key]:.3f}",
        )

    console.print(table)


def print_summary(results: list[dict], metric: str) -> None:
    def avg(key: str, model: str) -> float:
        return sum(r[model][key] for r in results) / len(results)

    console.print("\n[bold]Summary (mean over all examples)[/]")
    if metric == "bertscore":
        console.print(
            f"  Base model    — BERTScore P: {avg('precision', 'base'):.4f}  "
            f"R: {avg('recall', 'base'):.4f}  "
            f"F1: {avg('f1', 'base'):.4f}"
        )
        console.print(
            f"  Fine-tuned    — BERTScore P: {avg('precision', 'finetuned'):.4f}  "
            f"R: {avg('recall', 'finetuned'):.4f}  "
            f"F1: {avg('f1', 'finetuned'):.4f}"
        )
        delta_p = avg("precision", "finetuned") - avg("precision", "base")
        delta_r = avg("recall", "finetuned") - avg("recall", "base")
        delta_f1 = avg("f1", "finetuned") - avg("f1", "base")
        color = "green" if delta_f1 >= 0 else "red"
        console.print(f"  [bold {color}]Delta         — BERTScore P: {delta_p:+.4f}  " f"R: {delta_r:+.4f}  " f"F1: {delta_f1:+.4f}[/]")
        return

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
    delta_r1 = avg("rouge1", "finetuned") - avg("rouge1", "base")
    delta_r2 = avg("rouge2", "finetuned") - avg("rouge2", "base")
    delta_rl = avg("rougeL", "finetuned") - avg("rougeL", "base")
    color = "green" if delta_rl >= 0 else "red"
    console.print(f"  [bold {color}]Delta         — ROUGE-1: {delta_r1:+.4f}  " f"ROUGE-2: {delta_r2:+.4f}  " f"ROUGE-L: {delta_rl:+.4f}[/]")


# ---------- Main ----------


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Qwen model vs base model on test JSONL")
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
        "--repetition-penalty",
        type=float,
        default=1.2,
        help="Repetition penalty used for base and fine-tuned generation (default: 1.2)",
    )
    parser.add_argument(
        "--metric",
        choices=["bertscore", "rouge"],
        default="bertscore",
        help="Evaluation metric (default: bertscore)",
    )
    parser.add_argument(
        "--bertscore-model",
        default="distilroberta-base",
        help="Hugging Face model for BERTScore when --metric bertscore (default: distilroberta-base)",
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
    base_predictions = run_base_model(examples, args.max_new_tokens, args.repetition_penalty)

    # Phase B: Fine-tuned model inference (via Docker HTTP API)
    wait_for_server(args.serve_url)
    ft_predictions = run_finetuned_model(examples, args.serve_url, args.max_new_tokens, args.repetition_penalty)

    # Phase C: metric scoring
    metric_name = args.metric.lower()
    console.print(f"\n[bold cyan]Phase C:[/] Computing {metric_name.upper()} scores...")

    if metric_name == "bertscore":
        base_metric_scores = compute_bertscore_batch(
            [ex["reference"] for ex in examples],
            base_predictions,
            args.bertscore_model,
        )
        ft_metric_scores = compute_bertscore_batch(
            [ex["reference"] for ex in examples],
            ft_predictions,
            args.bertscore_model,
        )
    else:
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        base_metric_scores = [compute_rouge(ex["reference"], pred, scorer) for ex, pred in zip(examples, base_predictions)]
        ft_metric_scores = [compute_rouge(ex["reference"], pred, scorer) for ex, pred in zip(examples, ft_predictions)]

    results = []
    for i, (ex, base_pred, ft_pred, base_score, ft_score) in enumerate(
        zip(examples, base_predictions, ft_predictions, base_metric_scores, ft_metric_scores), 1
    ):
        results.append(
            {
                "index": i,
                "instruction": ex["instruction"],
                "reference": ex["reference"],
                "base_prediction": base_pred,
                "finetuned_prediction": ft_pred,
                "metric": metric_name,
                "base": base_score,
                "finetuned": ft_score,
            }
        )

    # Print table and summary
    print_results_table(results, metric_name)
    print_summary(results, metric_name)

    # Save full results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\nFull results saved to [bold]{args.output}[/]")


if __name__ == "__main__":
    main()

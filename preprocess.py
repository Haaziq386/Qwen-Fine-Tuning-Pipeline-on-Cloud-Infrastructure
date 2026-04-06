#!/usr/bin/env python3
"""
Data Preprocessing Service
===========================

Supported output formats:
  - Alpaca:   {"instruction": ..., "input": ..., "output": ...}
  - ChatML:   {"messages": [{"role": ..., "content": ...}, ...]}

Usage:
python preprocess.py --input data/raw_data.jsonl --output data/final_data.jsonl --format chatml 
"""

import json
import random
import hashlib
import logging
import argparse
import re
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from rich import print as rprint

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Validators
# ─────────────────────────────────────────────


class ValidationError(Exception):
    pass


class DataValidator:
    """Validates converted records before writing to JSONL."""

    def __init__(self, fmt: str, max_seq_len: int = 2048, min_output_len: int = 5):
        self.fmt = fmt
        self.max_seq_len = max_seq_len
        self.min_output_len = min_output_len

    def validate(self, record: dict) -> Optional[str]:
        """
        Returns skip reason string if record should be skipped, else None.
        Reason keys match stats dict in preprocess.py.
        """
        if self.fmt == "alpaca":
            return self._validate_alpaca(record)
        elif self.fmt == "chatml":
            return self._validate_chatml(record)
        return None

    def _validate_alpaca(self, rec: dict) -> Optional[str]:
        instruction = rec.get("instruction", "")
        output = rec.get("output", "")

        if not instruction or not output:
            return "skipped_empty"

        if len(output) < self.min_output_len:
            return "skipped_too_short"

        total_len = len(instruction) + len(rec.get("input", "")) + len(output)
        if total_len > self.max_seq_len * 4:  # rough char→token ratio
            return "skipped_too_long"

        return None

    def _validate_chatml(self, rec: dict) -> Optional[str]:
        messages = rec.get("messages", [])
        if not messages:
            return "skipped_empty"

        # Need at least a user and assistant turn
        roles = {m["role"] for m in messages}
        if "user" not in roles or "assistant" not in roles:
            return "skipped_invalid"

        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        if not assistant_msgs or not assistant_msgs[-1].get("content", "").strip():
            return "skipped_empty"

        if len(assistant_msgs[-1]["content"]) < self.min_output_len:
            return "skipped_too_short"

        total_len = sum(len(m.get("content", "")) for m in messages)
        if total_len > self.max_seq_len * 4:
            return "skipped_too_long"

        return None


# ─────────────────────────────────────────────
# Format Converters
# ─────────────────────────────────────────────


def to_alpaca(row: dict[str, Any]) -> dict[str, Any]:
    """
    Alpaca format: standard instruction-following triple.
    Input CSV must have columns: instruction, [input], output
    """
    return {
        "instruction": str(row.get("instruction", row.get("prompt", ""))).strip(),
        "input": str(row.get("input", row.get("context", ""))).strip(),
        "output": str(row.get("output", row.get("response", row.get("completion", "")))).strip(),
    }


def to_chatml(row: dict[str, Any]) -> dict[str, Any]:
    """
    ChatML format: used by Mistral, Qwen, LLaMA-3-Instruct.
    Produces the messages list consumed by apply_chat_template.
    """
    messages = []

    system_prompt = row.get("system", row.get("system_prompt", ""))
    if system_prompt and str(system_prompt).strip():
        messages.append({"role": "system", "content": str(system_prompt).strip()})

    user_content = str(row.get("instruction", row.get("prompt", row.get("user", "")))).strip()
    ctx = str(row.get("input", row.get("context", ""))).strip()
    if ctx:
        user_content = f"{user_content}\n\n{ctx}"

    messages.append({"role": "user", "content": user_content})

    assistant_content = str(row.get("output", row.get("response", row.get("assistant", row.get("completion", ""))))).strip()
    messages.append({"role": "assistant", "content": assistant_content})

    return {"messages": messages}


FORMAT_CONVERTERS = {
    "alpaca": to_alpaca,
    "chatml": to_chatml,
}


# ─────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────


def load_csv(path: str) -> list[dict]:
    df = pd.read_csv(path, encoding="utf-8")
    df = df.fillna("")
    return df.to_dict(orient="records")


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_txt(path: str) -> list[dict]:
    """
    Plain text: each line is a document.
    Wraps into {"instruction": "Summarize:", "input": <line>, "output": ""}
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(
                    {
                        "instruction": "Continue the following text:",
                        "input": line,
                        "output": "",
                    }
                )
    return records


def load_json(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # Try common keys
        for key in ("data", "samples", "examples", "records"):
            if key in data:
                return data[key]
    return [data]


LOADERS = {
    ".csv": load_csv,
    ".jsonl": load_jsonl,
    ".json": load_json,
    ".txt": load_txt,
}


def load_raw_data(path: str) -> list[dict]:
    ext = Path(path).suffix.lower()
    loader = LOADERS.get(ext)
    if not loader:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {list(LOADERS)}")
    logger.info(f"Loading data from {path} (format: {ext})")
    records = loader(path)
    logger.info(f"Loaded {len(records):,} raw records")
    return records


# ─────────────────────────────────────────────
# Cleaning
# ─────────────────────────────────────────────


def clean_text(text: str, max_len: int = 4096) -> str:
    """Basic text cleaning."""
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    # Collapse excessive whitespace / newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    # Truncate if needed
    if len(text) > max_len:
        text = text[:max_len]
    return text


def deduplicate(records: list[dict], key_fields: list[str] | None = None) -> list[dict]:
    """Remove duplicate records by hashing specified fields."""
    seen = set()
    unique = []
    for rec in records:
        if key_fields:
            raw = json.dumps({k: rec.get(k, "") for k in key_fields}, sort_keys=True)
        else:
            raw = json.dumps(rec, sort_keys=True)
        h = hashlib.md5(raw.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(rec)
    removed = len(records) - len(unique)
    if removed:
        logger.info(f"Deduplication removed {removed:,} duplicates")
    return unique


# ─────────────────────────────────────────────
# Main Processing
# ─────────────────────────────────────────────


def process_records(
    records: list[dict],
    fmt: str,
    max_seq_len: int = 2048,
    min_output_len: int = 5,
) -> tuple[list[dict], dict]:
    """Convert and clean records into target format."""
    converter = FORMAT_CONVERTERS[fmt]
    validator = DataValidator(fmt, max_seq_len=max_seq_len, min_output_len=min_output_len)

    processed = []
    stats = {
        "total": len(records),
        "passed": 0,
        "skipped_empty": 0,
        "skipped_too_short": 0,
        "skipped_too_long": 0,
        "skipped_invalid": 0,
    }

    for rec in tqdm(records, desc="Processing records", unit="rec"):
        try:
            converted = converter(rec)

            # Clean text fields
            if fmt == "alpaca":
                converted["instruction"] = clean_text(converted["instruction"])
                converted["input"] = clean_text(converted["input"])
                converted["output"] = clean_text(converted["output"])
            elif fmt == "chatml":
                for msg in converted["messages"]:
                    msg["content"] = clean_text(msg["content"])

            # Validate
            skip_reason = validator.validate(converted)
            if skip_reason:
                stats[skip_reason] = stats.get(skip_reason, 0) + 1
                continue

            processed.append(converted)
            stats["passed"] += 1

        except (ValidationError, Exception) as e:
            stats["skipped_invalid"] += 1
            logger.debug(f"Skipped record due to: {e}")

    return processed, stats


def split_dataset(
    records: list[dict],
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Shuffle and split into train/val/test."""
    assert abs(train + val + test - 1.0) < 1e-6, "Splits must sum to 1.0"
    random.seed(seed)
    shuffled = records.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train)
    n_val = int(n * val)

    train_set = shuffled[:n_train]
    val_set = shuffled[n_train : n_train + n_val]
    test_set = shuffled[n_train + n_val :]

    return train_set, val_set, test_set


def write_jsonl(records: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(records):,} records → {path}")


def print_stats(stats: dict, splits: dict) -> None:
    console.rule("[bold cyan]Preprocessing Complete")

    t = Table(title="Processing Statistics", show_header=True, header_style="bold magenta")
    t.add_column("Metric", style="cyan")
    t.add_column("Count", justify="right", style="green")
    for k, v in stats.items():
        t.add_row(k.replace("_", " ").title(), str(v))
    console.print(t)

    t2 = Table(title="Dataset Splits", show_header=True, header_style="bold magenta")
    t2.add_column("Split", style="cyan")
    t2.add_column("Samples", justify="right", style="green")
    t2.add_column("Output File", style="yellow")
    for split_name, (count, path) in splits.items():
        t2.add_row(split_name, str(count), path)
    console.print(t2)


# ─────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="Convert raw data to instruction-following JSONL format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input", required=True, help="Input file (CSV, JSON, JSONL, TXT)")
    p.add_argument("--output", default="data/train.jsonl", help="Output base path (without _train/_val/_test suffix)")
    p.add_argument("--format", default="alpaca", choices=list(FORMAT_CONVERTERS), help="Output format")
    p.add_argument("--train-split", type=float, default=0.8)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--test-split", type=float, default=0.1)
    p.add_argument("--max-seq-len", type=int, default=2048, help="Max sequence length (chars)")
    p.add_argument("--min-output-len", type=int, default=5, help="Min output length (chars)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-dedup", action="store_true", help="Skip deduplication")
    return p.parse_args()


def main():
    args = parse_args()

    console.rule("[bold cyan]LLM Fine-Tuning Pipeline — Data Preprocessing")
    rprint(f"  Input:    [yellow]{args.input}[/]")
    rprint(f"  Format:   [yellow]{args.format}[/]")
    rprint(f"  Splits:   [yellow]{args.train_split}/{args.val_split}/{args.test_split}[/]")

    # Load
    records = load_raw_data(args.input)

    # Dedup
    if not args.no_dedup:
        records = deduplicate(records)

    # Process
    processed, stats = process_records(
        records,
        fmt=args.format,
        max_seq_len=args.max_seq_len,
        min_output_len=args.min_output_len,
    )

    if not processed:
        console.print("[bold red]ERROR: No valid records after processing. Check your input data.")
        raise SystemExit(1)

    # Split
    train_set, val_set, test_set = split_dataset(
        processed,
        train=args.train_split,
        val=args.val_split,
        test=args.test_split,
        seed=args.seed,
    )

    # Derive output paths
    base = args.output.replace(".jsonl", "").replace("_train", "")
    train_path = f"{base}_train.jsonl"
    val_path = f"{base}_val.jsonl"
    test_path = f"{base}_test.jsonl"

    write_jsonl(train_set, train_path)
    write_jsonl(val_set, val_path)
    write_jsonl(test_set, test_path)

    # Print summary
    splits = {
        "Train": (len(train_set), train_path),
        "Val": (len(val_set), val_path),
        "Test": (len(test_set), test_path),
    }
    print_stats(stats, splits)

    # Write a sample to stdout for verification
    if train_set:
        console.rule("[bold cyan]Sample Record (Train)")
        rprint(json.dumps(train_set[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

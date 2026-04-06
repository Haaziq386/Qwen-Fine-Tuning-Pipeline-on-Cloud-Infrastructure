"""
merge_adapter.py
----------------
Downloads the LoRA adapter from S3, merges it into the base Qwen model,
and saves the merged model locally for use by the inference server.

Run once before starting the inference server:
    python merge_adapter.py

The merged model is saved to ./models/merged/ (or --output-dir).
"""

import argparse
import gc
import os
import tempfile
from pathlib import Path

import boto3
import torch
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_S3_URI = "s3://cloud-project-model-registery/models/qwen-1.5b-finetuned/v20260406_095050"
DEFAULT_OUTPUT_DIR = "./models/merged"


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Split s3://bucket/prefix into (bucket, prefix)."""
    uri = uri.rstrip("/")
    assert uri.startswith("s3://"), f"Not an S3 URI: {uri}"
    parts = uri[5:].split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket, prefix


def download_adapter(s3_uri: str, local_dir: Path) -> None:
    """Download all adapter files from S3 to local_dir."""
    bucket_name, prefix = parse_s3_uri(s3_uri)

    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION", "ap-south-1"),
    )

    print(f"Listing objects at s3://{bucket_name}/{prefix}/")
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix + "/")

    files_downloaded = 0
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            # Preserve relative path inside the prefix
            relative = key[len(prefix):].lstrip("/")
            if not relative:
                continue
            dest = local_dir / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            print(f"  Downloading {key} → {dest}")
            s3.download_file(bucket_name, key, str(dest))
            files_downloaded += 1

    if files_downloaded == 0:
        raise RuntimeError(f"No files found at {s3_uri}. Check the S3 path and credentials.")
    print(f"Downloaded {files_downloaded} file(s) to {local_dir}")


def merge_and_save(adapter_dir: Path, output_dir: Path) -> None:
    """
    Load the base model in float32 (no quantization), apply the LoRA adapter,
    merge weights, and save the resulting full model.

    IMPORTANT: merge_and_unload() requires the base model to be in full
    precision. Never use BitsAndBytesConfig here — the quantization was only
    needed during training to fit on a small GPU.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading base model: {BASE_MODEL_ID}")
    print("(This downloads ~3 GB from HuggingFace on first run)")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token  # Qwen has no pad_token by default

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float32,
        device_map="cpu",
    )

    print(f"\nApplying LoRA adapter from {adapter_dir}")
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))

    print("Merging adapter weights into base model...")
    merged_model = peft_model.merge_and_unload()

    # Free the PEFT wrapper to recover memory before saving
    del peft_model
    gc.collect()

    print(f"\nSaving merged model to {output_dir}")
    merged_model.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))

    del merged_model
    gc.collect()

    print(f"\nDone. Merged model saved to: {output_dir.resolve()}")
    print("You can now run: docker compose up -d")


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter from S3 into base model")
    parser.add_argument(
        "--s3-path",
        default=DEFAULT_S3_URI,
        help=f"S3 URI of the adapter directory (default: {DEFAULT_S3_URI})",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Local path to save the merged model (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--adapter-cache",
        default="./models/adapter_tmp",
        help="Temporary local path for downloaded adapter files (default: ./models/adapter_tmp)",
    )
    args = parser.parse_args()

    adapter_dir = Path(args.adapter_cache)
    output_dir = Path(args.output_dir)

    # Check credentials
    if not os.getenv("AWS_ACCESS_KEY_ID"):
        raise EnvironmentError("AWS_ACCESS_KEY_ID not set. Check your .env file.")

    download_adapter(args.s3_path, adapter_dir)
    merge_and_save(adapter_dir, output_dir)


if __name__ == "__main__":
    main()

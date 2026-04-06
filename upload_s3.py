#!/usr/bin/env python3
"""
AWS S3 Model Registry Script
============================
Uploads the local LoRA adapter weights to an AWS S3 bucket.
Organizes uploads using a timestamp-based versioning system.

Usage:
    python register_model.py --model-dir ./final_adapter
"""

import os
import argparse
from datetime import datetime
from pathlib import Path

import boto3
from dotenv import load_dotenv
from rich.console import Console
from rich import print as rprint

console = Console()


def upload_directory_to_s3(local_dir: str, bucket_name: str, s3_prefix: str):
    """Recursively uploads a local directory to an S3 bucket."""
    s3_client = boto3.client("s3", aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"), 
                             aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"), 
                             region_name=os.getenv("AWS_REGION"))

    local_path = Path(local_dir)
    if not local_path.exists() or not local_path.is_dir():
        console.print(f"[bold red]Error: Directory '{local_dir}' does not exist.[/]")
        return False

    uploaded_files = 0
    console.print(f"[cyan]Uploading {local_dir} to s3://{bucket_name}/{s3_prefix}/[/]")

    for root, _, files in os.walk(local_dir):
        for file in files:
            local_file_path = os.path.join(root, file)
            # Calculate the relative path to maintain folder structure in S3
            relative_path = os.path.relpath(local_file_path, local_dir)
            s3_key = os.path.join(s3_prefix, relative_path).replace("\\", "/")

            try:
                s3_client.upload_file(local_file_path, bucket_name, s3_key)
                rprint(f"  [green]✓[/] Uploaded: {file}")
                uploaded_files += 1
            except Exception as e:
                console.print(f"  [bold red]✗ Failed to upload {file}: {e}[/]")
                return False

    console.print(f"\n[bold green]Success! {uploaded_files} files registered in S3.[/]")
    return True


def main():
    # Load environment variables from .env file
    load_dotenv()

    parser = argparse.ArgumentParser(description="Push model artifacts to AWS S3")
    parser.add_argument("--model-dir", required=True, help="Local directory containing the LoRA adapter")
    parser.add_argument("--model-name", default="qwen-1.5b-finetuned", help="Name of the model project")
    args = parser.parse_args()

    bucket_name = os.getenv("S3_BUCKET_NAME")
    if not bucket_name:
        console.print("[bold red]Error: S3_BUCKET_NAME not found in .env file.[/]")
        return

    # Create a unique version tag based on the current date and time
    version_tag = datetime.now().strftime("v%Y%m%d_%H%M%S")
    s3_prefix = f"models/{args.model_name}/{version_tag}"

    console.rule("[bold cyan]AWS S3 Model Registry[/]")
    upload_directory_to_s3(args.model_dir, bucket_name, s3_prefix)


if __name__ == "__main__":
    main()

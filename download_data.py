"""
Download Databricks Dolly 15k dataset.
Usage:
python download_data.py --output data/raw_data.jsonl
"""

from datasets import load_dataset
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="Output file path")
    args = parser.parse_args()

    print("Downloading Databricks Dolly 15k...")
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    dataset.to_json(args.output)
    print(f"Saved to {args.output}")

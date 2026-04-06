"""
This script registers the fine-tuned Qwen 1.5B LoRA adapter in MLflow's model registry,
linking it to the original Kaggle training run.

Run using:
    python register_model.py
"""

import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime

# --- Configuration ---
MLFLOW_TRACKING_URI = "file:./training_output/mlruns"
EXPERIMENT_NAME = "qwen-1.5b-chatml-lora"
MODEL_NAME = "qwen-1.5b-lora-adapter"
EXISTING_RUN_ID = "bb0586c1b23e4e5aa4f2ba9194e60698"

# S3 URI where the model weights are stored
S3_URI = "s3://cloud-project-model-registery/models/qwen-1.5b-finetuned/v20260406_095050"


def register_in_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    # 1. Create registered model if it doesn't exist
    try:
        client.create_registered_model(
            name=MODEL_NAME,
            description="QLoRA adapter for Qwen 1.5B Instruct",
        )
        print(f"Created registered model: {MODEL_NAME}")
    except mlflow.exceptions.MlflowException:
        print(f"Model already exists: {MODEL_NAME}")

    # 2. Register model version
    print("Registering version...")
    version = client.create_model_version(
        name=MODEL_NAME,
        source=S3_URI,
        run_id=EXISTING_RUN_ID,  # Links it to your training metrics
        description=f"Uploaded to S3 on {datetime.utcnow().strftime('%Y-%m-%d')}",
    )

    # 3. Transition to "Staging"
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version.version,
        stage="Staging",
    )

    print("-" * 40)
    print(f"✅ Success! Registered: {MODEL_NAME} version {version.version}")
    print(f"📍 Linked to Kaggle Run ID: {EXISTING_RUN_ID}")
    print("-" * 40)


if __name__ == "__main__":
    register_in_mlflow()

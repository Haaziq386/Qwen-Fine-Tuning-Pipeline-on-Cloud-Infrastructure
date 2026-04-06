"""
This script registers the fine-tuned Qwen 1.5B LoRA adapter in MLflow's model registry.
Run using: python register_model.py
"""

import json
from pathlib import Path
from datetime import datetime
import mlflow
from mlflow.tracking import MlflowClient

# --- Configuration ---
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000" # Pointing to our SQLite-backed server
EXPERIMENT_NAME = "qwen-1.5b-chatml-lora"
MODEL_NAME = "qwen-1.5b-lora-adapter"
LOCAL_ADAPTER_DIR = "./training_output/qwen-lora/final_adapter"

# Replace this with the EXACT S3 URI your upload script outputted
S3_URI = "s3://cloud-project-model-registery/models/qwen-1.5b-finetuned/v20260406_095050" 

def register_in_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
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

    # 2. Start a run to attach metadata
    print("Logging metadata...")
    with mlflow.start_run(run_name="adapter-registration") as run:
        mlflow.log_params({
            "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
            "s3_uri": S3_URI,
            "upload_time": datetime.utcnow().isoformat(),
        })

        # Read adapter config to log hyperparams
        config_file = Path(LOCAL_ADAPTER_DIR) / "adapter_config.json"
        if config_file.exists():
            with open(config_file) as f:
                adapter_cfg = json.load(f)
            mlflow.log_params({
                "lora_r": adapter_cfg.get("r", ""),
                "lora_alpha": adapter_cfg.get("lora_alpha", ""),
                "lora_target_modules": str(adapter_cfg.get("target_modules", "")),
            })

        run_id = run.info.run_id

    # 3. Register model version pointing to S3
    print("Registering version...")
    version = client.create_model_version(
        name=MODEL_NAME,
        source=S3_URI,
        run_id=run_id,
        description=f"Uploaded to S3 on {datetime.utcnow().strftime('%Y-%m-%d')}",
    )
    
    # 4. Transition to "Staging"
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version.version,
        stage="Staging",
    )

    print("-" * 40)
    print(f"✅ Success! Registered: {MODEL_NAME} version {version.version}")
    print(f"📍 Stage: Staging")
    print(f"🔗 MLflow run ID: {run_id}")
    print("-" * 40)

if __name__ == "__main__":
    register_in_mlflow()

import os
import time
import torch
import mlflow
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer, SFTConfig


class Config:
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    train_file = "/kaggle/input/datasets/roopamtaneja/qwen-finetuning-dolly-dataset/final_data_train.jsonl"
    val_file = "/kaggle/input/datasets/roopamtaneja/qwen-finetuning-dolly-dataset/final_data_val.jsonl"
    output_dir = "/kaggle/working/qwen-lora"

    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    epochs = 3
    batch_size = 2
    grad_accum_steps = 8
    learning_rate = 2e-4
    max_seq_length = 1024
    warmup_steps = 50

    experiment_name = "qwen-1.5b-chatml-lora"
    tracking_uri = "file:///kaggle/working/mlruns"


class MLflowCallback(TrainerCallback):
    def __init__(self, cfg):
        self.cfg = cfg
        self._start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        mlflow.set_tracking_uri(self.cfg.tracking_uri)
        mlflow.set_experiment(self.cfg.experiment_name)
        mlflow.start_run(run_name="qlora_run")
        self._start_time = time.time()
        mlflow.set_tags(
            {
                "model": self.cfg.model_id,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
                "format": "chatml",
            }
        )
        mlflow.log_params(
            {
                "epochs": self.cfg.epochs,
                "learning_rate": self.cfg.learning_rate,
                "lora_r": self.cfg.lora_r,
                "lora_alpha": self.cfg.lora_alpha,
                "batch_size": self.cfg.batch_size,
                "grad_accum_steps": self.cfg.grad_accum_steps,
                "warmup_steps": self.cfg.warmup_steps,
            }
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            metrics = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
            if torch.cuda.is_available():
                metrics["gpu_allocated_gb"] = round(torch.cuda.memory_allocated(0) / (1024**3), 2)
            mlflow.log_metrics(metrics, step=state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        mlflow.log_metric("total_time_minutes", (time.time() - self._start_time) / 60)
        mlflow.end_run()


def main():
    print(f"Loading Model: {Config.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(Config.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Loading datasets...")
    dataset = load_dataset(
        "json",
        data_files={
            "train": Config.train_file,
            "val": Config.val_file,
        },
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
    )

    print(f"Loading model: {Config.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        Config.model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.use_cache = False

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=Config.lora_r,
        lora_alpha=Config.lora_alpha,
        target_modules=Config.target_modules,
        lora_dropout=Config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = SFTConfig(
        output_dir=Config.output_dir,
        num_train_epochs=Config.epochs,
        per_device_train_batch_size=Config.batch_size,
        per_device_eval_batch_size=Config.batch_size,
        gradient_accumulation_steps=Config.grad_accum_steps,
        learning_rate=Config.learning_rate,
        optim="paged_adamw_32bit",
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=False,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        warmup_steps=Config.warmup_steps,
        lr_scheduler_type="cosine",
        report_to="none",
        gradient_checkpointing=True,
        max_length=Config.max_seq_length,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        args=training_args,
        callbacks=[MLflowCallback(Config)],
    )

    print("Starting Training...")
    trainer.train()

    adapter_path = os.path.join(Config.output_dir, "final_adapter")
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"Training Complete! Adapter saved to: {adapter_path}")


if __name__ == "__main__":
    main()


# Kaggle cell after this script to download output:
# import os
# import shutil
# from IPython.display import FileLink

# # 1. Define the name of your output zip file
# output_filename = "all_kaggle_outputs"

# # 2. Create a zip archive of the /kaggle/working directory
# # This will save 'all_kaggle_outputs.zip' in your current directory
# shutil.make_archive(output_filename, 'zip', '/kaggle/working')

# # 3. Create a clickable link to download the file
# FileLink(f"{output_filename}.zip")

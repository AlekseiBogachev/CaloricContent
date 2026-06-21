import logging
import random
from datetime import datetime
from functools import partial
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed as transformers_set_seed,
)

from caloric_content.caloric_model import CaloricModel, CaloricModelConfig
from caloric_content.data import (
    apply_transforms_to_dataset,
    collate_fn,
    get_dataset,
)

logger = logging.getLogger(__name__)


def set_random_state(seed=42):
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    transformers_set_seed(seed)
    logger.info(f"Set random seed to {seed}")


def get_optimizer(config, model):
    image_backbone_lr = config["training"]["image_backbone_lr"]
    lr = config["training"]["lr"]

    image_backbone_params = list()
    other_params = list()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "image_encoder" in name:
            image_backbone_params.append(param)
        else:
            other_params.append(param)

    return torch.optim.AdamW(
        [
            {"params": image_backbone_params, "lr": image_backbone_lr},
            {"params": other_params, "lr": lr},
        ],
        weight_decay=config["training"]["weight_decay"],
    )


class CSVLoggerCallback(TrainerCallback):
    def __init__(self, csv_path="metrics.csv"):
        self.csv_path = csv_path

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            df = pd.DataFrame([{"step": state.global_step, **logs}])
            if Path(self.csv_path).exists():
                df.to_csv(self.csv_path, mode="a", header=False, index=False)
            else:
                df.to_csv(self.csv_path, mode="w", header=True, index=False)


def run_training(config):
    logger.info("Start training CaloricModel")

    mlflow.set_tracking_uri(config["training"]["MLFlow"]["tracking_uri"])
    mlflow.set_experiment(config["training"]["MLFlow"]["experiment_name"])
    run_name = config["training"]["MLFlow"]["run_name"]
    logger.info(
        "Setup logging to MLFlow "
        f"tracking_uri={config['training']['MLFlow']['tracking_uri']} "
        f"Experiment={config['training']['MLFlow']['experiment_name']}"
        f"base_run_name={run_name}"
    )

    device = config["training"]["device"]
    logger.info(f"Selected device is '{device}'")
    if (device == "cuda") and (not torch.cuda.is_available()):
        device = "cpu"
        logger.warning("Cuda is not available. Set device='cpu'.")

    dataset = get_dataset(config)
    train_dataset = apply_transforms_to_dataset(
        dataset["train"], config, split="train"
    )
    valid_dataset = apply_transforms_to_dataset(
        dataset["valid"], config, split="test"
    )
    test_dataset = apply_transforms_to_dataset(
        dataset["test"], config, split="test"
    )

    model_config = CaloricModelConfig(**config["model"])
    model = CaloricModel(model_config)

    optimizer = get_optimizer(config, model)
    logger.info("Inintialized optimizer.")

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["text_backbone"])
    logger.info(
        f"Initialized tokenizer from {config['model']['text_backbone']}"
    )

    training_args = TrainingArguments(
        output_dir=config["models_dir"],
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=5,
        logging_strategy="steps",
        logging_steps=100,
        learning_rate=config["training"]["lr"],
        per_device_train_batch_size=config["training"]["batch_size"],
        per_device_eval_batch_size=config["training"]["batch_size"],
        num_train_epochs=config["training"]["epochs"],
        weight_decay=config["training"]["weight_decay"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="mlflow",
        push_to_hub=False,
        seed=config.SEED,
        data_seed=config.SEED,
        dataloader_num_workers=config["training"]["num_workers"],
        use_cpu=(device == "cpu"),
        lr_scheduler_type="cosine",
        warmup_ratio=config["training"]["warmup_ratio"],
    )

    # model.to(device)

    time_str = datetime.now().strftime("%Y_%m_%dT%H_%M_%S")
    run_name = f"{run_name}_{time_str}"
    csv_path = Path(config["metrics_dir"]).joinpath(
        f"training_logs_{run_name}.csv"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        optimizers=(optimizer, None),
        data_collator=partial(collate_fn, tokenizer=tokenizer),
        callbacks=[
            CSVLoggerCallback(csv_path=csv_path),
        ],
    )

    logger.info("Initialized Trainer.")

    with mlflow.start_run(run_name=run_name):
        logger.info(f"Started MLFlow run '{run_name}'")
        logger.info("Start model training.")

        mlflow.log_dict(config, "config.json")
        logger.debug("MLFLow logged config dict")

        trainer.train()
        logger.info("Training is finished.")

        best_models_path = Path(config["models_dir"]).joinpath(
            f"best_{run_name}"
        )
        trainer.save_model(best_models_path)
        logger.info(f"Best model saved to {best_models_path}")
        mlflow.log_artifacts(best_models_path, "best_checkpoint")
        logger.debug(f"MLFlow logged best checkpoint from {best_models_path}")

        log_history_df = pd.DataFrame(trainer.state.log_history)
        log_history_path = Path(config["metrics_dir"]).joinpath(
            f"log_history_{run_name}.csv"
        )
        log_history_df.to_csv(
            log_history_path,
            index=False,
        )
        logger.info(f"Saved log history to {log_history_path}")
        mlflow.log_artifact(log_history_path)
        logger.debug(f"MLFlow logged log history {log_history_path}")

        logger.info("Start model test.")
        test_results = trainer.evaluate(
            eval_dataset=test_dataset, metric_key_prefix="test"
        )
        logger.info("Test is finished.")
        test_results_path = Path(config["metrics_dir"]).joinpath(
            f"test_results_{run_name}.csv"
        )
        pd.Series(test_results).to_csv(test_results_path)
        logger.info(f"Test results is saved to {test_results_path}")
        mlflow.log_artifact(test_results_path)
        logger.debug(f"MLFlow logged test results {test_results_path}")


def run_test(config):
    raise NotImplementedError


def run_predict(config):
    raise NotImplementedError

"""Stage 4: Model Training Component.

Fine-tunes facebook/bart-large-cnn on the SAMSum dataset with safeguards
against overfitting and a local smoke-training mode.

Training highlights:
    - Label smoothing to reduce over-confidence
    - Gradient clipping for training stability
    - Early stopping (patience + threshold) for overfitting control
    - Seeded training for reproducibility
    - Smoke mode for fast local validation before Kaggle runs
"""

import json
import os
import inspect
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
)

from textSummarizer.entity import ModelTrainerConfig
from textSummarizer.logging import logger


class ModelTrainer:
    """Fine-tunes a seq2seq model for dialogue summarization."""

    SMOKE_MODEL_CKPT = "hf-internal-testing/tiny-random-bart"
    SMOKE_MAX_INPUT_LENGTH = 64
    SMOKE_MAX_TARGET_LENGTH = 24
    SMOKE_MAX_TRAIN_SAMPLES = 16
    SMOKE_MAX_EVAL_SAMPLES = 8

    def __init__(self, config: ModelTrainerConfig) -> None:
        self.config = config

    def _resolve_device(self) -> str:
        """Resolve training device with explicit logging."""
        if torch.cuda.is_available():
            logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
            return "cuda"

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Using Apple MPS")
            return "mps"

        logger.info("Using CPU (training will be slow)")
        return "cpu"

    def _load_model_and_tokenizer(self, device: str, smoke_train: bool = False) -> Tuple[Any, Any, str]:
        """Load model/tokenizer with fallback from base checkpoint to local model."""
        local_fallback = Path(self.config.root_dir) / "bart-samsum-model"
        checkpoint_candidates = []

        if smoke_train:
            checkpoint_candidates.append(self.SMOKE_MODEL_CKPT)

        checkpoint_candidates.append(self.config.model_ckpt)
        if local_fallback.exists():
            checkpoint_candidates.append(str(local_fallback))

        # Deduplicate while preserving order.
        checkpoint_candidates = list(dict.fromkeys(checkpoint_candidates))

        last_error: Exception | None = None
        for checkpoint in checkpoint_candidates:
            try:
                if smoke_train and checkpoint == self.SMOKE_MODEL_CKPT:
                    logger.info(f"Loading lightweight smoke checkpoint: {checkpoint}")
                else:
                    logger.info(f"Loading model checkpoint: {checkpoint}")
                tokenizer = AutoTokenizer.from_pretrained(checkpoint)
                model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)
                return tokenizer, model, checkpoint
            except Exception as exc:  # pragma: no cover - depends on local cache/network
                last_error = exc
                logger.warning(f"Checkpoint load failed for {checkpoint}: {exc}")

        raise RuntimeError(
            "Unable to load any training checkpoint. "
            "Check internet access or ensure artifacts/model_trainer/bart-samsum-model exists."
        ) from last_error

    def _tokenize_raw_dataset(
        self,
        raw_dataset: DatasetDict,
        tokenizer: Any,
        max_input_length: int | None = None,
        max_target_length: int | None = None,
    ) -> DatasetDict:
        """Tokenize a raw dialogue dataset into model-ready tensors."""
        max_input = max_input_length or self.config.max_input_length
        max_target = max_target_length or self.config.max_target_length

        def _preprocess(batch: Dict[str, Any]) -> Dict[str, Any]:
            model_inputs = tokenizer(
                batch[self.config.text_column],
                max_length=max_input,
                truncation=True,
                padding="max_length",
            )
            labels = tokenizer(
                text_target=batch[self.config.summary_column],
                max_length=max_target,
                truncation=True,
                padding="max_length",
            )
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        return raw_dataset.map(
            _preprocess,
            batched=True,
            remove_columns=raw_dataset["train"].column_names,
            desc="Tokenizing smoke dataset",
        )

    def _build_synthetic_smoke_dataset(self) -> DatasetDict:
        """Build a tiny synthetic dataset so smoke training can run fully offline."""
        total = min(max(self.config.smoke_train_samples, 8), self.SMOKE_MAX_TRAIN_SAMPLES)
        eval_total = min(max(self.config.smoke_eval_samples, 4), self.SMOKE_MAX_EVAL_SAMPLES)

        train_dialogues = [
            (
                f"Alice: We need to submit milestone {i} today. "
                f"Bob: I will finish the draft and send it in 30 minutes. "
                f"Alice: Great, I will review and upload it."
            )
            for i in range(total)
        ]
        train_summaries = [
            f"Alice and Bob coordinate milestone {i} submission and split drafting and review tasks."
            for i in range(total)
        ]

        eval_dialogues = [
            (
                f"Priya: Can we move standup {i} to 11 AM? "
                f"Rahul: Yes, 11 AM works for me. "
                f"Priya: Perfect, I will update the calendar."
            )
            for i in range(eval_total)
        ]
        eval_summaries = [
            f"Priya and Rahul agree to reschedule standup {i} to 11 AM."
            for i in range(eval_total)
        ]

        return DatasetDict(
            {
                "train": Dataset.from_dict(
                    {
                        "id": [f"synthetic-train-{i}" for i in range(total)],
                        self.config.text_column: train_dialogues,
                        self.config.summary_column: train_summaries,
                    }
                ),
                "validation": Dataset.from_dict(
                    {
                        "id": [f"synthetic-val-{i}" for i in range(eval_total)],
                        self.config.text_column: eval_dialogues,
                        self.config.summary_column: eval_summaries,
                    }
                ),
            }
        )

    def _load_training_dataset(self, tokenizer: Any, smoke_train: bool) -> DatasetDict:
        """Load tokenized dataset from artifacts, or build a tiny smoke dataset if needed."""
        if smoke_train:
            logger.info(
                "Smoke mode: using a tiny synthetic dataset with short sequence lengths "
                "for deterministic local training validation."
            )
            raw_subset = self._build_synthetic_smoke_dataset()
            return self._tokenize_raw_dataset(
                raw_subset,
                tokenizer,
                max_input_length=min(self.config.max_input_length, self.SMOKE_MAX_INPUT_LENGTH),
                max_target_length=min(self.config.max_target_length, self.SMOKE_MAX_TARGET_LENGTH),
            )

        data_path = Path(self.config.data_path)
        if data_path.exists() and any(data_path.iterdir()):
            logger.info(f"Loading tokenized dataset from: {data_path}")
            return load_from_disk(str(data_path))

        raise FileNotFoundError(
            f"Tokenized dataset not found at {data_path}. Run stages 1-3 first."
        )

    def _write_training_diagnostics(
        self,
        trainer: Trainer,
        checkpoint_used: str,
        smoke_train: bool,
        device: str,
    ) -> None:
        """Write a compact training diagnostics artifact for reproducibility."""
        train_losses = [x["loss"] for x in trainer.state.log_history if "loss" in x]
        eval_losses = [x["eval_loss"] for x in trainer.state.log_history if "eval_loss" in x]

        diagnostics: Dict[str, Any] = {
            "smoke_train": smoke_train,
            "checkpoint_used": checkpoint_used,
            "device": device,
            "global_step": int(trainer.state.global_step),
            "best_metric": (
                float(trainer.state.best_metric)
                if trainer.state.best_metric is not None
                else None
            ),
            "final_train_loss": float(train_losses[-1]) if train_losses else None,
            "min_eval_loss": float(min(eval_losses)) if eval_losses else None,
            "config": {
                "learning_rate": self.config.learning_rate,
                "label_smoothing_factor": self.config.label_smoothing_factor,
                "max_grad_norm": self.config.max_grad_norm,
                "lr_scheduler_type": self.config.lr_scheduler_type,
                "num_train_epochs": self.config.num_train_epochs,
                "seed": self.config.seed,
            },
        }

        final_train_loss = diagnostics["final_train_loss"]
        min_eval_loss = diagnostics["min_eval_loss"]
        if final_train_loss is not None and min_eval_loss is not None:
            overfit_gap = min_eval_loss - final_train_loss
            diagnostics["train_eval_loss_gap"] = float(overfit_gap)
            if overfit_gap > 0.5:
                logger.warning(
                    "Potential overfitting detected: "
                    f"train/eval loss gap={overfit_gap:.4f}."
                )

        diagnostics_path = Path(self.config.root_dir) / "training_diagnostics.json"
        with open(diagnostics_path, "w", encoding="utf-8") as f:
            json.dump(diagnostics, f, indent=2)
        logger.info(f"Training diagnostics written to: {diagnostics_path}")

    def train(self, smoke_train: bool = False) -> None:
        """Execute model training with full configuration.

        Workflow:
        1. Detect device (CUDA/MPS/CPU)
        2. Load tokenizer and model from checkpoint
        3. Configure training arguments with checkpointing
        4. Train with early stopping callback
        5. Save best model and tokenizer
        """
        device = self._resolve_device()

        tokenizer, model, checkpoint_used = self._load_model_and_tokenizer(
            device,
            smoke_train=smoke_train,
        )

        # Use gradient checkpointing for full training only.
        if not smoke_train:
            model.gradient_checkpointing_enable()

        # Data collator handles dynamic padding
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            padding=True,
            label_pad_token_id=tokenizer.pad_token_id,
        )

        dataset = self._load_training_dataset(tokenizer, smoke_train=smoke_train)

        # Determine fp16 availability
        use_fp16 = self.config.fp16 and device == "cuda"
        if device == "cpu" or os.name == "nt":
            dataloader_num_workers = 0
        else:
            dataloader_num_workers = self.config.dataloader_num_workers

        train_examples = len(dataset["train"])
        eval_examples = len(dataset["validation"])
        if train_examples < 4 or eval_examples < 2:
            raise ValueError(
                "Dataset too small for stable training. "
                f"train={train_examples}, validation={eval_examples}"
            )

        training_kwargs: Dict[str, Any] = {
            "output_dir": str(self.config.root_dir),
            "num_train_epochs": self.config.num_train_epochs,
            "warmup_steps": self.config.warmup_steps,
            "per_device_train_batch_size": self.config.per_device_train_batch_size,
            "per_device_eval_batch_size": self.config.per_device_eval_batch_size,
            "weight_decay": self.config.weight_decay,
            "logging_steps": self.config.logging_steps,
            "eval_strategy": self.config.eval_strategy,
            "eval_steps": self.config.eval_steps,
            "save_steps": self.config.save_steps,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "learning_rate": self.config.learning_rate,
            "fp16": use_fp16,
            "save_total_limit": self.config.save_total_limit,
            "load_best_model_at_end": self.config.load_best_model_at_end,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "report_to": self.config.report_to,
            "logging_dir": os.path.join(str(self.config.root_dir), "logs"),
            "dataloader_num_workers": dataloader_num_workers,
            "seed": self.config.seed,
            "max_grad_norm": self.config.max_grad_norm,
            "label_smoothing_factor": self.config.label_smoothing_factor,
            "lr_scheduler_type": self.config.lr_scheduler_type,
            "group_by_length": self.config.group_by_length,
        }

        if smoke_train:
            smoke_max_steps = max(1, min(self.config.smoke_max_steps, 2))
            training_kwargs.update(
                {
                    "num_train_epochs": 1,
                    "max_steps": smoke_max_steps,
                    "logging_steps": 1,
                    "eval_steps": 1,
                    "save_steps": max(smoke_max_steps + 1, 10),
                    "save_total_limit": 1,
                    "load_best_model_at_end": False,
                    "per_device_train_batch_size": 1,
                    "per_device_eval_batch_size": 1,
                    "gradient_accumulation_steps": 1,
                    "warmup_steps": 0,
                    "group_by_length": False,
                    "label_smoothing_factor": 0.0,
                }
            )

        supported_args = set(inspect.signature(TrainingArguments.__init__).parameters)
        if "eval_strategy" in training_kwargs and "eval_strategy" not in supported_args:
            if "evaluation_strategy" in supported_args:
                training_kwargs["evaluation_strategy"] = training_kwargs.pop("eval_strategy")
            else:
                training_kwargs.pop("eval_strategy")

        unsupported_args = [k for k in training_kwargs if k not in supported_args]
        for arg in unsupported_args:
            logger.warning(
                "TrainingArguments compatibility: dropping unsupported arg "
                f"'{arg}' for installed transformers version"
            )
            training_kwargs.pop(arg)

        training_args = TrainingArguments(**training_kwargs)

        callbacks = []
        if not smoke_train:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.early_stopping_patience,
                    early_stopping_threshold=self.config.early_stopping_threshold,
                )
            )

        trainer_kwargs: Dict[str, Any] = {
            "model": model,
            "args": training_args,
            "data_collator": data_collator,
            "train_dataset": dataset["train"],
            "eval_dataset": dataset["validation"],
            "callbacks": callbacks,
        }
        trainer_signature = set(inspect.signature(Trainer.__init__).parameters)
        if "tokenizer" in trainer_signature:
            trainer_kwargs["tokenizer"] = tokenizer
        elif "processing_class" in trainer_signature:
            trainer_kwargs["processing_class"] = tokenizer

        trainer = Trainer(**trainer_kwargs)

        effective_epochs = training_args.num_train_epochs
        effective_train_batch = training_args.per_device_train_batch_size
        effective_grad_accum = training_args.gradient_accumulation_steps
        effective_label_smoothing = getattr(
            training_args, "label_smoothing_factor", self.config.label_smoothing_factor
        )
        effective_scheduler = getattr(training_args, "lr_scheduler_type", self.config.lr_scheduler_type)

        logger.info("Starting training...")
        logger.info(f"  Smoke mode: {smoke_train}")
        logger.info(f"  Checkpoint: {checkpoint_used}")
        logger.info(f"  Epochs: {effective_epochs}")
        logger.info(f"  Batch size: {effective_train_batch}")
        logger.info(f"  Grad accum steps: {effective_grad_accum}")
        logger.info(f"  Effective batch size: {effective_train_batch * effective_grad_accum}")
        logger.info(f"  FP16: {use_fp16}")
        logger.info(f"  Label smoothing: {effective_label_smoothing}")
        logger.info(f"  Max grad norm: {self.config.max_grad_norm}")
        logger.info(f"  LR scheduler: {effective_scheduler}")
        logger.info(f"  Train examples: {train_examples}")
        logger.info(f"  Eval examples: {eval_examples}")

        trainer.train()
        self._write_training_diagnostics(
            trainer=trainer,
            checkpoint_used=checkpoint_used,
            smoke_train=smoke_train,
            device=device,
        )

        if smoke_train:
            logger.info("Smoke training completed successfully. Skipping model persistence.")
            return

        model_save_path = os.path.join(str(self.config.root_dir), "bart-samsum-model")
        tokenizer_save_path = os.path.join(str(self.config.root_dir), "tokenizer")

        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(tokenizer_save_path)

        logger.info(f"Model saved to: {model_save_path}")
        logger.info(f"Tokenizer saved to: {tokenizer_save_path}")

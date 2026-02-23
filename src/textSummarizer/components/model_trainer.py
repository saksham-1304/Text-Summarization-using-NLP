"""Stage 4: Model Training Component.

Fine-tunes facebook/bart-large-cnn on the SAMSum dataset.

Training features:
  - Mixed precision (fp16) for faster training on GPUs
  - Gradient accumulation for effective larger batch sizes
  - Checkpoint saving at regular intervals
  - Early stopping based on eval loss via load_best_model_at_end
  - Proper learning rate scheduling with warmup

Recommended: Train on GPU (Kaggle T4/P100 or Colab A100).
On CPU, training will be very slow. Use the Kaggle notebook for full training.
"""

import os
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from datasets import load_from_disk
from textSummarizer.logging import logger
from textSummarizer.entity import ModelTrainerConfig


class ModelTrainer:
    """Fine-tunes a seq2seq model for dialogue summarization."""

    def __init__(self, config: ModelTrainerConfig) -> None:
        self.config = config

    def train(self) -> None:
        """Execute model training with full configuration.

        Workflow:
        1. Detect device (CUDA/MPS/CPU)
        2. Load tokenizer and model from checkpoint
        3. Configure training arguments with checkpointing
        4. Train with early stopping callback
        5. Save best model and tokenizer
        """
        # Device selection
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            logger.info("Using Apple MPS")
        else:
            device = "cpu"
            logger.info("Using CPU (training will be slow)")

        # Load tokenizer and model
        logger.info(f"Loading model checkpoint: {self.config.model_ckpt}")
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)

        # Enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()

        # Data collator handles dynamic padding
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            padding=True,
            label_pad_token_id=tokenizer.pad_token_id,
        )

        # Load tokenized data
        logger.info(f"Loading tokenized dataset from: {self.config.data_path}")
        dataset = load_from_disk(str(self.config.data_path))

        # Determine fp16 availability
        use_fp16 = self.config.fp16 and device == "cuda"

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.config.root_dir),
            num_train_epochs=self.config.num_train_epochs,
            warmup_steps=self.config.warmup_steps,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            eval_strategy=self.config.eval_strategy,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            fp16=use_fp16,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=self.config.report_to,
            logging_dir=os.path.join(str(self.config.root_dir), "logs"),
            dataloader_num_workers=0 if device == "cpu" else 2,
        )

        # Initialize trainer with early stopping
        trainer = Trainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            data_collator=data_collator,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        logger.info("Starting training...")
        logger.info(f"  Epochs: {self.config.num_train_epochs}")
        logger.info(f"  Batch size: {self.config.per_device_train_batch_size}")
        logger.info(f"  Grad accum steps: {self.config.gradient_accumulation_steps}")
        logger.info(f"  Effective batch size: {self.config.per_device_train_batch_size * self.config.gradient_accumulation_steps}")
        logger.info(f"  FP16: {use_fp16}")
        logger.info(f"  Train examples: {len(dataset['train'])}")
        logger.info(f"  Eval examples: {len(dataset['validation'])}")

        trainer.train()

        # Save best model
        model_save_path = os.path.join(str(self.config.root_dir), "bart-samsum-model")
        tokenizer_save_path = os.path.join(str(self.config.root_dir), "tokenizer")

        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(tokenizer_save_path)

        logger.info(f"Model saved to: {model_save_path}")
        logger.info(f"Tokenizer saved to: {tokenizer_save_path}")

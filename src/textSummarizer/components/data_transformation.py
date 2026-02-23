"""Stage 3: Data Transformation Component.

Tokenizes the SAMSum dataset using the BART tokenizer:
  - Encodes dialogues as input sequences (max 1024 tokens)
  - Encodes summaries as target sequences (max 128 tokens)
  - Saves tokenized dataset in Arrow format for efficient training

The tokenized dataset maintains the same splits (train/val/test).
"""

import os
from datasets import load_from_disk
from transformers import AutoTokenizer
from textSummarizer.logging import logger
from textSummarizer.entity import DataTransformationConfig


class DataTransformation:
    """Tokenizes raw text data into model-ready format."""

    def __init__(self, config: DataTransformationConfig) -> None:
        self.config = config
        logger.info(f"Loading tokenizer: {config.tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    def convert_examples_to_features(self, example_batch: dict) -> dict:
        """Tokenize a batch of dialogue-summary pairs.

        Args:
            example_batch: Dict with keys matching text_column and summary_column.

        Returns:
            Dict with input_ids, attention_mask, and labels.
        """
        input_encodings = self.tokenizer(
            example_batch[self.config.text_column],
            max_length=self.config.max_input_length,
            truncation=True,
            padding="max_length",
        )

        target_encodings = self.tokenizer(
            text_target=example_batch[self.config.summary_column],
            max_length=self.config.max_target_length,
            truncation=True,
            padding="max_length",
        )

        return {
            "input_ids": input_encodings["input_ids"],
            "attention_mask": input_encodings["attention_mask"],
            "labels": target_encodings["input_ids"],
        }

    def convert(self) -> None:
        """Load raw dataset, tokenize all splits, and save to disk."""
        logger.info(f"Loading dataset from: {self.config.data_path}")
        dataset = load_from_disk(str(self.config.data_path))

        logger.info("Tokenizing dataset (this may take a few minutes)...")
        tokenized_dataset = dataset.map(
            self.convert_examples_to_features,
            batched=True,
            remove_columns=dataset["train"].column_names,
            desc="Tokenizing",
        )

        output_path = os.path.join(self.config.root_dir, "samsum_dataset")
        tokenized_dataset.save_to_disk(output_path)

        for split_name in tokenized_dataset:
            logger.info(
                f"  Tokenized {split_name}: {len(tokenized_dataset[split_name])} examples"
            )
        logger.info(f"Tokenized dataset saved to: {output_path}")



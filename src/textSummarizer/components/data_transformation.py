"""Stage 3: Data Transformation Component.

Tokenizes the SAMSum dataset using BART tokenizer with robust preprocessing:
    - Normalizes dialogue and summary text
    - Applies augmentation to training split only (anti-overfitting)
    - Encodes dialogues as input sequences (max 1024 tokens)
    - Encodes summaries as target sequences (max 128 tokens)
    - Saves tokenized dataset in Arrow format for efficient training

The tokenized dataset maintains the same splits (train/val/test).
"""

import re
import os
from datasets import load_from_disk
from transformers import AutoTokenizer
from textSummarizer.logging import logger
from textSummarizer.entity import DataTransformationConfig
from textSummarizer.components.data_augmentation import get_augmentation_strategy


class DataTransformation:
    """Tokenizes raw text data into model-ready format with augmentation."""

    def __init__(self, config: DataTransformationConfig) -> None:
        self.config = config
        logger.info(f"Loading tokenizer: {config.tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

        self.augmenter = get_augmentation_strategy(
            augment_prob=self.config.augmentation_probability,
            enable_augmentation=self.config.enable_augmentation,
            seed=42,
        )

    @staticmethod
    def _normalize_dialogue_text(text: str) -> str:
        """Normalize dialogue formatting without changing semantics."""
        text = text if isinstance(text, str) else str(text or "")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{2,}", "\n", text)
        # Normalize speaker prefixes at turn starts: "Name : hi" -> "Name: hi"
        text = re.sub(
            r"(^|\n)\s*([A-Za-z][A-Za-z0-9_ ]{0,24})\s*:\s*",
            r"\1\2: ",
            text,
        )
        return text.strip()

    @staticmethod
    def _normalize_summary_text(text: str) -> str:
        """Normalize generated/target summary text."""
        text = text if isinstance(text, str) else str(text or "")
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"([.!?])\1+", r"\1", text)
        return text

    def _preprocess_batch(self, example_batch: dict, apply_augmentation: bool = False) -> dict:
        """Normalize text and optionally augment dialogue for training only."""
        processed = {k: list(v) if isinstance(v, list) else v for k, v in example_batch.items()}

        normalized_dialogues = []
        for dialogue in processed[self.config.text_column]:
            text = dialogue
            if self.config.enable_text_normalization:
                text = self._normalize_dialogue_text(text)
            if apply_augmentation and self.config.enable_augmentation:
                text = self.augmenter.augment_text(text)
            normalized_dialogues.append(text)
        processed[self.config.text_column] = normalized_dialogues

        if self.config.summary_column in processed:
            if self.config.enable_text_normalization:
                processed[self.config.summary_column] = [
                    self._normalize_summary_text(summary)
                    for summary in processed[self.config.summary_column]
                ]

        return processed

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
        """Load raw dataset, apply augmentation, tokenize all splits, and save to disk."""
        logger.info(f"Loading dataset from: {self.config.data_path}")
        dataset = load_from_disk(str(self.config.data_path))

        logger.info("Applying preprocessing to dataset splits...")
        for split_name in dataset.keys():
            apply_aug = split_name == "train"
            dataset[split_name] = dataset[split_name].map(
                self._preprocess_batch,
                batched=True,
                fn_kwargs={"apply_augmentation": apply_aug},
                desc=f"Preprocessing {split_name}",
            )

        if self.config.enable_augmentation:
            logger.info(
                "Training split augmentation enabled with probability "
                f"{self.config.augmentation_probability}"
            )
        else:
            logger.info("Training split augmentation is disabled")

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



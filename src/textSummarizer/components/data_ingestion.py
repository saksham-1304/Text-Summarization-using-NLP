"""Stage 1: Data Ingestion Component.

Downloads the SAMSum dataset from HuggingFace Hub and saves it locally.
SAMSum is a dialogue summarization dataset with ~16k examples:
  - Train: 14,732 conversations
  - Validation: 818 conversations
  - Test: 819 conversations

Reference: https://huggingface.co/datasets/samsum
Paper: https://arxiv.org/abs/1911.12237
"""

import os
from pathlib import Path
from datasets import load_dataset
from textSummarizer.logging import logger
from textSummarizer.entity import DataIngestionConfig


class DataIngestion:
    """Handles downloading and persisting the dataset."""

    def __init__(self, config: DataIngestionConfig) -> None:
        self.config = config

    def download_dataset(self) -> None:
        """Download SAMSum dataset from HuggingFace Hub.

        Uses the `datasets` library to download directly from the Hub.
        Saves to disk in Arrow format for fast subsequent loading.
        If the dataset already exists locally, skips download.
        """
        local_data_dir = Path(self.config.local_data_dir)

        if local_data_dir.exists() and any(local_data_dir.iterdir()):
            logger.info(
                f"Dataset already exists at {local_data_dir}. "
                f"Skipping download."
            )
            return

        logger.info(f"Downloading dataset: {self.config.dataset_name}")
        dataset = load_dataset(self.config.dataset_name, trust_remote_code=True)

        os.makedirs(local_data_dir, exist_ok=True)
        dataset.save_to_disk(str(local_data_dir))

        # Log dataset statistics
        for split_name, split_data in dataset.items():
            logger.info(
                f"  {split_name}: {len(split_data)} examples, "
                f"columns={split_data.column_names}"
            )

        logger.info(
            f"Dataset '{self.config.dataset_name}' saved to {local_data_dir}"
        )
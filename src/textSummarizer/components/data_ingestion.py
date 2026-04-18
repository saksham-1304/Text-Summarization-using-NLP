"""Stage 1: Data Ingestion Component.

Downloads the SAMSum dataset from HuggingFace Hub and saves it locally.

SAMSum Dataset:
  - 16,397 total examples across 3 splits
  - Train: 14,732 conversations
  - Validation: 818 conversations  
  - Test: 819 conversations
  - Task: Dialogue summarization
  - Avg dialogue length: ~140 words
  - Avg summary length: ~25 words

Reference: https://huggingface.co/datasets/knkarthick/samsum
Paper: https://arxiv.org/abs/1911.12237
"""

import os
from pathlib import Path
from typing import Any

try:
    from datasets import load_dataset, load_from_disk  # type: ignore[import-not-found]
except ImportError:
    load_dataset = None  # type: ignore[assignment]
    load_from_disk = None  # type: ignore[assignment]

from textSummarizer.logging import logger
from textSummarizer.entity import DataIngestionConfig


class DataIngestion:
    """Handles downloading and persisting the SAMSum dataset.
    
    This component manages the first stage of the ML pipeline:
    1. Download dataset from HuggingFace Hub (or cache if exists)
    2. Validate structure and splits
    3. Save to disk for subsequent pipeline stages
    
    Attributes:
        config (DataIngestionConfig): Configuration with paths and dataset name.
    """

    def __init__(self, config: DataIngestionConfig) -> None:
        """Initialize data ingestion component.
        
        Args:
            config: Frozen dataclass with dataset name and save paths.
        
        Raises:
            ValueError: If config paths are invalid.
        """
        if not config:
            raise ValueError("config cannot be None")
        self.config = config

    def download_dataset(self) -> Any:
        """Download SAMSum dataset from HuggingFace Hub and save locally.

        Uses the `datasets` library to download directly from the Hub in
        streaming mode (memory efficient). Saves to disk in Arrow format
        for fast subsequent loading.
        
        Skips download if dataset already exists locally (idempotent).
        
        Returns:
            Any: Loaded dataset object with train/val/test splits.
        
        Raises:
            FileNotFoundError: If local_data_dir cannot be created.
            ConnectionError: If dataset download fails.
            ValueError: If dataset structure is invalid.
            
        Side Effects:
            Creates local_data_dir if it doesn't exist.
            Writes dataset files to disk (~1GB).
        """
        local_data_dir = Path(self.config.local_data_dir)

        if load_dataset is None or load_from_disk is None:
            raise ImportError(
                "Missing optional dependency 'datasets'. Install with: "
                "pip install datasets>=2.16.0"
            )

        if local_data_dir.exists() and any(local_data_dir.iterdir()):
            logger.info(
                f"✓ Dataset already exists at {local_data_dir}. "
                f"Skipping download."
            )
            # Load and return cached dataset
            try:
                dataset = load_from_disk(str(local_data_dir))
                return dataset
            except Exception as e:
                logger.error(f"Failed to load cached dataset: {e}")
                raise

        logger.info(f"Downloading dataset: {self.config.dataset_name}")
        try:
            dataset = load_dataset(self.config.dataset_name)
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            raise ConnectionError(f"Cannot download {self.config.dataset_name}: {e}") from e

        # Validate dataset structure without requiring hard dependency types.
        if not hasattr(dataset, "keys"):
            raise ValueError(f"Expected dataset with split keys, got {type(dataset)}")
        
        required_splits = {"train", "validation", "test"}
        available_splits = set(dataset.keys())
        if not required_splits.issubset(available_splits):
            raise ValueError(
                f"Missing splits. Required: {required_splits}, "
                f"Available: {available_splits}"
            )

        # Create output directory
        try:
            os.makedirs(local_data_dir, exist_ok=True)
        except Exception as e:
            raise FileNotFoundError(
                f"Cannot create directory {local_data_dir}: {e}"
            ) from e
        
        # Save dataset
        try:
            dataset.save_to_disk(str(local_data_dir))
        except Exception as e:
            logger.error(f"Failed to save dataset: {e}")
            raise

        # Log dataset statistics
        for split_name, split_data in dataset.items():
            logger.info(
                f"  ✓ {split_name}: {len(split_data)} examples, "
                f"columns={split_data.column_names}"
            )

        logger.info(
            f"✓ Dataset '{self.config.dataset_name}' saved to {local_data_dir}"
        )
        
        return dataset
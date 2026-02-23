"""Entity definitions for Text Summarization pipeline.

Each dataclass represents the configuration contract for a pipeline stage.
Using frozen=True ensures immutability after creation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict


@dataclass(frozen=True)
class DataIngestionConfig:
    """Configuration for Stage 1: Data Ingestion."""
    root_dir: Path
    dataset_name: str
    local_data_dir: Path


@dataclass(frozen=True)
class DataValidationConfig:
    """Configuration for Stage 2: Data Validation."""
    root_dir: Path
    status_file: Path
    local_data_dir: Path
    required_splits: List[str]
    required_columns: Dict[str, List[str]]


@dataclass(frozen=True)
class DataTransformationConfig:
    """Configuration for Stage 3: Data Transformation."""
    root_dir: Path
    data_path: Path
    tokenizer_name: str
    max_input_length: int
    max_target_length: int
    text_column: str
    summary_column: str


@dataclass(frozen=True)
class ModelTrainerConfig:
    """Configuration for Stage 4: Model Training."""
    root_dir: Path
    data_path: Path
    model_ckpt: str
    num_train_epochs: int
    warmup_steps: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    weight_decay: float
    logging_steps: int
    eval_strategy: str
    eval_steps: int
    save_steps: int
    gradient_accumulation_steps: int
    learning_rate: float
    fp16: bool
    save_total_limit: int
    load_best_model_at_end: bool
    report_to: str


@dataclass(frozen=True)
class ModelEvaluationConfig:
    """Configuration for Stage 5: Model Evaluation."""
    root_dir: Path
    data_path: Path
    model_path: Path
    tokenizer_path: Path
    metric_file_name: Path
    batch_size: int
    max_input_length: int
    max_target_length: int
    text_column: str
    summary_column: str
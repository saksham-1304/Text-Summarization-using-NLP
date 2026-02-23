"""Tests for entity dataclass definitions."""

import pytest
from pathlib import Path
from textSummarizer.entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
)


class TestDataIngestionConfig:
    """Tests for DataIngestionConfig entity."""

    def test_creation(self):
        config = DataIngestionConfig(
            root_dir=Path("artifacts/data_ingestion"),
            dataset_name="samsum",
            local_data_dir=Path("artifacts/data_ingestion/samsum_dataset"),
        )
        assert config.dataset_name == "samsum"
        assert config.root_dir == Path("artifacts/data_ingestion")

    def test_frozen(self):
        config = DataIngestionConfig(
            root_dir=Path("test"),
            dataset_name="samsum",
            local_data_dir=Path("test/data"),
        )
        with pytest.raises(AttributeError):
            config.dataset_name = "other"


class TestDataValidationConfig:
    """Tests for DataValidationConfig entity."""

    def test_creation(self):
        config = DataValidationConfig(
            root_dir=Path("artifacts/data_validation"),
            status_file=Path("artifacts/data_validation/status.txt"),
            local_data_dir=Path("artifacts/data_ingestion/samsum_dataset"),
            required_splits=["train", "test", "validation"],
            required_columns={"train": ["id", "dialogue", "summary"]},
        )
        assert len(config.required_splits) == 3
        assert "dialogue" in config.required_columns["train"]


class TestDataTransformationConfig:
    """Tests for DataTransformationConfig entity."""

    def test_creation(self):
        config = DataTransformationConfig(
            root_dir=Path("test"),
            data_path=Path("test/data"),
            tokenizer_name="facebook/bart-large-cnn",
            max_input_length=1024,
            max_target_length=128,
            text_column="dialogue",
            summary_column="summary",
        )
        assert config.max_input_length == 1024
        assert config.tokenizer_name == "facebook/bart-large-cnn"


class TestModelTrainerConfig:
    """Tests for ModelTrainerConfig entity."""

    def test_creation(self):
        config = ModelTrainerConfig(
            root_dir=Path("test"),
            data_path=Path("test/data"),
            model_ckpt="facebook/bart-large-cnn",
            num_train_epochs=3,
            warmup_steps=500,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            weight_decay=0.01,
            logging_steps=50,
            eval_strategy="steps",
            eval_steps=500,
            save_steps=500,
            gradient_accumulation_steps=8,
            learning_rate=2e-5,
            fp16=True,
            save_total_limit=3,
            load_best_model_at_end=True,
            report_to="none",
        )
        assert config.num_train_epochs == 3
        assert config.fp16 is True
        assert config.learning_rate == 2e-5


class TestModelEvaluationConfig:
    """Tests for ModelEvaluationConfig entity."""

    def test_creation(self):
        config = ModelEvaluationConfig(
            root_dir=Path("test"),
            data_path=Path("test/data"),
            model_path=Path("test/model"),
            tokenizer_path=Path("test/tokenizer"),
            metric_file_name=Path("test/metrics.csv"),
            batch_size=8,
            max_input_length=1024,
            max_target_length=128,
            text_column="dialogue",
            summary_column="summary",
        )
        assert config.batch_size == 8
        assert config.text_column == "dialogue"

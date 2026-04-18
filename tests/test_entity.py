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
            dataset_name="knkarthick/samsum",
            local_data_dir=Path("artifacts/data_ingestion/samsum_dataset"),
        )
        assert config.dataset_name == "knkarthick/samsum"
        assert config.root_dir == Path("artifacts/data_ingestion")

    def test_frozen(self):
        config = DataIngestionConfig(
            root_dir=Path("test"),
            dataset_name="knkarthick/samsum",
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
            enable_augmentation=True,
            augmentation_probability=0.25,
            enable_text_normalization=True,
        )
        assert config.max_input_length == 1024
        assert config.tokenizer_name == "facebook/bart-large-cnn"
        assert config.enable_augmentation is True
        assert config.augmentation_probability == 0.25


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
            seed=42,
            max_grad_norm=1.0,
            label_smoothing_factor=0.1,
            lr_scheduler_type="linear",
            early_stopping_patience=3,
            early_stopping_threshold=0.0,
            group_by_length=True,
            dataloader_num_workers=2,
            max_input_length=1024,
            max_target_length=128,
            text_column="dialogue",
            summary_column="summary",
            smoke_max_steps=5,
            smoke_train_samples=64,
            smoke_eval_samples=32,
        )
        assert config.num_train_epochs == 3
        assert config.fp16 is True
        assert config.learning_rate == 2e-5
        assert config.label_smoothing_factor == 0.1
        assert config.smoke_max_steps == 5


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
            default_num_beams=1,
            default_length_penalty=0.8,
            default_no_repeat_ngram_size=5,
            enable_decoding_sweep=True,
            decoding_sweep_max_samples=200,
            decoding_selection_metric="rougeLsum",
            decoding_num_beams=[2, 4, 6],
            decoding_length_penalties=[0.8, 1.0, 1.2],
            decoding_no_repeat_ngram_sizes=[3, 4, 5],
        )
        assert config.batch_size == 8
        assert config.text_column == "dialogue"
        assert config.enable_decoding_sweep is True

"""Tests for ConfigurationManager."""

import pytest
import tempfile
from pathlib import Path

from textSummarizer.utils.common import save_json
from textSummarizer.entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
)


class TestConfigurationManager:
    """Tests for the ConfigurationManager class.

    Note: These tests use the actual config.yaml and params.yaml files.
    They validate that the configuration manager correctly reads and
    constructs typed config objects.
    """

    @pytest.fixture
    def config_manager(self):
        """Create a ConfigurationManager using actual config files."""
        from textSummarizer.config.configuration import ConfigurationManager
        return ConfigurationManager()

    def test_data_ingestion_config(self, config_manager):
        config = config_manager.get_data_ingestion_config()
        assert isinstance(config, DataIngestionConfig)
        assert config.dataset_name == "knkarthick/samsum"
        assert "data_ingestion" in str(config.root_dir)

    def test_data_validation_config(self, config_manager):
        config = config_manager.get_data_validation_config()
        assert isinstance(config, DataValidationConfig)
        assert len(config.required_splits) == 3
        assert "train" in config.required_splits
        assert "test" in config.required_splits
        assert "validation" in config.required_splits

    def test_data_transformation_config(self, config_manager):
        config = config_manager.get_data_transformation_config()
        assert isinstance(config, DataTransformationConfig)
        assert config.tokenizer_name == "facebook/bart-large-cnn"
        assert config.max_input_length == 1024
        assert config.max_target_length == 128
        assert config.enable_augmentation is True
        assert config.augmentation_probability == 0.25
        assert config.enable_text_normalization is True

    def test_model_trainer_config(self, config_manager):
        config = config_manager.get_model_trainer_config()
        assert isinstance(config, ModelTrainerConfig)
        assert config.model_ckpt == "facebook/bart-large-cnn"
        assert config.num_train_epochs == 3
        assert config.learning_rate == 2e-5
        assert config.fp16 is True
        assert config.seed == 42
        assert config.max_grad_norm == 1.0
        assert config.label_smoothing_factor == 0.1
        assert config.lr_scheduler_type == "linear"
        assert config.early_stopping_patience == 3
        assert config.group_by_length is True
        assert config.smoke_max_steps == 5
        assert config.smoke_train_samples == 16
        assert config.smoke_eval_samples == 8

    def test_model_evaluation_config(self, config_manager):
        config = config_manager.get_model_evaluation_config()
        assert isinstance(config, ModelEvaluationConfig)
        assert config.batch_size == 8
        assert config.text_column == "dialogue"
        assert config.summary_column == "summary"
        assert config.default_num_beams == 1
        assert config.default_length_penalty == 0.8
        assert config.default_no_repeat_ngram_size == 5
        assert config.enable_decoding_sweep is True

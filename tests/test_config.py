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
        assert config.dataset_name == "samsum"
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

    def test_model_trainer_config(self, config_manager):
        config = config_manager.get_model_trainer_config()
        assert isinstance(config, ModelTrainerConfig)
        assert config.model_ckpt == "facebook/bart-large-cnn"
        assert config.num_train_epochs == 3
        assert config.learning_rate == 2e-5
        assert config.fp16 is True

    def test_model_evaluation_config(self, config_manager):
        config = config_manager.get_model_evaluation_config()
        assert isinstance(config, ModelEvaluationConfig)
        assert config.batch_size == 8
        assert config.text_column == "dialogue"
        assert config.summary_column == "summary"

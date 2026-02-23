"""Configuration Manager for the Text Summarization pipeline.

Reads config.yaml and params.yaml, then constructs strongly-typed
configuration objects for each pipeline stage.
"""

from textSummarizer.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from textSummarizer.utils.common import read_yaml, create_directories
from textSummarizer.entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
)


class ConfigurationManager:
    """Central configuration manager that reads YAML configs and
    provides typed config objects for each pipeline stage."""

    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH,
    ) -> None:
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """Build config for Stage 1: Data Ingestion."""
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        return DataIngestionConfig(
            root_dir=config.root_dir,
            dataset_name=config.dataset_name,
            local_data_dir=config.local_data_dir,
        )

    def get_data_validation_config(self) -> DataValidationConfig:
        """Build config for Stage 2: Data Validation."""
        config = self.config.data_validation
        create_directories([config.root_dir])

        return DataValidationConfig(
            root_dir=config.root_dir,
            status_file=config.status_file,
            local_data_dir=config.local_data_dir,
            required_splits=config.required_splits,
            required_columns=config.required_columns,
        )

    def get_data_transformation_config(self) -> DataTransformationConfig:
        """Build config for Stage 3: Data Transformation."""
        config = self.config.data_transformation
        params = self.params.DataTransformation
        create_directories([config.root_dir])

        return DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            tokenizer_name=config.tokenizer_name,
            max_input_length=params.max_input_length,
            max_target_length=params.max_target_length,
            text_column=params.text_column,
            summary_column=params.summary_column,
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        """Build config for Stage 4: Model Training."""
        config = self.config.model_trainer
        params = self.params.TrainingArguments
        create_directories([config.root_dir])

        return ModelTrainerConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            model_ckpt=config.model_ckpt,
            num_train_epochs=params.num_train_epochs,
            warmup_steps=params.warmup_steps,
            per_device_train_batch_size=params.per_device_train_batch_size,
            per_device_eval_batch_size=params.per_device_eval_batch_size,
            weight_decay=params.weight_decay,
            logging_steps=params.logging_steps,
            eval_strategy=params.eval_strategy,
            eval_steps=params.eval_steps,
            save_steps=params.save_steps,
            gradient_accumulation_steps=params.gradient_accumulation_steps,
            learning_rate=params.learning_rate,
            fp16=params.fp16,
            save_total_limit=params.save_total_limit,
            load_best_model_at_end=params.load_best_model_at_end,
            report_to=params.report_to,
        )

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        """Build config for Stage 5: Model Evaluation."""
        config = self.config.model_evaluation
        params = self.params.ModelEvaluation
        create_directories([config.root_dir])

        return ModelEvaluationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            model_path=config.model_path,
            tokenizer_path=config.tokenizer_path,
            metric_file_name=config.metric_file_name,
            batch_size=params.batch_size,
            max_input_length=params.max_input_length,
            max_target_length=params.max_target_length,
            text_column=params.text_column,
            summary_column=params.summary_column,
        )

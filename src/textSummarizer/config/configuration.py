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

        enable_augmentation = bool(getattr(params, "enable_augmentation", True))
        augmentation_probability = float(getattr(params, "augmentation_probability", 0.25))
        enable_text_normalization = bool(getattr(params, "enable_text_normalization", True))

        return DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            tokenizer_name=config.tokenizer_name,
            max_input_length=params.max_input_length,
            max_target_length=params.max_target_length,
            text_column=params.text_column,
            summary_column=params.summary_column,
            enable_augmentation=enable_augmentation,
            augmentation_probability=augmentation_probability,
            enable_text_normalization=enable_text_normalization,
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        """Build config for Stage 4: Model Training."""
        config = self.config.model_trainer
        params = self.params.TrainingArguments
        transform_params = self.params.DataTransformation
        create_directories([config.root_dir])

        seed = int(getattr(params, "seed", 42))
        max_grad_norm = float(getattr(params, "max_grad_norm", 1.0))
        label_smoothing_factor = float(getattr(params, "label_smoothing_factor", 0.0))
        lr_scheduler_type = str(getattr(params, "lr_scheduler_type", "linear"))
        early_stopping_patience = int(getattr(params, "early_stopping_patience", 3))
        early_stopping_threshold = float(getattr(params, "early_stopping_threshold", 0.0))
        group_by_length = bool(getattr(params, "group_by_length", False))
        dataloader_num_workers = int(getattr(params, "dataloader_num_workers", 0))
        smoke_max_steps = int(getattr(params, "smoke_max_steps", 5))
        smoke_train_samples = int(getattr(params, "smoke_train_samples", 64))
        smoke_eval_samples = int(getattr(params, "smoke_eval_samples", 32))

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
            seed=seed,
            max_grad_norm=max_grad_norm,
            label_smoothing_factor=label_smoothing_factor,
            lr_scheduler_type=lr_scheduler_type,
            early_stopping_patience=early_stopping_patience,
            early_stopping_threshold=early_stopping_threshold,
            group_by_length=group_by_length,
            dataloader_num_workers=dataloader_num_workers,
            max_input_length=transform_params.max_input_length,
            max_target_length=transform_params.max_target_length,
            text_column=transform_params.text_column,
            summary_column=transform_params.summary_column,
            smoke_max_steps=smoke_max_steps,
            smoke_train_samples=smoke_train_samples,
            smoke_eval_samples=smoke_eval_samples,
        )

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        """Build config for Stage 5: Model Evaluation."""
        config = self.config.model_evaluation
        params = self.params.ModelEvaluation
        create_directories([config.root_dir])

        default_num_beams = int(getattr(params, "default_num_beams", 1))
        default_length_penalty = float(getattr(params, "default_length_penalty", 0.8))
        default_no_repeat_ngram_size = int(getattr(params, "default_no_repeat_ngram_size", 5))
        enable_decoding_sweep = bool(getattr(params, "enable_decoding_sweep", False))
        decoding_sweep_max_samples = int(getattr(params, "decoding_sweep_max_samples", 200))
        decoding_selection_metric = str(getattr(params, "decoding_selection_metric", "rougeLsum"))
        decoding_num_beams = list(getattr(params, "decoding_num_beams", [2, 4, 6]))
        decoding_length_penalties = list(
            getattr(params, "decoding_length_penalties", [0.8, 1.0, 1.2])
        )
        decoding_no_repeat_ngram_sizes = list(
            getattr(params, "decoding_no_repeat_ngram_sizes", [3, 4, 5])
        )

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
            default_num_beams=default_num_beams,
            default_length_penalty=default_length_penalty,
            default_no_repeat_ngram_size=default_no_repeat_ngram_size,
            enable_decoding_sweep=enable_decoding_sweep,
            decoding_sweep_max_samples=decoding_sweep_max_samples,
            decoding_selection_metric=decoding_selection_metric,
            decoding_num_beams=[int(x) for x in decoding_num_beams],
            decoding_length_penalties=[float(x) for x in decoding_length_penalties],
            decoding_no_repeat_ngram_sizes=[int(x) for x in decoding_no_repeat_ngram_sizes],
        )

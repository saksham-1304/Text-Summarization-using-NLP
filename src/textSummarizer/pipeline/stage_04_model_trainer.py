"""Stage 4 Pipeline: Model Training.

Fine-tunes BART-large-CNN on SAMSum dataset.
"""

from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.components.model_trainer import ModelTrainer
from textSummarizer.logging import logger


STAGE_NAME = "Model Training"


class ModelTrainerTrainingPipeline:
    """Orchestrates the model training stage."""

    def __init__(self) -> None:
        pass

    def main(self) -> None:
        """Execute model training pipeline."""
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config)
        model_trainer.train()
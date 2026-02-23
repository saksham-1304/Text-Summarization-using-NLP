"""Stage 5 Pipeline: Model Evaluation.

Evaluates the trained model using ROUGE metrics.
"""

from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.components.model_evaluation import ModelEvaluation
from textSummarizer.logging import logger


STAGE_NAME = "Model Evaluation"


class ModelEvaluationTrainingPipeline:
    """Orchestrates the model evaluation stage."""

    def __init__(self) -> None:
        pass

    def main(self) -> dict:
        """Execute model evaluation pipeline.

        Returns:
            Dict of metric name -> score.
        """
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        metrics = model_evaluation.evaluate()
        return metrics
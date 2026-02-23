"""Stage 3 Pipeline: Data Transformation.

Tokenizes raw dataset into model-ready format.
"""

from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.components.data_transformation import DataTransformation
from textSummarizer.logging import logger


STAGE_NAME = "Data Transformation"


class DataTransformationTrainingPipeline:
    """Orchestrates the data transformation stage."""

    def __init__(self) -> None:
        pass

    def main(self) -> None:
        """Execute data transformation pipeline."""
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.convert()
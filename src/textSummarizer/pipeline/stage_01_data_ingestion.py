"""Stage 1 Pipeline: Data Ingestion.

Downloads SAMSum dataset from HuggingFace Hub.
"""

from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.components.data_ingestion import DataIngestion
from textSummarizer.logging import logger


STAGE_NAME = "Data Ingestion"


class DataIngestionTrainingPipeline:
    """Orchestrates the data ingestion stage."""

    def __init__(self) -> None:
        pass

    def main(self) -> None:
        """Execute data ingestion pipeline."""
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_dataset()

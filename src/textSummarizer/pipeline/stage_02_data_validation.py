"""Stage 2 Pipeline: Data Validation.

Validates dataset integrity, schema, and quality.
"""

from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.components.data_validation import DataValidation
from textSummarizer.logging import logger


STAGE_NAME = "Data Validation"


class DataValidationTrainingPipeline:
    """Orchestrates the data validation stage."""

    def __init__(self) -> None:
        pass

    def main(self) -> bool:
        """Execute data validation pipeline.

        Returns:
            True if validation passes, False otherwise.

        Raises:
            RuntimeError: If validation fails.
        """
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        is_valid = data_validation.validate_all_files_exist()

        if not is_valid:
            raise RuntimeError(
                "Data validation failed. Check the status file for details: "
                f"{data_validation_config.status_file}"
            )

        return is_valid
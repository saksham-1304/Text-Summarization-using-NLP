"""Stage 2 Pipeline: Data Validation.

Validates dataset integrity, schema, and quality.
"""

from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.components.data_validation import DataValidation
from textSummarizer.logging import logger


STAGE_NAME = "Data Validation"


class DataValidationTrainingPipeline:
    """Orchestrates the data validation stage."""

    def main(self) -> bool:
        """Execute data validation pipeline.

        Returns:
            True if validation passes, False otherwise.

        Raises:
            RuntimeError: If validation fails (BEFORE any side effects).
        """
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        
        # Check validity FIRST (before any side effects like file writes)
        is_valid = data_validation.validate_all_files_exist()

        # ONLY after validation succeeds, continue
        if not is_valid:
            raise RuntimeError(
                "❌ Data validation failed. Check status file for details: "
                f"{data_validation_config.status_file}"
            )

        logger.info("✓ Data validation passed")
        return is_valid
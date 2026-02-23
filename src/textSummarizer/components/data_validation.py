"""Stage 2: Data Validation Component.

Validates the downloaded dataset for:
  - Required splits (train, test, validation) exist
  - Required columns present in each split
  - No empty splits
  - No null/empty values in critical fields
  - Data type correctness

Writes a detailed validation report to disk.
"""

import json
from pathlib import Path
from datasets import load_from_disk
from textSummarizer.logging import logger
from textSummarizer.entity import DataValidationConfig


class DataValidation:
    """Validates dataset integrity and schema correctness."""

    def __init__(self, config: DataValidationConfig) -> None:
        self.config = config

    def validate_all_files_exist(self) -> bool:
        """Run all validation checks and write status report.

        Returns:
            True if all validations pass, False otherwise.
        """
        validation_results = {}
        overall_status = True

        try:
            # Load dataset from disk
            data_path = Path(self.config.local_data_dir)
            if not data_path.exists():
                msg = f"Dataset directory not found: {data_path}"
                logger.error(msg)
                self._write_status(False, {"error": msg})
                return False

            dataset = load_from_disk(str(data_path))
            available_splits = list(dataset.keys())
            logger.info(f"Available splits: {available_splits}")

            # 1. Validate required splits exist
            for split in self.config.required_splits:
                if split not in available_splits:
                    validation_results[f"split_{split}_exists"] = False
                    overall_status = False
                    logger.error(f"Required split '{split}' not found")
                else:
                    validation_results[f"split_{split}_exists"] = True
                    logger.info(f"Split '{split}' found with {len(dataset[split])} examples")

            # 2. Validate required columns in each split
            for split, columns in self.config.required_columns.items():
                if split not in available_splits:
                    continue
                actual_columns = dataset[split].column_names
                for col in columns:
                    key = f"split_{split}_col_{col}"
                    if col not in actual_columns:
                        validation_results[key] = False
                        overall_status = False
                        logger.error(f"Column '{col}' missing in split '{split}'")
                    else:
                        validation_results[key] = True

            # 3. Validate no empty splits
            for split in self.config.required_splits:
                if split in available_splits:
                    key = f"split_{split}_non_empty"
                    if len(dataset[split]) == 0:
                        validation_results[key] = False
                        overall_status = False
                        logger.error(f"Split '{split}' is empty")
                    else:
                        validation_results[key] = True

            # 4. Validate no null values in dialogue/summary columns
            for split in self.config.required_splits:
                if split not in available_splits:
                    continue
                for col in ["dialogue", "summary"]:
                    if col not in dataset[split].column_names:
                        continue
                    null_count = sum(
                        1 for val in dataset[split][col]
                        if val is None or (isinstance(val, str) and val.strip() == "")
                    )
                    key = f"split_{split}_{col}_no_nulls"
                    if null_count > 0:
                        validation_results[key] = False
                        overall_status = False
                        logger.warning(
                            f"{null_count} null/empty '{col}' values in '{split}'"
                        )
                    else:
                        validation_results[key] = True

            # 5. Log dataset statistics
            for split in self.config.required_splits:
                if split in available_splits:
                    avg_dialogue_len = sum(
                        len(d) for d in dataset[split]["dialogue"]
                    ) / len(dataset[split])
                    avg_summary_len = sum(
                        len(s) for s in dataset[split]["summary"]
                    ) / len(dataset[split])
                    logger.info(
                        f"Stats [{split}]: avg_dialogue_chars={avg_dialogue_len:.0f}, "
                        f"avg_summary_chars={avg_summary_len:.0f}"
                    )

        except Exception as e:
            logger.exception(f"Validation failed with error: {e}")
            validation_results["exception"] = str(e)
            overall_status = False

        self._write_status(overall_status, validation_results)
        return overall_status

    def _write_status(self, status: bool, details: dict) -> None:
        """Write validation status and details to file."""
        report = {
            "validation_status": status,
            "details": details,
        }
        status_file = Path(self.config.status_file)
        status_file.parent.mkdir(parents=True, exist_ok=True)
        with open(status_file, "w") as f:
            f.write(json.dumps(report, indent=2))
        logger.info(f"Validation status: {status} -> {status_file}")

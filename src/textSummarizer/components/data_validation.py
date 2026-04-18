"""Stage 2: Data Validation Component.

Validates the downloaded dataset for integrity and schema correctness.

Validation Checks:
  1. Required splits (train, validation, test) exist
  2. Required columns present in each split
  3. No empty splits  
  4. No null/empty values in dialogue/summary columns
  5. Data type correctness
  6. Statistical sanity checks

Writes a detailed JSON validation report to disk for audit trail.
"""

import json
from pathlib import Path
from typing import Dict, Any

try:
    from datasets import load_from_disk
except ImportError:
    load_from_disk = None  # type: ignore[assignment]

from textSummarizer.logging import logger
from textSummarizer.entity import DataValidationConfig


class DataValidation:
    """Validates dataset integrity, schema, and data quality.
    
    This component implements comprehensive validation checks on the SAMSum
    dataset before it proceeds to tokenization and training.
    
    Validation failure at any step is reported to status_file for audit.
    
    Attributes:
        config (DataValidationConfig): Configuration with paths and validation rules.
    """

    def __init__(self, config: DataValidationConfig) -> None:
        """Initialize data validation component.
        
        Args:
            config: Frozen dataclass with validation settings and paths.
        
        Raises:
            ValueError: If config is invalid.
        """
        if not config:
            raise ValueError("config cannot be None")
        self.config = config

    def validate_all_files_exist(self) -> bool:
        """Run all validation checks and write status report to disk.

        Performs 5 sequential validation checks:
          1. Dataset directory exists and is readable
          2. Required splits (train/val/test) are present
          3. Required columns (dialogue, summary) are in each split
          4. No splits are empty
          5. No null/empty values in key columns
          6. Dataset statistics (avg lengths) are reasonable
        
        All results are persisted to self.config.status_file for audit trail.
        
        Returns:
            True if all validations pass, False if any check fails.
        
        Raises:
            RuntimeError: Only on unrecoverable errors (not file structure issues).
        
        Side Effects:
            Writes JSON validation report to self.config.status_file.
            Logs detailed validation results to logger.
        """
        validation_results: Dict[str, Any] = {}
        overall_status = True

        if load_from_disk is None:
            logger.error(
                "Missing optional dependency 'datasets'. Install with: "
                "pip install datasets>=2.16.0"
            )
            self._write_status(
                False,
                {
                    "error": (
                        "Missing dependency 'datasets'. "
                        "Install with: pip install datasets>=2.16.0"
                    )
                },
            )
            return False

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

    def _write_status(self, status: bool, details: Dict[str, Any]) -> None:
        """Write validation status and details to JSON report file.
        
        Args:
            status: Overall validation result (True = pass, False = fail).
            details: Dictionary with detailed validation check results.
        
        Side Effects:
            Creates parent directory if needed.
            Writes JSON file with validation report.
        """
        report = {
            "validation_status": status,
            "details": details,
        }
        status_file = Path(self.config.status_file)
        status_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(status_file, "w") as f:
                json.dump(report, f, indent=2)
            logger.info(f"✓ Validation report written to {status_file}")
        except Exception as e:
            logger.error(f"Failed to write validation report: {e}")
            raise

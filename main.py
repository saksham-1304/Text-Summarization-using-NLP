"""Main training orchestrator for the Text Summarization pipeline.

Executes all 5 stages sequentially:
  1. Data Ingestion    - Download SAMSum from HuggingFace Hub
  2. Data Validation   - Validate schema, splits, and data quality
  3. Data Transformation - Tokenize with BART tokenizer
  4. Model Training    - Fine-tune BART-large-CNN on SAMSum
  5. Model Evaluation  - Evaluate with ROUGE metrics

Usage:
    python main.py                    # Run all stages
    python main.py --stage 1          # Run only stage 1
    python main.py --stage 1 --to 3   # Run stages 1 through 3
"""

import argparse
import sys
import time

from textSummarizer.pipeline.stage_01_data_ingestion import (
    DataIngestionTrainingPipeline,
    STAGE_NAME as STAGE_1_NAME,
)
from textSummarizer.pipeline.stage_02_data_validation import (
    DataValidationTrainingPipeline,
    STAGE_NAME as STAGE_2_NAME,
)
from textSummarizer.pipeline.stage_03_data_transformation import (
    DataTransformationTrainingPipeline,
    STAGE_NAME as STAGE_3_NAME,
)
from textSummarizer.pipeline.stage_04_model_trainer import (
    ModelTrainerTrainingPipeline,
    STAGE_NAME as STAGE_4_NAME,
)
from textSummarizer.pipeline.stage_05_model_evaluation import (
    ModelEvaluationTrainingPipeline,
    STAGE_NAME as STAGE_5_NAME,
)
from textSummarizer.logging import logger


STAGES = {
    1: (STAGE_1_NAME, DataIngestionTrainingPipeline),
    2: (STAGE_2_NAME, DataValidationTrainingPipeline),
    3: (STAGE_3_NAME, DataTransformationTrainingPipeline),
    4: (STAGE_4_NAME, ModelTrainerTrainingPipeline),
    5: (STAGE_5_NAME, ModelEvaluationTrainingPipeline),
}


def run_stage(stage_num: int) -> None:
    """Execute a single pipeline stage.

    Args:
        stage_num: Stage number (1-5).
    """
    stage_name, pipeline_class = STAGES[stage_num]
    logger.info(f"{'='*60}")
    logger.info(f">>> Stage {stage_num}: {stage_name} - STARTED")
    logger.info(f"{'='*60}")

    start_time = time.time()

    pipeline = pipeline_class()
    pipeline.main()

    elapsed = time.time() - start_time
    logger.info(f">>> Stage {stage_num}: {stage_name} - COMPLETED ({elapsed:.1f}s)")
    logger.info("")


def main(start_stage: int = 1, end_stage: int = 5) -> None:
    """Run the training pipeline from start_stage to end_stage.

    Args:
        start_stage: First stage to execute (1-5).
        end_stage: Last stage to execute (1-5).
    """
    logger.info(f"Text Summarization Training Pipeline")
    logger.info(f"Running stages {start_stage} through {end_stage}")
    logger.info(f"Model: facebook/bart-large-cnn | Dataset: SAMSum")
    logger.info("")

    total_start = time.time()

    for stage_num in range(start_stage, end_stage + 1):
        try:
            run_stage(stage_num)
        except Exception as e:
            logger.exception(
                f"Stage {stage_num} failed with error: {e}"
            )
            sys.exit(1)

    total_elapsed = time.time() - total_start
    logger.info(f"{'='*60}")
    logger.info(f"ALL STAGES COMPLETED SUCCESSFULLY ({total_elapsed:.1f}s)")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Text Summarization Training Pipeline"
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
        help="Starting stage number (default: 1)",
    )
    parser.add_argument(
        "--to",
        type=int,
        default=5,
        choices=[1, 2, 3, 4, 5],
        help="Ending stage number (default: 5)",
    )
    args = parser.parse_args()

    if args.stage > args.to:
        print(f"Error: --stage ({args.stage}) must be <= --to ({args.to})")
        sys.exit(1)

    main(start_stage=args.stage, end_stage=args.to)






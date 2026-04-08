"""Entry point for the end-to-end pipeline.

Runs data preparation, preprocessing, training, and optionally fine-tuning
in sequence.

Usage::

    python scripts/run_full_pipeline.py
"""

import sys

from app.config import settings
from app.core.dataset_validator import DatasetValidator
from app.facades.data_pipeline_facade import DataPipelineFacade
from app.facades.training_pipeline_facade import TrainingPipelineFacade
from app.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Execute the full end-to-end pipeline."""
    # Stage 1: Data preparation
    logger.info("=== DATA PREPARATION ===")
    data_facade = DataPipelineFacade()
    data_facade.run()

    # Stage 1.5: Validate dataset
    logger.info("=== DATASET VALIDATION ===")
    validator = DatasetValidator(settings.YOLO_ROOT)
    if not validator.validate_all():
        logger.error("Validation failed — aborting pipeline")
        sys.exit(1)

    # Stage 2: Training pipeline (includes preprocessing + training)
    logger.info("=== TRAINING PIPELINE ===")
    training_facade = TrainingPipelineFacade(
        dataset_path=str(settings.YOLO_ROOT),
        preprocessed_path=str(settings.PREPROCESSED_ROOT),
    )
    results = training_facade.run()

    logger.info("=== PIPELINE COMPLETE ===")
    for key, value in results.items():
        logger.info("%s: %s", key, value)


if __name__ == "__main__":
    main()

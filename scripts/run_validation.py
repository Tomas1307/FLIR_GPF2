"""Entry point for dataset validation.

Runs all validation checks on the YOLO dataset and exits with code 0 if
all checks pass, 1 otherwise.

Usage::

    python scripts/run_validation.py
"""

import sys

from app.config import settings
from app.core.dataset_validator import DatasetValidator
from app.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Validate the YOLO dataset."""
    logger.info("Starting dataset validation")
    validator = DatasetValidator(settings.YOLO_ROOT)
    if validator.validate_all():
        logger.info("Validation passed — dataset is clean")
        sys.exit(0)
    else:
        logger.error("Validation failed — fix issues before training")
        sys.exit(1)


if __name__ == "__main__":
    main()

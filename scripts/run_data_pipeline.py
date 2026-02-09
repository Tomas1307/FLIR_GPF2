"""Entry point for the data preparation pipeline.

Stages executed:
  1. Unify raw data.
  2. Strategic split.
  3. Class augmentation.
  4. Label cleaning.
  5. Duplicate removal.
  6. Warehouse balancing.

Usage::

    python scripts/run_data_pipeline.py
"""

from app.facades.data_pipeline_facade import DataPipelineFacade
from app.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Run the complete data preparation pipeline."""
    logger.info("Starting data preparation pipeline")
    facade = DataPipelineFacade()
    stats = facade.run()

    for split, dist in stats.items():
        logger.info("%s: %s", split.upper(), dist)

    logger.info("Data pipeline finished")


if __name__ == "__main__":
    main()

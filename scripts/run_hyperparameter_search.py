"""Entry point for hyperparameter search.

Usage::

    python scripts/run_hyperparameter_search.py
"""

from app.config import settings
from app.training.hyperparameter_searcher import HyperparameterSearcher
from app.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Run hyperparameter search optimising for target class recall."""
    datasets = [str(settings.PREPROCESSED_ROOT)]

    logger.info("Starting hyperparameter search")
    searcher = HyperparameterSearcher(
        dataset_paths=datasets,
        target_class=settings.TARGET_CLASS,
        base_epochs=settings.EPOCHS,
        gpu_memory_gb=settings.GPU_MEMORY_GB,
    )
    results = searcher.run_search()
    logger.info("Search complete: %d experiments run", len(results))


if __name__ == "__main__":
    main()

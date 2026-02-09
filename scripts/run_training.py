"""Entry point for conservative model training.

Usage::

    python scripts/run_training.py
"""

from app.config import settings
from app.schemas.training_schema import TrainingConfig
from app.training.conservative_trainer import ConservativeTrainer
from app.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Train the conservative mining detector."""
    dataset = str(settings.PREPROCESSED_ROOT)
    logger.info("Starting conservative training on %s", dataset)

    trainer = ConservativeTrainer(TrainingConfig())
    adapter, metrics = trainer.train(dataset)

    if metrics:
        recall = metrics["recall_optimised"]["recall"]
        logger.info("Training complete. Recall: %.4f (%.1f%%)", recall, recall * 100)
    else:
        logger.warning("Training did not produce metrics")


if __name__ == "__main__":
    main()

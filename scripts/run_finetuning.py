"""Entry point for model fine-tuning.

Usage::

    python scripts/run_finetuning.py --model <path_to_best.pt>
"""

import argparse
from pathlib import Path

from app.config import settings
from app.schemas.finetuning_schema import FinetuningConfig
from app.training.finetuner import Finetuner
from app.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Fine-tune a pretrained model."""
    parser = argparse.ArgumentParser(description="Fine-tune a YOLO model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the pretrained model weights (.pt)",
    )
    parser.add_argument(
        "--dataset-yaml",
        type=str,
        default=str(settings.PREPROCESSED_ROOT / "dataset.yaml"),
        help="Path to the dataset YAML",
    )
    args = parser.parse_args()

    config = FinetuningConfig(
        model_path=args.model,
        dataset_yaml=args.dataset_yaml,
    )

    logger.info("Starting fine-tuning: %s", args.model)
    finetuner = Finetuner(config)
    adapter, metrics = finetuner.finetune()

    if metrics.class_metrics:
        logger.info(
            "Fine-tuning complete. Recall: %.4f", metrics.class_metrics.recall
        )
    else:
        logger.info("Fine-tuning complete")


if __name__ == "__main__":
    main()

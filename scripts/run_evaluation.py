"""Entry point for model evaluation.

Usage::

    python scripts/run_evaluation.py --model <path_to_best.pt>
"""

import argparse

from app.config import settings
from app.training.metrics_extractor import MetricsExtractor
from app.training.model_factory import ModelFactory
from app.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Evaluate a trained model on the test split."""
    parser = argparse.ArgumentParser(description="Evaluate a YOLO model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the model weights (.pt)",
    )
    parser.add_argument(
        "--dataset-yaml",
        type=str,
        default=str(settings.PREPROCESSED_ROOT / "dataset.yaml"),
        help="Path to the dataset YAML",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate on",
    )
    args = parser.parse_args()

    factory = ModelFactory()
    adapter = factory.create_from_checkpoint(args.model)

    extractor = MetricsExtractor()
    result = extractor.evaluate_by_class(
        adapter, args.dataset_yaml, split=args.split
    )

    logger.info("Overall -- P: %.4f  R: %.4f  mAP50: %.4f  mAP50-95: %.4f",
                result.precision_all, result.recall_all,
                result.map50_all, result.map50_95_all)

    if result.class_metrics:
        cm = result.class_metrics
        logger.info(
            "Target class %d -- P: %.4f  R: %.4f  AP50: %.4f",
            settings.TARGET_CLASS, cm.precision, cm.recall, cm.ap50,
        )


if __name__ == "__main__":
    main()

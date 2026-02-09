from typing import Any

from app.adapters.yolo_adapter import YoloAdapter
from app.config import settings
from app.schemas.metrics_schema import MetricsResult, MiningMetrics
from app.utils.logger import get_logger

logger = get_logger(__name__)


class MetricsExtractor:
    """Extracts per-class and overall detection metrics from YOLO validation
    results.

    Attributes:
        target_class: The class id for which detailed metrics are reported.
    """

    def __init__(self, target_class: int = settings.TARGET_CLASS) -> None:
        """Initialise the MetricsExtractor.

        Args:
            target_class: The class id of primary interest.
        """
        self.target_class = target_class

    def extract_class_metrics(
        self, val_results: Any, eval_type: str
    ) -> MiningMetrics:
        """Extract target-class metrics from validation results.

        Args:
            val_results: The object returned by ``YOLO.val()``.
            eval_type: A label describing the evaluation mode.

        Returns:
            A ``MiningMetrics`` schema populated from the results.
        """
        if hasattr(val_results, "box") and val_results.box is not None:
            per_class_recall = val_results.box.r
            per_class_precision = val_results.box.p
            per_class_ap50 = val_results.box.ap50

            if len(per_class_recall) > self.target_class:
                return MiningMetrics(
                    eval_type=eval_type,
                    recall=float(per_class_recall[self.target_class]),
                    precision=float(per_class_precision[self.target_class]),
                    ap50=float(per_class_ap50[self.target_class]),
                    overall_map50=float(val_results.box.map50),
                    overall_map=float(val_results.box.map),
                )

        return MiningMetrics(eval_type=eval_type)

    def evaluate_by_class(
        self,
        model: YoloAdapter,
        dataset_yaml: str,
        split: str = "test",
        image_size: int = settings.IMAGE_SIZE,
    ) -> MetricsResult:
        """Run full validation and return overall and per-class metrics.

        Args:
            model: A ``YoloAdapter`` wrapping the model to evaluate.
            dataset_yaml: Path to the dataset YAML file.
            split: Dataset split to evaluate on.
            image_size: Image resolution for evaluation.

        Returns:
            A ``MetricsResult`` with overall and target-class metrics.
        """
        val_results = model.validate(
            data=dataset_yaml,
            split=split,
            imgsz=image_size,
            iou=0.5,
            verbose=True,
        )

        overall = MetricsResult(
            precision_all=float(val_results.box.p.mean()),
            recall_all=float(val_results.box.r.mean()),
            map50_all=float(val_results.box.map50),
            map50_95_all=float(val_results.box.map),
        )

        class_metrics = self.extract_class_metrics(val_results, "full_evaluation")
        overall.class_metrics = class_metrics

        logger.info(
            "Target class %d -- Recall: %.4f, Precision: %.4f, AP50: %.4f",
            self.target_class,
            class_metrics.recall,
            class_metrics.precision,
            class_metrics.ap50,
        )

        return overall

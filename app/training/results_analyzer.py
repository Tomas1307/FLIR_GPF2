from typing import Any, Dict, List

import pandas as pd

from app.schemas.metrics_schema import MiningMetrics
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ResultsAnalyzer:
    """Analyses training and evaluation results and generates actionable
    recommendations.
    """

    def __init__(self) -> None:
        """Initialise the ResultsAnalyzer."""

    @staticmethod
    def generate_recommendations(metrics: MiningMetrics) -> List[str]:
        """Generate textual recommendations based on mining-class metrics.

        Args:
            metrics: Metrics for the target class.

        Returns:
            A list of recommendation strings.
        """
        recommendations: List[str] = []
        recall = metrics.recall
        precision = metrics.precision

        if recall >= 0.95:
            recommendations.append("Excellent recall achieved.")
        elif recall >= 0.90:
            recommendations.append("Very good recall. Minor tuning may help.")
        elif recall >= 0.85:
            recommendations.append(
                "Good recall. Consider more augmentation or a lower confidence threshold."
            )
        else:
            recommendations.append(
                "Recall is below target. Review augmentation strategy and class balance."
            )

        if precision < 0.5:
            recommendations.append(
                "Precision is low. Consider raising the confidence threshold for deployment."
            )
        elif precision < 0.7:
            recommendations.append("Precision is moderate. Acceptable for recall-first applications.")
        else:
            recommendations.append("Excellent precision.")

        return recommendations

    @staticmethod
    def summarise_search_results(
        results: List[Dict[str, Any]], target_class: int
    ) -> None:
        """Log a summary of hyperparameter search results sorted by recall.

        Args:
            results: List of result dictionaries from the search.
            target_class: The class id of interest.
        """
        df = pd.DataFrame(results)
        valid = df[df["target_class_recall"] > 0].copy()

        if valid.empty:
            logger.warning("No valid results found")
            return

        valid = valid.sort_values("target_class_recall", ascending=False)

        logger.info(
            "Top 5 configurations by recall for class %d:", target_class
        )
        for i, (_, row) in enumerate(valid.head(5).iterrows()):
            logger.info(
                "#%d - %s | Recall: %.4f | Precision: %.4f | mAP50: %.4f",
                i + 1,
                row.get("config_name", f"Run {row['run_id']}"),
                row["target_class_recall"],
                row["target_class_precision"],
                row["overall_map50"],
            )

        recalls = valid["target_class_recall"]
        logger.info(
            "Recall stats -- Best: %.4f, Mean: %.4f, Min: %.4f, >0.9: %d",
            recalls.max(),
            recalls.mean(),
            recalls.min(),
            int((recalls > 0.9).sum()),
        )

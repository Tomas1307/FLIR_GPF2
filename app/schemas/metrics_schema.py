from typing import List, Optional

from pydantic import BaseModel, Field


class MiningMetrics(BaseModel):
    """Schema for class-specific detection metrics.

    Attributes:
        eval_type: Label describing the evaluation mode (e.g. recall-optimised).
        recall: Recall for the target class.
        precision: Precision for the target class.
        ap50: Average Precision at IoU 0.50 for the target class.
        overall_map50: Mean Average Precision at IoU 0.50 across all classes.
        overall_map: Mean Average Precision at IoU 0.50:0.95 across all classes.
    """

    eval_type: str = Field(..., description="Evaluation mode label.")
    recall: float = Field(default=0.0)
    precision: float = Field(default=0.0)
    ap50: float = Field(default=0.0)
    overall_map50: float = Field(default=0.0)
    overall_map: float = Field(default=0.0)


class MetricsResult(BaseModel):
    """Schema aggregating overall and per-class evaluation metrics.

    Attributes:
        precision_all: Mean precision across all classes.
        recall_all: Mean recall across all classes.
        map50_all: Mean AP at IoU 0.50 across all classes.
        map50_95_all: Mean AP at IoU 0.50:0.95 across all classes.
        class_metrics: Optional per-class metrics for the target class.
        recommendations: Optional list of textual recommendations.
    """

    precision_all: float = Field(default=0.0)
    recall_all: float = Field(default=0.0)
    map50_all: float = Field(default=0.0)
    map50_95_all: float = Field(default=0.0)
    class_metrics: Optional[MiningMetrics] = Field(default=None)
    recommendations: List[str] = Field(default_factory=list)

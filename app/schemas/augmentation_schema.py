from pydantic import BaseModel, Field


class AugmentationTarget(BaseModel):
    """Schema defining the augmentation objective for a single class.

    Attributes:
        class_id: The class id to augment.
        current_count: Number of samples currently in the training set.
        target_count: Desired number of samples after augmentation.
        has_bboxes: Whether augmentation should preserve bounding boxes.
    """

    class_id: int = Field(..., description="Class id to augment.")
    current_count: int = Field(
        ..., ge=0, description="Current sample count in training set."
    )
    target_count: int = Field(
        ..., ge=0, description="Target sample count after augmentation."
    )
    has_bboxes: bool = Field(
        default=True,
        description="Whether bounding boxes should be preserved.",
    )

    @property
    def samples_needed(self) -> int:
        """Return the number of new samples required."""
        return max(0, self.target_count - self.current_count)

    @property
    def multiplication_factor(self) -> float:
        """Return the multiplication factor from current to target."""
        if self.current_count == 0:
            return 0.0
        return self.target_count / self.current_count

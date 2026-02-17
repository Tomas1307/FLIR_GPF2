import albumentations as A


class MiningAugmentationPipeline:
    """Augmentation pipeline specifically tuned for the illegal-mining class.

    Uses moderate, realistic transformations to multiply the number of
    mining samples while preserving bounding-box accuracy.
    """

    def __init__(self) -> None:
        """Initialise the MiningAugmentationPipeline."""
        self._pipeline = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.6),
                A.MotionBlur(blur_limit=3, p=0.3),
                A.RandomGamma(p=0.3),
                A.Affine(
                    scale=(0.8, 1.2),
                    translate_percent=0.1,
                    rotate=(-20, 20),
                    p=0.5,
                ),
                A.GaussianBlur(blur_limit=3, p=0.2),
            ],
            bbox_params=A.BboxParams(
                format="yolo", label_fields=["class_labels"]
            ),
        )

    @property
    def pipeline(self) -> A.Compose:
        """Return the underlying Albumentations pipeline.

        Returns:
            The composed augmentation pipeline.
        """
        return self._pipeline

from typing import Optional, Union

import albumentations as A

from app.augmentation.mining_augmentation import MiningAugmentationPipeline
from app.augmentation.normal_augmentation import NormalAugmentationPipeline
from app.augmentation.background_augmentation import BackgroundAugmentation
from app.config import settings


class AugmentationFactory:
    """Factory that creates the appropriate augmentation pipeline based on
    the class id.

    * **class_id == -1** (background) -> ``BackgroundAugmentation``
    * **class_id == TARGET_CLASS** (mining) -> ``MiningAugmentationPipeline``
    * **all other classes** -> ``NormalAugmentationPipeline``
    """

    def __init__(self) -> None:
        """Initialise the AugmentationFactory."""

    @staticmethod
    def create(
        class_id: int,
    ) -> Union[BackgroundAugmentation, MiningAugmentationPipeline, NormalAugmentationPipeline]:
        """Return the augmentation component for a given class id.

        Args:
            class_id: The class id that determines which pipeline to return.

        Returns:
            The appropriate augmentation object.
        """
        if class_id == -1:
            return BackgroundAugmentation()
        if class_id == settings.TARGET_CLASS:
            return MiningAugmentationPipeline()
        return NormalAugmentationPipeline()

    @staticmethod
    def get_pipeline(class_id: int) -> Optional[A.Compose]:
        """Convenience method that returns the underlying Albumentations
        ``Compose`` pipeline (or ``None`` for background).

        Args:
            class_id: The class id.

        Returns:
            An ``A.Compose`` instance, or ``None`` for background images.
        """
        if class_id == -1:
            return None
        component = AugmentationFactory.create(class_id)
        return component.pipeline

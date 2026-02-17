from typing import Tuple

import numpy as np

from app.preprocessing.noise_filter import NoiseFilter
from app.preprocessing.contrast_enhancer import ContrastEnhancer
from app.schemas.preprocessing_schema import PreprocessingConfig


class PreprocessingPipeline:
    """Facade that chains noise filtering and contrast enhancement into a
    single preprocessing step.

    This class composes ``NoiseFilter`` and ``ContrastEnhancer`` so that
    callers need only invoke a single ``process`` method.

    Attributes:
        noise_filter: Noise reduction component.
        contrast_enhancer: Contrast enhancement component.
        config: Preprocessing configuration.
    """

    def __init__(self, config: PreprocessingConfig = PreprocessingConfig()) -> None:
        """Initialise the PreprocessingPipeline.

        Args:
            config: Preprocessing parameters (thresholds, kernel sizes, etc.).
        """
        self.config = config
        self.noise_filter = NoiseFilter(
            threshold=config.noise_threshold,
            kernel_size=config.median_kernel,
        )
        self.contrast_enhancer = ContrastEnhancer(
            clip_limit=config.clahe_clip,
            tile_grid_size=config.clahe_tile_grid,
        )

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Apply the full preprocessing chain to *image*.

        Steps executed (when enabled in *config*):
          1. Salt-and-pepper noise reduction via median filter.
          2. CLAHE contrast enhancement.

        Args:
            image: Input image in BGR colour space.

        Returns:
            A tuple of ``(processed_image, noise_detected)``.
        """
        noise_detected = False

        if self.config.apply_noise_filter:
            image, noise_detected = self.noise_filter.apply(image)

        if self.config.apply_contrast_enhancement:
            image = self.contrast_enhancer.apply(image)

        return image, noise_detected

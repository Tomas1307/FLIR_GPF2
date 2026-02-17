from typing import Tuple

import cv2
import numpy as np

from app.config import settings


class NoiseFilter:
    """Applies a median-blur filter to reduce salt-and-pepper noise when the
    proportion of extreme pixel values exceeds a configurable threshold.

    Attributes:
        threshold: Minimum ratio of extreme pixels to trigger filtering.
        kernel_size: Kernel size for the median blur (must be odd).
    """

    def __init__(
        self,
        threshold: float = settings.NOISE_THRESHOLD,
        kernel_size: int = settings.MEDIAN_KERNEL,
    ) -> None:
        """Initialise the NoiseFilter.

        Args:
            threshold: Ratio of very dark + very bright pixels that
                triggers the filter.
            kernel_size: Median blur kernel size.
        """
        self.threshold = threshold
        self.kernel_size = kernel_size

    def apply(self, image: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Apply salt-and-pepper noise reduction if needed.

        The image is converted to greyscale internally to check the ratio of
        extreme pixels (< 30 or > 225).  If the ratio exceeds the threshold
        the median filter is applied; otherwise the original image is returned
        unmodified.

        Args:
            image: Input image in BGR colour space.

        Returns:
            A tuple of ``(filtered_image, noise_detected)`` where
            *noise_detected* is ``True`` when the filter was applied.
        """
        grey = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if len(image.shape) > 2
            else image.copy()
        )

        total_pixels = grey.shape[0] * grey.shape[1]
        very_dark = np.sum(grey < 30) / total_pixels
        very_bright = np.sum(grey > 225) / total_pixels

        if very_dark + very_bright > self.threshold:
            result = cv2.medianBlur(image, self.kernel_size)
            return result, True

        return image, False

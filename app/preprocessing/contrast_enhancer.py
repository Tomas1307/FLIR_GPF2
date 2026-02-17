from typing import Tuple

import cv2
import numpy as np

from app.config import settings


class ContrastEnhancer:
    """Enhances image contrast using CLAHE (Contrast Limited Adaptive
    Histogram Equalisation) applied to the L channel of the LAB colour
    space.

    Attributes:
        clip_limit: CLAHE clip limit controlling contrast amplification.
        tile_grid_size: Grid size for the CLAHE tile partitioning.
    """

    def __init__(
        self,
        clip_limit: float = settings.CLAHE_CLIP,
        tile_grid_size: Tuple[int, int] = settings.CLAHE_TILE_GRID,
    ) -> None:
        """Initialise the ContrastEnhancer.

        Args:
            clip_limit: CLAHE clip limit.
            tile_grid_size: CLAHE tile grid size.
        """
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE contrast enhancement to *image*.

        The image is converted from BGR to LAB, CLAHE is applied to the
        lightness channel, and the result is converted back to BGR.

        Args:
            image: Input image in BGR colour space.

        Returns:
            The contrast-enhanced image in BGR.
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size
        )
        l_enhanced = clahe.apply(l_channel)

        lab_enhanced = cv2.merge((l_enhanced, a_channel, b_channel))
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

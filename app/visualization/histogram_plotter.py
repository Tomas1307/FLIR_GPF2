from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np


class HistogramPlotter:
    """Plots greyscale intensity histograms for a list of images.

    Useful for comparing the pixel distribution before and after
    preprocessing steps.
    """

    def __init__(self) -> None:
        """Initialise the HistogramPlotter."""

    @staticmethod
    def plot(
        images: List[np.ndarray],
        titles: List[str],
    ) -> None:
        """Plot histograms for each image side by side.

        Args:
            images: List of images in RGB or greyscale format.
            titles: Corresponding display titles.
        """
        n = len(images)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
        if n == 1:
            axes = [axes]

        for ax, img, title in zip(axes, images, titles):
            grey = (
                cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                if len(img.shape) == 3
                else img
            )
            hist = cv2.calcHist([grey], [0], None, [256], [0, 256])
            ax.plot(hist)
            ax.set_title(f"Histogram: {title}")
            ax.set_xlim([0, 256])
            ax.grid(True)

        plt.tight_layout()
        plt.show()

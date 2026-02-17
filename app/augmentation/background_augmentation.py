import random

import cv2
import numpy as np


class BackgroundAugmentation:
    """OpenCV-based augmentation for background images (no bounding boxes).

    Background images require purely pixel-level transforms since there are
    no annotations to preserve.  This class applies a randomised sequence of
    flips, brightness changes, gamma corrections, blurs, rotations, and
    Gaussian noise.
    """

    def __init__(self) -> None:
        """Initialise the BackgroundAugmentation."""

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply random augmentations to a background image.

        Args:
            image: Input image in BGR colour space.

        Returns:
            The augmented image.
        """
        result = image.copy()
        h, w = result.shape[:2]

        if random.random() < 0.5:
            result = cv2.flip(result, 1)

        if random.random() < 0.5:
            alpha = random.uniform(0.5, 1.5)
            beta = random.randint(-50, 50)
            result = cv2.convertScaleAbs(result, alpha=alpha, beta=beta)

        if random.random() < 0.3:
            gamma = random.uniform(0.5, 2.0)
            lut = np.empty((1, 256), np.uint8)
            for i in range(256):
                lut[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
            result = cv2.LUT(result, lut)

        if random.random() < 0.3:
            size = random.choice([3, 5, 7])
            kernel = np.zeros((size, size))
            kernel[int((size - 1) / 2), :] = np.ones(size)
            kernel = kernel / size
            result = cv2.filter2D(result, -1, kernel)

        if random.random() < 0.2:
            ksize = random.choice([3, 5])
            result = cv2.GaussianBlur(result, (ksize, ksize), 0)

        if random.random() < 0.4:
            angle = random.uniform(-15, 15)
            centre = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(centre, angle, 1.0)
            result = cv2.warpAffine(result, rotation_matrix, (w, h))

        if random.random() < 0.4:
            noise = np.random.normal(
                0, random.randint(10, 30), result.shape
            ).astype(np.uint8)
            result = cv2.add(result, noise)

        return result

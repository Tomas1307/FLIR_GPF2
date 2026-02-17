from pathlib import Path

import cv2
import numpy as np


def load_image(path: Path) -> np.ndarray:
    """Load an image from disk in BGR colour space.

    Args:
        path: File system path to the image.

    Returns:
        The loaded image as a NumPy array (BGR).

    Raises:
        FileNotFoundError: If the image cannot be read.
    """
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def save_image(path: Path, image: np.ndarray) -> None:
    """Write an image to disk.

    Args:
        path: Destination file path.
        image: Image array (BGR) to save.
    """
    cv2.imwrite(str(path), image)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to RGB colour space.

    Args:
        image: Image in BGR format.

    Returns:
        Image converted to RGB.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

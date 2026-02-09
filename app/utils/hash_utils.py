import hashlib
from pathlib import Path
from typing import Optional

import cv2


def compute_image_hash(path: Path, size: int = 64) -> Optional[str]:
    """Compute a perceptual hash for an image file.

    The image is loaded in greyscale, resized to a fixed grid, and hashed
    using SHA-256.  This provides a fast duplicate-detection mechanism.

    Args:
        path: Path to the image file.
        size: Side length of the square to which the image is resized before
            hashing.  Smaller values increase collision tolerance.

    Returns:
        Hex-encoded SHA-256 digest, or ``None`` if the image cannot be read.
    """
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    resized = cv2.resize(image, (size, size))
    return hashlib.sha256(resized.tobytes()).hexdigest()

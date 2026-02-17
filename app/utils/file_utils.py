import shutil
from pathlib import Path
from typing import List

from app.utils.logger import get_logger

logger = get_logger(__name__)


def copy_file(src: Path, dst: Path) -> None:
    """Copy a single file preserving metadata.

    Args:
        src: Source file path.
        dst: Destination file path.
    """
    shutil.copy2(src, dst)


def ensure_dir(path: Path) -> Path:
    """Create a directory (and parents) if it does not already exist.

    Args:
        path: Directory path to create.

    Returns:
        The same ``path`` for convenient chaining.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def glob_images(directory: Path, extensions: tuple = (".jpg", ".jpeg", ".png")) -> List[Path]:
    """Return all image files in *directory* matching common extensions.

    Args:
        directory: Directory to search.
        extensions: Tuple of lowercase file extensions to include.

    Returns:
        Sorted list of matching ``Path`` objects.
    """
    images: List[Path] = []
    for ext in extensions:
        images.extend(directory.glob(f"*{ext}"))
    return sorted(images)

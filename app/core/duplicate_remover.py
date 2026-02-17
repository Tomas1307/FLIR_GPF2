from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from app.config import settings
from app.utils.file_utils import glob_images
from app.utils.hash_utils import compute_image_hash
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DuplicateRemover:
    """Detects and removes duplicate images based on perceptual hashing.

    Attributes:
        yolo_root: Root directory of the YOLO dataset.
    """

    def __init__(self, yolo_root: Path = settings.YOLO_ROOT) -> None:
        """Initialise the DuplicateRemover.

        Args:
            yolo_root: Root directory of the YOLO dataset.
        """
        self.yolo_root = Path(yolo_root)

    def _find_duplicates(
        self, image_dir: Path
    ) -> List[Tuple[str, str]]:
        """Find duplicate images within a directory using perceptual hashing.

        Args:
            image_dir: Directory containing images.

        Returns:
            List of ``(duplicate_filename, original_filename)`` pairs.
        """
        hash_map: Dict[str, Path] = {}
        duplicates: List[Tuple[str, str]] = []

        for img_path in glob_images(image_dir):
            h = compute_image_hash(img_path)
            if h is None:
                continue
            if h in hash_map:
                duplicates.append((img_path.name, hash_map[h].name))
            else:
                hash_map[h] = img_path

        return duplicates

    def _remove_duplicates(
        self,
        duplicates: List[Tuple[str, str]],
        image_dir: Path,
        label_dir: Path,
    ) -> int:
        """Delete duplicate image and label files.

        Args:
            duplicates: List of ``(duplicate_filename, original_filename)`` pairs.
            image_dir: Directory containing images.
            label_dir: Directory containing labels.

        Returns:
            Number of duplicate pairs removed.
        """
        removed = 0
        for dup_name, _ in duplicates:
            img_path = image_dir / dup_name
            lbl_path = label_dir / (Path(dup_name).stem + ".txt")
            if img_path.exists():
                img_path.unlink()
                removed += 1
            if lbl_path.exists():
                lbl_path.unlink()
        return removed

    def clean(self) -> Dict[str, int]:
        """Scan all splits for duplicates and remove them.

        Returns:
            Dictionary mapping split name to the number of duplicates removed.
        """
        results: Dict[str, int] = {}

        for split in ("train", "val", "test"):
            image_dir = self.yolo_root / split / "images"
            label_dir = self.yolo_root / split / "labels"

            if not image_dir.exists():
                continue

            logger.info("Checking duplicates in %s", split)
            duplicates = self._find_duplicates(image_dir)

            if duplicates:
                logger.info(
                    "Found %d duplicates in %s -- removing",
                    len(duplicates),
                    split,
                )
                removed = self._remove_duplicates(
                    duplicates, image_dir, label_dir
                )
                results[split] = removed
            else:
                logger.info("No duplicates found in %s", split)
                results[split] = 0

        return results

from pathlib import Path
from typing import Tuple

from tqdm import tqdm

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class LabelCleaner:
    """Cleans YOLO label files by normalising class ids that were
    accidentally written as decimals (e.g. ``4.0`` instead of ``4``).

    Attributes:
        yolo_root: Root directory of the YOLO dataset.
    """

    def __init__(self, yolo_root: Path = settings.YOLO_ROOT) -> None:
        """Initialise the LabelCleaner.

        Args:
            yolo_root: Root directory of the YOLO dataset.
        """
        self.yolo_root = Path(yolo_root)

    def clean(self) -> Tuple[int, int]:
        """Fix decimal class ids across all splits.

        Iterates over every label file in ``train``, ``val``, and ``test``
        splits and converts class ids like ``4.0`` to ``4``.

        Returns:
            A tuple of ``(files_modified, lines_corrected)``.
        """
        files_modified = 0
        lines_corrected = 0

        for split in ("train", "val", "test"):
            lbl_dir = self.yolo_root / split / "labels"
            if not lbl_dir.exists():
                continue

            logger.info("Cleaning labels in %s", split)

            for lbl_path in tqdm(
                list(lbl_dir.glob("*.txt")), desc=f"Cleaning {split}"
            ):
                content = lbl_path.read_text().strip()
                if not content:
                    continue

                new_lines = []
                file_changed = False

                for line in content.splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        original_class = parts[0]
                        try:
                            clean_class = str(int(float(original_class)))
                            if original_class != clean_class:
                                file_changed = True
                                lines_corrected += 1
                            new_lines.append(
                                f"{clean_class} {' '.join(parts[1:])}"
                            )
                        except ValueError:
                            new_lines.append(line)
                    else:
                        new_lines.append(line)

                if file_changed:
                    lbl_path.write_text("\n".join(new_lines))
                    files_modified += 1

        logger.info(
            "Label cleaning complete: %d files modified, %d lines corrected",
            files_modified,
            lines_corrected,
        )
        return files_modified, lines_corrected

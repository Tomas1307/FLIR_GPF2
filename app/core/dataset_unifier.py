import shutil
from pathlib import Path

from app.config import settings
from app.utils.file_utils import ensure_dir, glob_images
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DatasetUnifier:
    """Merges images and labels from multiple split directories into a single
    unified dataset folder.

    The original data layout may scatter images across ``train``, ``val``, and
    ``test`` directories.  This class copies every image and its corresponding
    label into a flat ``images/`` and ``labels/`` structure under the unified
    root so that a fresh strategic split can be performed later.

    Attributes:
        data_root: Source data directory (contains ``Imagenes/`` and ``Etiquetas/``).
        unified_root: Destination directory for the merged output.
    """

    def __init__(
        self,
        data_root: Path = settings.DATA_ROOT,
        unified_root: Path = settings.UNIFIED_ROOT,
    ) -> None:
        """Initialise the DatasetUnifier.

        Args:
            data_root: Root of the raw data directory.
            unified_root: Destination root for unified images and labels.
        """
        self.data_root = Path(data_root)
        self.unified_root = Path(unified_root)

    def unify(self) -> int:
        """Copy all images and labels into the unified directory.

        For each image found, the corresponding label file is copied if it
        exists; otherwise an empty label file is created (background image).

        Returns:
            The total number of images processed.
        """
        img_dirs = [
            self.data_root / "Imagenes" / "train",
            self.data_root / "Imagenes" / "val",
            self.data_root / "Imagenes" / "test",
        ]
        lbl_dirs = [
            self.data_root / "Etiquetas" / "train",
            self.data_root / "Etiquetas" / "val",
            self.data_root / "Etiquetas" / "test",
        ]

        dest_images = ensure_dir(self.unified_root / "images")
        dest_labels = ensure_dir(self.unified_root / "labels")

        files_copied = 0

        for img_dir, lbl_dir in zip(img_dirs, lbl_dirs):
            if not img_dir.exists():
                logger.warning("Image directory not found: %s", img_dir)
                continue

            logger.info("Processing: %s", img_dir)

            for img_path in glob_images(img_dir):
                shutil.copy2(img_path, dest_images / img_path.name)

                lbl_path = lbl_dir / f"{img_path.stem}.txt"
                dest_lbl = dest_labels / f"{img_path.stem}.txt"

                if lbl_path.exists():
                    shutil.copy2(lbl_path, dest_lbl)
                else:
                    dest_lbl.write_text("")

                files_copied += 1

        logger.info("Unification complete: %d files processed", files_copied)
        return files_copied

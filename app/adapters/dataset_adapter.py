from pathlib import Path
from typing import Dict

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DatasetAdapter:
    """Adapter that resolves path inconsistencies in the raw data layout.

    The original dataset uses inconsistent naming (``val`` vs ``validation``,
    ``Imagenes`` vs ``images``, etc.).  This adapter normalises these paths
    into a consistent interface.

    Attributes:
        data_root: Root directory of the raw dataset.
    """

    def __init__(self, data_root: Path = settings.DATA_ROOT) -> None:
        """Initialise the DatasetAdapter.

        Args:
            data_root: Root directory containing the raw data.
        """
        self.data_root = Path(data_root)

    def resolve_image_dirs(self) -> Dict[str, Path]:
        """Resolve the image directories for each split.

        Checks both ``Imagenes`` and ``images`` folder names, and both
        ``val`` and ``validation`` split names.

        Returns:
            Dictionary mapping split name (``train``, ``val``, ``test``)
            to the resolved image directory path.
        """
        dirs: Dict[str, Path] = {}
        image_folder = self._resolve_folder("Imagenes", "images")

        for split, alternatives in [
            ("train", ["train"]),
            ("val", ["val", "validation", "valid"]),
            ("test", ["test"]),
        ]:
            for alt in alternatives:
                candidate = self.data_root / image_folder / alt
                if candidate.exists():
                    dirs[split] = candidate
                    break
            if split not in dirs:
                logger.warning("Image directory for split '%s' not found", split)

        return dirs

    def resolve_label_dirs(self) -> Dict[str, Path]:
        """Resolve the label directories for each split.

        Checks both ``Etiquetas`` and ``labels`` folder names, and handles
        ``val`` / ``validation`` variations.

        Returns:
            Dictionary mapping split name to the resolved label directory path.
        """
        dirs: Dict[str, Path] = {}
        label_folder = self._resolve_folder("Etiquetas", "labels")

        for split, alternatives in [
            ("train", ["train"]),
            ("val", ["val", "validation", "valid"]),
            ("test", ["test"]),
        ]:
            for alt in alternatives:
                candidate = self.data_root / label_folder / alt
                if candidate.exists():
                    dirs[split] = candidate
                    break
            if split not in dirs:
                logger.warning("Label directory for split '%s' not found", split)

        return dirs

    def _resolve_folder(self, *candidates: str) -> str:
        """Return the first folder name that exists under *data_root*.

        Args:
            *candidates: Potential folder names to check.

        Returns:
            The first existing folder name, or the first candidate as default.
        """
        for name in candidates:
            if (self.data_root / name).exists():
                return name
        return candidates[0]

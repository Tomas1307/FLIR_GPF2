import random
from pathlib import Path
from typing import Tuple

from tqdm import tqdm

from app.config import settings
from app.utils.file_utils import glob_images
from app.utils.label_utils import parse_yolo_label
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ClassBalancer:
    """Removes excess augmented images of a dominant class to improve
    dataset balance.

    The original pipeline over-augments the warehouses class (class 1).
    This class identifies augmented warehouse-only images and removes a
    configurable fraction of them.

    Attributes:
        yolo_root: Root directory of the YOLO dataset.
        removal_ratio: Fraction of augmented images to remove (0.0 -- 1.0).
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        yolo_root: Path = settings.YOLO_ROOT,
        removal_ratio: float = settings.BODEGA_REMOVAL_RATIO,
        seed: int = settings.RANDOM_SEED,
    ) -> None:
        """Initialise the ClassBalancer.

        Args:
            yolo_root: Root directory of the YOLO dataset.
            removal_ratio: Fraction of augmented images to remove.
            seed: Random seed.
        """
        self.yolo_root = Path(yolo_root)
        self.removal_ratio = removal_ratio
        self.seed = seed

    def remove_excess(self, class_id: int = 1) -> Tuple[int, int]:
        """Remove a fraction of augmented images belonging exclusively to
        *class_id*.

        Only images whose filename starts with ``aug_c{class_id}_`` and whose
        label contains only the target class are considered for removal.

        Args:
            class_id: The class whose augmented samples should be reduced.

        Returns:
            A tuple of ``(images_removed, objects_removed)``.
        """
        random.seed(self.seed)

        train_img_dir = self.yolo_root / "train" / "images"
        train_lbl_dir = self.yolo_root / "train" / "labels"

        augmented_stems = []

        for img_path in train_img_dir.glob(f"aug_c{class_id}_*.jpg"):
            lbl_path = train_lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue

            annotations = parse_yolo_label(lbl_path)
            has_target = False
            only_target = True

            for cls, *_ in annotations:
                if cls == class_id:
                    has_target = True
                else:
                    only_target = False
                    break

            if has_target and only_target:
                augmented_stems.append(img_path.stem)

        logger.info(
            "Found %d augmented images for class %d",
            len(augmented_stems),
            class_id,
        )

        n_remove = int(len(augmented_stems) * self.removal_ratio)
        random.shuffle(augmented_stems)
        to_remove = augmented_stems[:n_remove]

        images_removed = 0
        objects_removed = 0

        for stem in tqdm(to_remove, desc=f"Removing class {class_id} excess"):
            img_path = train_img_dir / f"{stem}.jpg"
            lbl_path = train_lbl_dir / f"{stem}.txt"

            if lbl_path.exists():
                annotations = parse_yolo_label(lbl_path)
                objects_removed += sum(
                    1 for cls, *_ in annotations if cls == class_id
                )

            if img_path.exists():
                img_path.unlink()
            if lbl_path.exists():
                lbl_path.unlink()
            images_removed += 1

        logger.info(
            "Removed %d images and ~%d objects for class %d",
            images_removed,
            objects_removed,
            class_id,
        )
        return images_removed, objects_removed

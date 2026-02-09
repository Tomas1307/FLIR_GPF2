import random
from pathlib import Path
from typing import Dict, Optional

import cv2
from tqdm import tqdm

from app.augmentation.augmentation_factory import AugmentationFactory
from app.augmentation.background_augmentation import BackgroundAugmentation
from app.config import settings
from app.utils.file_utils import glob_images
from app.utils.image_utils import load_image, save_image
from app.utils.label_utils import parse_yolo_label, write_yolo_label
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ClassAugmentor:
    """Applies class-specific augmentation to reach configurable target
    counts for every class in the training set.

    Attributes:
        yolo_root: Root of the YOLO dataset.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        yolo_root: Path = settings.YOLO_ROOT,
        seed: int = settings.RANDOM_SEED,
    ) -> None:
        """Initialise the ClassAugmentor.

        Args:
            yolo_root: Root of the YOLO dataset.
            seed: Random seed.
        """
        self.yolo_root = Path(yolo_root)
        self.seed = seed

    def _collect_class_images(
        self, class_id: int, train_img_dir: Path, train_lbl_dir: Path
    ) -> list:
        """Collect image stems that contain the given class.

        Args:
            class_id: Target class id (-1 for background).
            train_img_dir: Training images directory.
            train_lbl_dir: Training labels directory.

        Returns:
            List of image stem strings.
        """
        stems = []
        for img_path in glob_images(train_img_dir):
            lbl_path = train_lbl_dir / f"{img_path.stem}.txt"

            if class_id == -1:
                if not lbl_path.exists() or not lbl_path.read_text().strip():
                    stems.append(img_path.stem)
            else:
                annotations = parse_yolo_label(lbl_path)
                for cls, *_ in annotations:
                    if cls == class_id:
                        stems.append(img_path.stem)
                        break

        return stems

    def augment_class(self, class_id: int, target_count: int) -> int:
        """Augment a single class to reach *target_count* samples.

        Args:
            class_id: The class to augment (-1 for background).
            target_count: Desired sample count after augmentation.

        Returns:
            Number of new images generated.
        """
        random.seed(self.seed)

        train_img_dir = self.yolo_root / "train" / "images"
        train_lbl_dir = self.yolo_root / "train" / "labels"

        class_images = self._collect_class_images(
            class_id, train_img_dir, train_lbl_dir
        )

        current = len(class_images)
        needed = target_count - current

        if needed <= 0:
            logger.info(
                "Class %d: already has %d >= %d", class_id, current, target_count
            )
            return 0

        logger.info(
            "Class %d: %d -> %d (+%d)", class_id, current, target_count, needed
        )

        augmentor = AugmentationFactory.create(class_id)
        is_background = class_id == -1
        pipeline = None if is_background else augmentor.pipeline

        generated = 0
        attempts = 0
        max_attempts = needed * 3

        with tqdm(total=needed, desc=f"Class {class_id}") as pbar:
            while generated < needed and attempts < max_attempts:
                stem = random.choice(class_images)
                img_path = train_img_dir / f"{stem}.jpg"
                lbl_path = train_lbl_dir / f"{stem}.txt"

                try:
                    image = cv2.imread(str(img_path))
                    if image is None:
                        attempts += 1
                        continue
                except Exception:
                    attempts += 1
                    continue

                try:
                    aug_stem = f"aug_c{class_id}_{stem}_{generated:04d}"
                    aug_img_path = train_img_dir / f"{aug_stem}.jpg"
                    aug_lbl_path = train_lbl_dir / f"{aug_stem}.txt"

                    if is_background:
                        aug_img = augmentor.apply(image)
                        save_image(aug_img_path, aug_img)
                        aug_lbl_path.write_text("")
                        generated += 1
                        pbar.update(1)
                    else:
                        annotations = parse_yolo_label(lbl_path)
                        bboxes = []
                        class_labels = []

                        for cls, x, y, w, h in annotations:
                            if cls == class_id:
                                bboxes.append([x, y, w, h])
                                class_labels.append(class_id)

                        if bboxes:
                            transformed = pipeline(
                                image=image,
                                bboxes=bboxes,
                                class_labels=class_labels,
                            )
                            if transformed["bboxes"]:
                                save_image(
                                    aug_img_path, transformed["image"]
                                )
                                new_annotations = [
                                    (int(cls), x, y, w, h)
                                    for (x, y, w, h), cls in zip(
                                        transformed["bboxes"],
                                        transformed["class_labels"],
                                    )
                                ]
                                write_yolo_label(aug_lbl_path, new_annotations)
                                generated += 1
                                pbar.update(1)

                except Exception:
                    pass

                attempts += 1

        logger.info("Class %d: %d new images generated", class_id, generated)
        return generated

    def augment_all(
        self, targets: Optional[Dict[int, int]] = None
    ) -> Dict[int, int]:
        """Augment all classes according to the provided target counts.

        Args:
            targets: Mapping of class id to target count.  Defaults to
                ``settings.augmentation_targets``.

        Returns:
            Dictionary mapping class id to the number of images generated.
        """
        if targets is None:
            targets = settings.augmentation_targets

        results: Dict[int, int] = {}
        for class_id, target in targets.items():
            results[class_id] = self.augment_class(class_id, target)

        return results

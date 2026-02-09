import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from app.config import settings
from app.utils.file_utils import ensure_dir
from app.utils.label_utils import parse_yolo_label
from app.utils.logger import get_logger

logger = get_logger(__name__)


class StrategicSplitter:
    """Splits a unified dataset into train / val / test sets while ensuring
    that every split contains samples from the rare target class.

    Attributes:
        unified_root: Path to the unified dataset.
        yolo_root: Path to the YOLO-formatted output.
        test_size: Fraction reserved for testing.
        val_size: Fraction reserved for validation.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        unified_root: Path = settings.UNIFIED_ROOT,
        yolo_root: Path = settings.YOLO_ROOT,
        test_size: float = settings.TEST_SIZE,
        val_size: float = settings.VAL_SIZE,
        seed: int = settings.RANDOM_SEED,
    ) -> None:
        """Initialise the StrategicSplitter.

        Args:
            unified_root: Root of the unified dataset.
            yolo_root: Destination root for the YOLO split dataset.
            test_size: Fraction of data for testing.
            val_size: Fraction of data for validation.
            seed: Random seed.
        """
        self.unified_root = Path(unified_root)
        self.yolo_root = Path(yolo_root)
        self.test_size = test_size
        self.val_size = val_size
        self.seed = seed

    def split(
        self,
        images_by_class: Dict[int, List[str]],
    ) -> Dict[str, List[str]]:
        """Perform a strategic split ensuring the target class is in every set.

        The target class (``settings.TARGET_CLASS``) images are explicitly
        distributed first, then remaining classes are split proportionally.

        Args:
            images_by_class: Mapping of class id to list of image stems.

        Returns:
            Dictionary with keys ``train``, ``val``, ``test`` mapping to lists
            of image stem strings.
        """
        random.seed(self.seed)

        splits: Dict[str, List[str]] = {"train": [], "val": [], "test": []}

        target = settings.TARGET_CLASS
        target_images = images_by_class.get(target, []).copy()
        random.shuffle(target_images)
        n_target = len(target_images)

        logger.info(
            "Target class %d: %d images total", target, n_target
        )

        if n_target < 6:
            n_test = max(1, n_target // 5)
            n_val = max(1, n_target // 5)
        else:
            n_test = max(2, int(n_target * self.test_size))
            n_val = max(2, int(n_target * self.val_size))

        splits["test"].extend(target_images[:n_test])
        splits["val"].extend(target_images[n_test : n_test + n_val])
        splits["train"].extend(target_images[n_test + n_val :])

        logger.info(
            "Target class split - Train: %d, Val: %d, Test: %d",
            len(splits["train"]),
            n_val,
            n_test,
        )

        for class_id, stems in images_by_class.items():
            if class_id == target:
                continue
            stems_copy = stems.copy()
            random.shuffle(stems_copy)
            n_t = int(len(stems_copy) * self.test_size)
            n_v = int(len(stems_copy) * self.val_size)
            splits["test"].extend(stems_copy[:n_t])
            splits["val"].extend(stems_copy[n_t : n_t + n_v])
            splits["train"].extend(stems_copy[n_t + n_v :])

        for key in splits:
            splits[key] = list(set(splits[key]))

        logger.info(
            "Final split - Train: %d, Val: %d, Test: %d",
            len(splits["train"]),
            len(splits["val"]),
            len(splits["test"]),
        )

        return splits

    def copy_to_yolo(
        self,
        splits: Dict[str, List[str]],
    ) -> Dict[str, Dict[int, int]]:
        """Copy images and labels from unified root into the YOLO directory
        structure according to the provided splits.

        Args:
            splits: Dictionary mapping split name to list of image stems.

        Returns:
            Per-split class counts ``{split: {class_id: count}}``.
        """
        unified_images = self.unified_root / "images"
        unified_labels = self.unified_root / "labels"

        stats_per_split: Dict[str, Dict[int, int]] = {}

        for split_name, img_stems in splits.items():
            logger.info("Copying %s: %d images", split_name, len(img_stems))

            dest_img = ensure_dir(self.yolo_root / split_name / "images")
            dest_lbl = ensure_dir(self.yolo_root / split_name / "labels")

            class_counts: Dict[int, int] = defaultdict(int)
            copied = 0

            for stem in tqdm(img_stems, desc=f"Copying {split_name}"):
                src_img = unified_images / f"{stem}.jpg"
                if not src_img.exists():
                    continue

                shutil.copy2(src_img, dest_img / src_img.name)

                src_lbl = unified_labels / f"{stem}.txt"
                dst_lbl = dest_lbl / f"{stem}.txt"

                if src_lbl.exists():
                    shutil.copy2(src_lbl, dst_lbl)
                    annotations = parse_yolo_label(src_lbl)
                    if not annotations:
                        class_counts[-1] += 1
                    else:
                        for cls_id, *_ in annotations:
                            class_counts[cls_id] += 1
                else:
                    dst_lbl.write_text("")
                    class_counts[-1] += 1

                copied += 1

            stats_per_split[split_name] = dict(class_counts)
            logger.info("%s: %d files copied", split_name, copied)

        return stats_per_split

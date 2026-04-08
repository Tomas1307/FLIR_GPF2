import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

SPLITS = ("train", "val", "test")


class DatasetValidator:
    """Validates a YOLO-formatted dataset for common issues such as data
    leakage, missing labels, and incorrect augmentation placement.

    Attributes:
        yolo_root: Path to the YOLO dataset root directory.
    """

    def __init__(self, yolo_root: Path = settings.YOLO_ROOT) -> None:
        """Initialise the DatasetValidator.

        Args:
            yolo_root: Root of the YOLO dataset (must contain train/val/test
                subdirectories each with images/ and labels/ folders).
        """
        self.yolo_root = Path(yolo_root)

    def _image_stems(self, split: str) -> Set[str]:
        """Return all image stems for a given split."""
        img_dir = self.yolo_root / split / "images"
        if not img_dir.exists():
            return set()
        return {p.stem for p in img_dir.iterdir() if p.suffix in (".jpg", ".png", ".jpeg")}

    def _file_md5(self, path: Path) -> str:
        """Compute MD5 hash of a file."""
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def check_data_leakage(self) -> bool:
        """Check for duplicate image stems or content hashes across splits.

        Returns:
            ``True`` if no leakage is found.
        """
        passed = True
        stems_by_split: Dict[str, Set[str]] = {}
        for split in SPLITS:
            stems_by_split[split] = self._image_stems(split)

        for i, s1 in enumerate(SPLITS):
            for s2 in SPLITS[i + 1 :]:
                overlap = stems_by_split[s1] & stems_by_split[s2]
                if overlap:
                    logger.error(
                        "LEAKAGE: %d stems shared between %s and %s: %s",
                        len(overlap), s1, s2, list(overlap)[:10],
                    )
                    passed = False

        hash_to_split: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        for split in SPLITS:
            img_dir = self.yolo_root / split / "images"
            if not img_dir.exists():
                continue
            for img in img_dir.iterdir():
                if img.suffix in (".jpg", ".png", ".jpeg"):
                    h = self._file_md5(img)
                    hash_to_split[h].append((split, img.stem))

        for h, locations in hash_to_split.items():
            splits_with = {loc[0] for loc in locations}
            if len(splits_with) > 1:
                logger.error(
                    "LEAKAGE (content hash): same image in %s — %s",
                    splits_with, locations[:5],
                )
                passed = False

        if passed:
            logger.info("Data leakage check: PASSED")
        return passed

    def check_split_proportions(self) -> bool:
        """Verify train/val/test sizes are approximately 70/15/15.

        Returns:
            ``True`` if proportions are within tolerance.
        """
        counts = {s: len(self._image_stems(s)) for s in SPLITS}
        total = sum(counts.values())
        if total == 0:
            logger.error("No images found in any split")
            return False

        ratios = {s: counts[s] / total for s in SPLITS}
        logger.info(
            "Split proportions — train: %.1f%%, val: %.1f%%, test: %.1f%% (total: %d)",
            ratios["train"] * 100, ratios["val"] * 100, ratios["test"] * 100, total,
        )

        passed = True
        if ratios["train"] < 0.55:
            logger.warning("Train split is unusually small: %.1f%%", ratios["train"] * 100)
            passed = False
        for s in ("val", "test"):
            if ratios[s] < 0.05 or ratios[s] > 0.30:
                logger.warning("%s split ratio out of range: %.1f%%", s, ratios[s] * 100)
                passed = False

        if passed:
            logger.info("Split proportions check: PASSED")
        return passed

    def check_target_class_presence(self) -> bool:
        """Confirm the target class exists in all three splits.

        Returns:
            ``True`` if the target class is present in every split.
        """
        target = settings.TARGET_CLASS
        passed = True
        for split in SPLITS:
            lbl_dir = self.yolo_root / split / "labels"
            if not lbl_dir.exists():
                logger.error("Missing labels directory for %s", split)
                passed = False
                continue
            found = False
            for lbl in lbl_dir.iterdir():
                if not lbl.suffix == ".txt":
                    continue
                text = lbl.read_text().strip()
                for line in text.split("\n"):
                    if line.strip() and int(line.split()[0]) == target:
                        found = True
                        break
                if found:
                    break
            if not found:
                logger.error("Target class %d NOT found in %s", target, split)
                passed = False

        if passed:
            logger.info("Target class presence check: PASSED")
        return passed

    def check_augmentation_isolation(self) -> bool:
        """Verify augmented images (``aug_*``) exist ONLY in train.

        Returns:
            ``True`` if no augmented images are in val or test.
        """
        passed = True
        for split in ("val", "test"):
            stems = self._image_stems(split)
            aug_stems = {s for s in stems if s.startswith("aug_")}
            if aug_stems:
                logger.error(
                    "Augmented images found in %s: %d files (e.g. %s)",
                    split, len(aug_stems), list(aug_stems)[:5],
                )
                passed = False

        if passed:
            logger.info("Augmentation isolation check: PASSED")
        return passed

    def check_label_integrity(self) -> bool:
        """Check image<->label pairing and class ID validity.

        Returns:
            ``True`` if all checks pass.
        """
        passed = True
        for split in SPLITS:
            img_dir = self.yolo_root / split / "images"
            lbl_dir = self.yolo_root / split / "labels"
            if not img_dir.exists() or not lbl_dir.exists():
                continue

            img_stems = {p.stem for p in img_dir.iterdir() if p.suffix in (".jpg", ".png", ".jpeg")}
            lbl_stems = {p.stem for p in lbl_dir.iterdir() if p.suffix == ".txt"}

            missing_labels = img_stems - lbl_stems
            if missing_labels:
                logger.warning(
                    "%s: %d images without labels (e.g. %s)",
                    split, len(missing_labels), list(missing_labels)[:5],
                )

            orphan_labels = lbl_stems - img_stems
            if orphan_labels:
                logger.warning(
                    "%s: %d orphan labels without images (e.g. %s)",
                    split, len(orphan_labels), list(orphan_labels)[:5],
                )
                passed = False

            for lbl_file in lbl_dir.iterdir():
                if lbl_file.suffix != ".txt":
                    continue
                text = lbl_file.read_text().strip()
                if not text:
                    continue
                for line_num, line in enumerate(text.split("\n"), 1):
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        cls_id = float(parts[0])
                        if cls_id != int(cls_id):
                            logger.error(
                                "%s/%s line %d: decimal class ID %.2f",
                                split, lbl_file.name, line_num, cls_id,
                            )
                            passed = False
                    except ValueError:
                        logger.error(
                            "%s/%s line %d: invalid class ID '%s'",
                            split, lbl_file.name, line_num, parts[0],
                        )
                        passed = False

        if passed:
            logger.info("Label integrity check: PASSED")
        return passed

    def get_class_distribution(self) -> Dict[str, Dict[int, int]]:
        """Return per-split object counts by class.

        Returns:
            ``{split: {class_id: count}}``.
        """
        dist: Dict[str, Dict[int, int]] = {}
        for split in SPLITS:
            lbl_dir = self.yolo_root / split / "labels"
            counts: Dict[int, int] = defaultdict(int)
            if lbl_dir.exists():
                for lbl in lbl_dir.iterdir():
                    if lbl.suffix != ".txt":
                        continue
                    text = lbl.read_text().strip()
                    if not text:
                        counts[-1] = counts.get(-1, 0) + 1
                        continue
                    for line in text.split("\n"):
                        parts = line.strip().split()
                        if parts:
                            try:
                                counts[int(float(parts[0]))] += 1
                            except ValueError:
                                pass
            dist[split] = dict(counts)

        for split in SPLITS:
            logger.info("Class distribution [%s]: %s", split, dist.get(split, {}))
        return dist

    def validate_all(self) -> bool:
        """Run all validation checks.

        Returns:
            ``True`` if every check passes.
        """
        logger.info("Running dataset validation on %s", self.yolo_root)
        results = [
            self.check_data_leakage(),
            self.check_split_proportions(),
            self.check_target_class_presence(),
            self.check_augmentation_isolation(),
            self.check_label_integrity(),
        ]
        self.get_class_distribution()

        all_passed = all(results)
        if all_passed:
            logger.info("ALL VALIDATION CHECKS PASSED")
        else:
            logger.error("SOME VALIDATION CHECKS FAILED")
        return all_passed

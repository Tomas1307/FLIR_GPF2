from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from app.utils.label_utils import parse_yolo_label
from app.utils.file_utils import glob_images
from app.utils.logger import get_logger

logger = get_logger(__name__)


def count_classes(label_dir: Path) -> Dict[int, int]:
    """Count the number of object instances per class in a label directory.

    Background images (empty label files) are counted under class id ``-1``.

    Args:
        label_dir: Directory containing YOLO-format ``.txt`` label files.

    Returns:
        Dictionary mapping class id to instance count.
    """
    counts: Dict[int, int] = defaultdict(int)
    if not label_dir.exists():
        logger.warning("Label directory does not exist: %s", label_dir)
        return dict(counts)

    for txt_path in label_dir.glob("*.txt"):
        annotations = parse_yolo_label(txt_path)
        if not annotations:
            counts[-1] += 1
        else:
            for class_id, *_ in annotations:
                counts[class_id] += 1

    return dict(counts)


def compute_distribution(
    dataset_root: Path,
    splits: Tuple[str, ...] = ("train", "val", "test"),
) -> Dict[str, Dict[int, int]]:
    """Compute per-split class distributions for a YOLO dataset.

    Args:
        dataset_root: Root of the dataset containing split sub-directories.
        splits: Names of the splits to analyse.

    Returns:
        Nested dictionary ``{split_name: {class_id: count}}``.
    """
    distribution: Dict[str, Dict[int, int]] = {}
    for split in splits:
        label_dir = dataset_root / split / "labels"
        distribution[split] = count_classes(label_dir)
    return distribution


def images_by_class(
    image_dir: Path,
    label_dir: Path,
) -> Dict[int, List[str]]:
    """Group image stems by the classes they contain.

    Args:
        image_dir: Directory containing images.
        label_dir: Directory containing matching YOLO label files.

    Returns:
        Dictionary mapping class id to a list of image stems.
    """
    result: Dict[int, List[str]] = defaultdict(list)
    for img_path in glob_images(image_dir):
        lbl_path = label_dir / f"{img_path.stem}.txt"
        annotations = parse_yolo_label(lbl_path)
        if not annotations:
            result[-1].append(img_path.stem)
        else:
            seen_classes = set()
            for class_id, *_ in annotations:
                if class_id not in seen_classes:
                    result[class_id].append(img_path.stem)
                    seen_classes.add(class_id)
    return dict(result)

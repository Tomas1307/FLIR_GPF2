from pathlib import Path
from typing import List, Tuple


def parse_yolo_label(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """Parse a YOLO-format label file into a list of annotations.

    Each annotation is a tuple of ``(class_id, x_center, y_center, width, height)``
    with normalised coordinates.  Empty files return an empty list.

    Args:
        label_path: Path to the ``.txt`` label file.

    Returns:
        List of parsed annotations.
    """
    annotations: List[Tuple[int, float, float, float, float]] = []
    if not label_path.exists():
        return annotations

    text = label_path.read_text().strip()
    if not text:
        return annotations

    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            annotations.append((class_id, x_center, y_center, width, height))
        except (ValueError, IndexError):
            continue

    return annotations


def write_yolo_label(
    label_path: Path,
    annotations: List[Tuple[int, float, float, float, float]],
) -> None:
    """Write annotations to a YOLO-format label file.

    Args:
        label_path: Destination ``.txt`` file path.
        annotations: List of ``(class_id, x_center, y_center, width, height)`` tuples.
    """
    lines = [
        f"{int(cls)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
        for cls, x, y, w, h in annotations
    ]
    label_path.write_text("\n".join(lines))

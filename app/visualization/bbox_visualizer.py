from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from app.config import settings
from app.utils.label_utils import parse_yolo_label
from app.utils.logger import get_logger

logger = get_logger(__name__)


class BBoxVisualizer:
    """Draws bounding boxes on images and displays a grid of samples.

    Attributes:
        class_names: Mapping of class id to display name.
    """

    def __init__(
        self, class_names: Optional[Dict[int, str]] = None
    ) -> None:
        """Initialise the BBoxVisualizer.

        Args:
            class_names: Mapping of class id to human-readable name.
                Defaults to ``settings.CLASS_NAMES``.
        """
        self.class_names = class_names or settings.CLASS_NAMES

    def draw_boxes(
        self,
        image: Image.Image,
        label_path: Path,
        filter_class: Optional[int] = None,
        outline: str = "red",
        width: int = 2,
    ) -> Image.Image:
        """Draw YOLO bounding boxes on a PIL image.

        Args:
            image: PIL Image in RGB mode.
            label_path: Path to the YOLO label file.
            filter_class: If set, only draw boxes for this class id.
            outline: Box outline colour.
            width: Box outline width in pixels.

        Returns:
            The image with drawn bounding boxes.
        """
        draw = ImageDraw.Draw(image)
        w, h = image.size

        annotations = parse_yolo_label(label_path)
        for cls_id, x_c, y_c, bw, bh in annotations:
            if filter_class is not None and cls_id != filter_class:
                continue
            cx, cy = x_c * w, y_c * h
            bw_px, bh_px = bw * w, bh * h
            x1 = cx - bw_px / 2
            y1 = cy - bh_px / 2
            x2 = cx + bw_px / 2
            y2 = cy + bh_px / 2
            draw.rectangle([x1, y1, x2, y2], outline=outline, width=width)

        return image

    def show_grid(
        self,
        image_dir: Path,
        label_dir: Path,
        stems: List[str],
        title: str = "Sample Images",
        cols: int = 5,
    ) -> None:
        """Display a grid of images with bounding boxes.

        Args:
            image_dir: Directory containing images.
            label_dir: Directory containing labels.
            stems: List of image stems to display.
            title: Super-title for the figure.
            cols: Number of columns in the grid.
        """
        n = len(stems)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
        if rows == 1:
            axes = [axes] if cols == 1 else list(axes)
        else:
            axes = axes.flatten()

        for i, stem in enumerate(stems):
            ax = axes[i]
            img_path = image_dir / f"{stem}.jpg"
            lbl_path = label_dir / f"{stem}.txt"

            if not img_path.exists():
                ax.axis("off")
                continue

            img = Image.open(img_path).convert("RGB")
            img = self.draw_boxes(img, lbl_path)

            ax.imshow(img)
            ax.set_title(stem, fontsize=8)
            ax.axis("off")

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()

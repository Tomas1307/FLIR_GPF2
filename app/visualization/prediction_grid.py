import os
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

from app.adapters.yolo_adapter import YoloAdapter
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class PredictionGrid:
    """Displays a grid of model predictions for images containing the
    target class.

    Attributes:
        adapter: The YOLO model adapter used for prediction.
        class_index: Class id to filter images by.
    """

    def __init__(
        self,
        adapter: YoloAdapter,
        class_index: int = settings.TARGET_CLASS,
    ) -> None:
        """Initialise the PredictionGrid.

        Args:
            adapter: YOLO adapter wrapping the model.
            class_index: Class id to filter test images by.
        """
        self.adapter = adapter
        self.class_index = class_index

    def show(
        self,
        dataset_path: str,
        count: int = 25,
        image_size: int = settings.IMAGE_SIZE,
        save_path: str = "grid_predictions.png",
    ) -> None:
        """Show a 5-column grid of predictions on test images.

        Args:
            dataset_path: Root of the YOLO dataset.
            count: Maximum number of images to display.
            image_size: Image size for inference.
            save_path: File path to save the grid image.
        """
        labels_dir = os.path.join(dataset_path, "test", "labels")
        images_dir = os.path.join(dataset_path, "test", "images")

        matching = []
        for fname in os.listdir(labels_dir):
            label_path = os.path.join(labels_dir, fname)
            with open(label_path, "r") as f:
                lines = f.readlines()
                if any(
                    int(line.split()[0]) == self.class_index for line in lines
                ):
                    matching.append(fname.replace(".txt", ".jpg"))

        selected = random.sample(matching, min(count, len(matching)))

        cols = 5
        rows = (len(selected) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(18, 3.6 * rows))
        fig.suptitle(
            f"Predictions - class {self.class_index}", fontsize=20
        )

        flat_axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for ax, img_name in zip(flat_axes, selected):
            img_path = os.path.join(images_dir, img_name)
            results = self.adapter.predict(img_path, imgsz=image_size)
            pred_img = results[0].plot()
            pred_rgb = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)

            ax.imshow(pred_rgb)
            ax.set_title(img_name, fontsize=8)
            ax.axis("off")

        for j in range(len(selected), len(flat_axes)):
            flat_axes[j].axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.show()
        logger.info("Grid saved to %s", save_path)

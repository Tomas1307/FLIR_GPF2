import random
from pathlib import Path
from typing import List

import cv2
import matplotlib.pyplot as plt

from app.preprocessing.preprocessing_pipeline import PreprocessingPipeline
from app.schemas.preprocessing_schema import PreprocessingConfig
from app.utils.image_utils import bgr_to_rgb, load_image
from app.utils.file_utils import glob_images
from app.utils.logger import get_logger

logger = get_logger(__name__)


class PreprocessingComparison:
    """Displays side-by-side comparisons of original vs preprocessed images.

    Attributes:
        pipeline: The preprocessing pipeline to demonstrate.
    """

    def __init__(
        self, config: PreprocessingConfig = PreprocessingConfig()
    ) -> None:
        """Initialise the PreprocessingComparison.

        Args:
            config: Preprocessing configuration parameters.
        """
        self.pipeline = PreprocessingPipeline(config)

    def show(
        self,
        dataset_path: str,
        num_images: int = 5,
        seed: int = 42,
    ) -> None:
        """Show preprocessing results on a random sample of images.

        For each image three panels are shown:
          1. Original image.
          2. After noise reduction.
          3. After contrast enhancement (CLAHE).

        Args:
            dataset_path: Root of the YOLO dataset.
            num_images: Number of sample images.
            seed: Random seed for sample selection.
        """
        random.seed(seed)
        img_dir = Path(dataset_path) / "train" / "images"
        all_images = glob_images(img_dir)

        if not all_images:
            logger.warning("No images found in %s", img_dir)
            return

        sample: List[Path] = random.sample(
            all_images, min(num_images, len(all_images))
        )

        for img_path in sample:
            try:
                original = load_image(img_path)
            except FileNotFoundError:
                continue

            denoised, noise_detected = self.pipeline.noise_filter.apply(
                original
            )
            enhanced = self.pipeline.contrast_enhancer.apply(denoised)

            status = (
                "Noise detected and reduced"
                if noise_detected
                else "No significant noise"
            )

            original_rgb = bgr_to_rgb(original)
            denoised_rgb = bgr_to_rgb(denoised)
            enhanced_rgb = bgr_to_rgb(enhanced)

            fig, axs = plt.subplots(1, 3, figsize=(18, 6))
            axs[0].imshow(original_rgb)
            axs[0].set_title("Original")
            axs[0].axis("off")

            axs[1].imshow(denoised_rgb)
            axs[1].set_title(f"Noise Reduction\n({status})")
            axs[1].axis("off")

            axs[2].imshow(enhanced_rgb)
            axs[2].set_title("Contrast Enhancement (CLAHE)")
            axs[2].axis("off")

            plt.suptitle(f"Preprocessing: {img_path.name}", fontsize=16)
            plt.tight_layout()
            plt.show()

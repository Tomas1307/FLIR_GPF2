import shutil
from pathlib import Path

from tqdm import tqdm

from app.preprocessing.preprocessing_pipeline import PreprocessingPipeline
from app.schemas.preprocessing_schema import PreprocessingConfig
from app.utils.file_utils import ensure_dir, glob_images
from app.utils.image_utils import load_image, save_image
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DatasetPreprocessor:
    """Applies the preprocessing pipeline to every image in a YOLO dataset
    and writes the results to a new output directory.

    Labels are copied as-is because preprocessing does not alter bounding
    boxes.

    Attributes:
        pipeline: The composed preprocessing pipeline instance.
    """

    def __init__(
        self,
        config: PreprocessingConfig = PreprocessingConfig(),
    ) -> None:
        """Initialise the DatasetPreprocessor.

        Args:
            config: Preprocessing configuration.
        """
        self.pipeline = PreprocessingPipeline(config)

    def preprocess(
        self,
        source_path: Path,
        output_path: Path,
    ) -> Path:
        """Preprocess all images in *source_path* and save to *output_path*.

        The directory structure (``train/images``, ``train/labels``, etc.) is
        replicated in *output_path*.  If a ``dataset.yaml`` file exists in the
        source it is copied verbatim.

        Args:
            source_path: Root of the original YOLO dataset.
            output_path: Destination root for the preprocessed dataset.

        Returns:
            The *output_path* for convenient chaining.

        Raises:
            FileNotFoundError: If *source_path* does not exist.
        """
        source_path = Path(source_path)
        output_path = Path(output_path)

        if not source_path.exists():
            raise FileNotFoundError(
                f"Source dataset does not exist: {source_path}"
            )

        ensure_dir(output_path)

        yaml_src = source_path / "dataset.yaml"
        if yaml_src.exists():
            shutil.copy(yaml_src, output_path / "dataset.yaml")
            logger.info("Copied dataset.yaml")

        for split in ("train", "val", "test"):
            img_src_dir = source_path / split / "images"
            lbl_src_dir = source_path / split / "labels"

            if not img_src_dir.exists():
                logger.warning("Image directory not found: %s", img_src_dir)
                continue

            img_out_dir = ensure_dir(output_path / split / "images")
            lbl_out_dir = ensure_dir(output_path / split / "labels")

            all_images = glob_images(img_src_dir)
            logger.info("Preprocessing %d images in %s", len(all_images), split)

            noise_count = 0

            for img_path in tqdm(all_images, desc=f"Preprocessing {split}"):
                try:
                    image = load_image(img_path)
                except FileNotFoundError:
                    logger.warning("Could not read image: %s", img_path)
                    continue

                processed, noise_detected = self.pipeline.process(image)
                if noise_detected:
                    noise_count += 1

                save_image(img_out_dir / img_path.name, processed)

                lbl_path = lbl_src_dir / f"{img_path.stem}.txt"
                out_lbl = lbl_out_dir / f"{img_path.stem}.txt"
                if lbl_path.exists():
                    shutil.copy(lbl_path, out_lbl)
                else:
                    out_lbl.write_text("")

            logger.info(
                "Completed %s: %d images, %d with noise detected",
                split,
                len(all_images),
                noise_count,
            )

        logger.info("Preprocessing complete. Output: %s", output_path)
        return output_path

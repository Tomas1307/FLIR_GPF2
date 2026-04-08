import shutil
from pathlib import Path
from typing import Dict

from tqdm import tqdm

from app.augmentation.class_augmentor import ClassAugmentor
from app.config import settings
from app.core.class_balancer import ClassBalancer
from app.core.dataset_unifier import DatasetUnifier
from app.core.duplicate_remover import DuplicateRemover
from app.core.label_cleaner import LabelCleaner
from app.core.strategic_splitter import StrategicSplitter
from app.preprocessing.preprocessing_pipeline import PreprocessingPipeline
from app.schemas.preprocessing_schema import PreprocessingConfig
from app.utils.dataset_stats import images_by_class
from app.utils.file_utils import ensure_dir, glob_images
from app.utils.image_utils import load_image, save_image
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DataPipelineFacade:
    """Facade orchestrating the full data-preparation pipeline.

    Pipeline stages:
      1. Unify raw data into a single directory.
      2. Analyse the unified dataset.
      3. Strategically split with target-class guarantees.
      4. Copy splits into YOLO directory structure.
      5. Preprocess images (noise filter + CLAHE) — before augmentation
         so that augmented copies inherit the enhancement.
      6. Augment under-represented classes.
      7. Clean decimal class ids in labels.
      8. Remove duplicate images.
      9. Remove excess augmented warehouses.

    Attributes:
        data_root: Raw data directory.
        unified_root: Unified output directory.
        yolo_root: Final YOLO dataset directory.
    """

    def __init__(
        self,
        data_root: Path = settings.DATA_ROOT,
        unified_root: Path = settings.UNIFIED_ROOT,
        yolo_root: Path = settings.YOLO_ROOT,
    ) -> None:
        """Initialise the DataPipelineFacade.

        Args:
            data_root: Path to the raw data.
            unified_root: Path for the unified intermediate dataset.
            yolo_root: Path for the final YOLO dataset.
        """
        self.data_root = Path(data_root)
        self.unified_root = Path(unified_root)
        self.yolo_root = Path(yolo_root)

    def _preprocess_split_images(self, pipeline: PreprocessingPipeline) -> None:
        """Apply preprocessing in-place to all images already in YOLO_ROOT.

        Only processes original (non-augmented) images because this runs
        *before* augmentation.  Images are overwritten in place so that
        subsequent augmentation inherits the enhancement.
        """
        for split in ("train", "val", "test"):
            img_dir = self.yolo_root / split / "images"
            if not img_dir.exists():
                continue

            all_images = glob_images(img_dir)
            if not all_images:
                continue

            noise_count = 0
            for img_path in tqdm(all_images, desc=f"Preprocessing {split}"):
                try:
                    image = load_image(img_path)
                except FileNotFoundError:
                    logger.warning("Could not read image: %s", img_path)
                    continue

                processed, noise_detected = pipeline.process(image)
                if noise_detected:
                    noise_count += 1

                save_image(img_path, processed)

            logger.info(
                "Preprocessed %s: %d images, %d with noise detected",
                split, len(all_images), noise_count,
            )

    def run(self) -> Dict[str, Dict[int, int]]:
        """Execute all data preparation stages.

        Returns:
            Per-split class distribution after the full pipeline.
        """
        # -- Clean previous outputs ------------------------------------------
        if self.unified_root.exists():
            shutil.rmtree(self.unified_root)
        if self.yolo_root.exists():
            shutil.rmtree(self.yolo_root)

        for split in ("train", "val", "test"):
            ensure_dir(self.yolo_root / split / "images")
            ensure_dir(self.yolo_root / split / "labels")

        # 1. Unify
        logger.info("Stage 1/9: Unifying dataset")
        unifier = DatasetUnifier(self.data_root, self.unified_root)
        unifier.unify()

        # 2. Analyse
        logger.info("Stage 2/9: Analysing unified dataset")
        ibc = images_by_class(
            self.unified_root / "images", self.unified_root / "labels"
        )

        # 3. Strategic split
        logger.info("Stage 3/9: Strategic splitting")
        splitter = StrategicSplitter(self.unified_root, self.yolo_root)
        splits = splitter.split(ibc)

        # 4. Copy to YOLO
        logger.info("Stage 4/9: Copying splits to YOLO structure")
        stats = splitter.copy_to_yolo(splits)

        # 5. Preprocess (noise filter + CLAHE) — BEFORE augmentation
        logger.info("Stage 5/9: Preprocessing images (noise filter + CLAHE)")
        pipeline = PreprocessingPipeline(PreprocessingConfig())
        self._preprocess_split_images(pipeline)

        # 6. Augment (on already-preprocessed originals)
        logger.info("Stage 6/9: Augmenting classes")
        augmentor = ClassAugmentor(self.yolo_root)
        augmentor.augment_all()

        # 7. Clean labels
        logger.info("Stage 7/9: Cleaning labels")
        cleaner = LabelCleaner(self.yolo_root)
        cleaner.clean()

        # 8. Remove duplicates
        logger.info("Stage 8/9: Removing duplicates")
        deduper = DuplicateRemover(self.yolo_root)
        deduper.clean()

        # 9. Balance warehouses
        logger.info("Stage 9/9: Balancing warehouse class")
        balancer = ClassBalancer(self.yolo_root)
        balancer.remove_excess(class_id=1)

        logger.info("Data pipeline complete")
        return stats

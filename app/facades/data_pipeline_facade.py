import shutil
from pathlib import Path
from typing import Dict

from app.augmentation.class_augmentor import ClassAugmentor
from app.config import settings
from app.core.class_balancer import ClassBalancer
from app.core.dataset_unifier import DatasetUnifier
from app.core.duplicate_remover import DuplicateRemover
from app.core.label_cleaner import LabelCleaner
from app.core.strategic_splitter import StrategicSplitter
from app.utils.dataset_stats import images_by_class
from app.utils.file_utils import ensure_dir
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DataPipelineFacade:
    """Facade orchestrating the full data-preparation pipeline.

    Pipeline stages:
      1. Unify raw data into a single directory.
      2. Analyse the unified dataset.
      3. Strategically split with target-class guarantees.
      4. Copy splits into YOLO directory structure.
      5. Augment under-represented classes.
      6. Clean decimal class ids in labels.
      7. Remove duplicate images.
      8. Remove excess augmented warehouses.

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
        logger.info("Stage 1/8: Unifying dataset")
        unifier = DatasetUnifier(self.data_root, self.unified_root)
        unifier.unify()

        # 2. Analyse
        logger.info("Stage 2/8: Analysing unified dataset")
        ibc = images_by_class(
            self.unified_root / "images", self.unified_root / "labels"
        )

        # 3. Strategic split
        logger.info("Stage 3/8: Strategic splitting")
        splitter = StrategicSplitter(self.unified_root, self.yolo_root)
        splits = splitter.split(ibc)

        # 4. Copy to YOLO
        logger.info("Stage 4/8: Copying splits to YOLO structure")
        stats = splitter.copy_to_yolo(splits)

        # 5. Augment
        logger.info("Stage 5/8: Augmenting classes")
        augmentor = ClassAugmentor(self.yolo_root)
        augmentor.augment_all()

        # 6. Clean labels
        logger.info("Stage 6/8: Cleaning labels")
        cleaner = LabelCleaner(self.yolo_root)
        cleaner.clean()

        # 7. Remove duplicates
        logger.info("Stage 7/8: Removing duplicates")
        deduper = DuplicateRemover(self.yolo_root)
        deduper.clean()

        # 8. Balance warehouses
        logger.info("Stage 8/8: Balancing warehouse class")
        balancer = ClassBalancer(self.yolo_root)
        balancer.remove_excess(class_id=1)

        logger.info("Data pipeline complete")
        return stats

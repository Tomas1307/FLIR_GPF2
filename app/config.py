from pathlib import Path
from typing import Dict, Tuple

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Singleton configuration for the illegal mining detection pipeline.

    All environment-tunable parameters are declared here and accessed
    throughout the codebase as ``settings.VARIABLE_NAME``.  Values can be
    overridden via environment variables or a ``.env`` file.

    Attributes:
        DATA_ROOT: Path to the raw data directory containing images and labels.
        UNIFIED_ROOT: Path where the unified (merged) dataset is stored.
        YOLO_ROOT: Path to the YOLO-formatted dataset used for training.
        PREPROCESSED_ROOT: Path to the preprocessed dataset output.
        CLASS_NAMES: Mapping from class id to human-readable name.
        TARGET_CLASS: The class id of primary interest (illegal mining).
        NUM_CLASSES: Total number of object classes (excluding background).
        AUG_TARGET_MINING: Augmentation target count for mining class.
        AUG_TARGET_VEHICLES: Augmentation target count for vehicles class.
        AUG_TARGET_WAREHOUSES: Augmentation target count for warehouses class.
        AUG_TARGET_ROADS: Augmentation target count for roads class.
        AUG_TARGET_RIVERS: Augmentation target count for rivers class.
        AUG_TARGET_BACKGROUND: Augmentation target count for background class.
        TEST_SIZE: Fraction of data reserved for testing.
        VAL_SIZE: Fraction of data reserved for validation.
        NOISE_THRESHOLD: Minimum ratio of extreme pixels to trigger noise filter.
        MEDIAN_KERNEL: Kernel size for the median blur noise filter.
        CLAHE_CLIP: Clip limit for CLAHE contrast enhancement.
        CLAHE_TILE_GRID: Tile grid size for CLAHE.
        MODEL_WEIGHTS: Pretrained YOLO weights file name.
        BATCH_SIZE: Training batch size.
        LR0: Initial learning rate.
        LRF: Final learning rate ratio.
        EPOCHS: Number of training epochs.
        PATIENCE: Early stopping patience.
        DROPOUT: Dropout rate during training.
        WEIGHT_DECAY: L2 regularisation weight decay.
        MOSAIC: Mosaic augmentation probability.
        MIXUP: MixUp augmentation probability.
        IMAGE_SIZE: Training image resolution.
        COS_LR: Whether to use cosine learning rate scheduling.
        FINETUNE_LR: Learning rate for fine-tuning.
        FINETUNE_EPOCHS: Number of fine-tuning epochs.
        FREEZE_LAYERS: Number of backbone layers to freeze during fine-tuning.
        CONF_THRESHOLD_RECALL: Confidence threshold optimised for recall.
        CONF_THRESHOLD_NORMAL: Standard confidence threshold.
        IOU_THRESHOLD: IoU threshold for NMS.
        GPU_MEMORY_GB: Available GPU memory in gigabytes.
        RANDOM_SEED: Global random seed for reproducibility.
        BODEGA_REMOVAL_RATIO: Fraction of augmented warehouse images to remove.
        WORKERS: Number of data-loader workers.
    """

    # -- Paths ---------------------------------------------------------------
    DATA_ROOT: Path = Field(default=Path("data"))
    UNIFIED_ROOT: Path = Field(default=Path("data_unified"))
    YOLO_ROOT: Path = Field(default=Path("yolo_dataset"))
    PREPROCESSED_ROOT: Path = Field(default=Path("preprocessed"))

    # -- Class definitions ---------------------------------------------------
    CLASS_NAMES: Dict[int, str] = Field(default={
        -1: "Background",
        0: "Vehicles",
        1: "Warehouses",
        2: "Roads",
        3: "Rivers",
        4: "Illegal Mining",
    })
    TARGET_CLASS: int = Field(default=4)
    NUM_CLASSES: int = Field(default=5)

    # -- Augmentation targets ------------------------------------------------
    AUG_TARGET_MINING: int = Field(default=3500)
    AUG_TARGET_VEHICLES: int = Field(default=2000)
    AUG_TARGET_WAREHOUSES: int = Field(default=3000)
    AUG_TARGET_ROADS: int = Field(default=3000)
    AUG_TARGET_RIVERS: int = Field(default=3000)
    AUG_TARGET_BACKGROUND: int = Field(default=2500)

    # -- Split ratios --------------------------------------------------------
    TEST_SIZE: float = Field(default=0.15)
    VAL_SIZE: float = Field(default=0.15)

    # -- Preprocessing -------------------------------------------------------
    NOISE_THRESHOLD: float = Field(default=0.01)
    MEDIAN_KERNEL: int = Field(default=3)
    CLAHE_CLIP: float = Field(default=2.0)
    CLAHE_TILE_GRID: Tuple[int, int] = Field(default=(8, 8))

    # -- Training (conservative winner) --------------------------------------
    MODEL_WEIGHTS: str = Field(default="yolo11m.pt")
    BATCH_SIZE: int = Field(default=64)
    LR0: float = Field(default=0.01)
    LRF: float = Field(default=0.01)
    EPOCHS: int = Field(default=250)
    PATIENCE: int = Field(default=25)
    DROPOUT: float = Field(default=0.1)
    WEIGHT_DECAY: float = Field(default=0.001)
    MOSAIC: float = Field(default=0.8)
    MIXUP: float = Field(default=0.0)
    IMAGE_SIZE: int = Field(default=640)
    COS_LR: bool = Field(default=True)
    SAVE_PERIOD: int = Field(default=10)

    # -- Fine-tuning ---------------------------------------------------------
    FINETUNE_LR: float = Field(default=0.0005)
    FINETUNE_EPOCHS: int = Field(default=15)
    FREEZE_LAYERS: int = Field(default=10)

    # -- Inference thresholds ------------------------------------------------
    CONF_THRESHOLD_RECALL: float = Field(default=0.15)
    CONF_THRESHOLD_NORMAL: float = Field(default=0.25)
    IOU_THRESHOLD: float = Field(default=0.6)

    # -- Recall gate ---------------------------------------------------------
    RECALL_GATE: float = Field(default=0.81)

    # -- Hardware / misc -----------------------------------------------------
    GPU_MEMORY_GB: int = Field(default=48)
    RANDOM_SEED: int = Field(default=42)
    BODEGA_REMOVAL_RATIO: float = Field(default=0.80)
    WORKERS: int = Field(default=16)

    model_config = {"env_prefix": "FLIR_", "env_file": ".env", "extra": "ignore"}

    @property
    def augmentation_targets(self) -> Dict[int, int]:
        """Return a mapping of class id to augmentation target count."""
        return {
            4: self.AUG_TARGET_MINING,
            0: self.AUG_TARGET_VEHICLES,
            1: self.AUG_TARGET_WAREHOUSES,
            2: self.AUG_TARGET_ROADS,
            3: self.AUG_TARGET_RIVERS,
            -1: self.AUG_TARGET_BACKGROUND,
        }


# Singleton instance -- import and use as ``from app.config import settings``
settings = Settings()

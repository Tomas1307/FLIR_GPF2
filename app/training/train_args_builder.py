from pathlib import Path
from typing import Any, Dict

from app.config import settings
from app.schemas.training_schema import TrainingConfig


class TrainArgsBuilder:
    """Builds the full dictionary of training arguments accepted by the
    Ultralytics YOLO ``train()`` method.

    This eliminates the duplication that existed between the hyperparameter
    search and final-training scripts, which both maintained near-identical
    copies of a ~50-key arguments dictionary.

    Attributes:
        config: The training configuration schema.
    """

    def __init__(self, config: TrainingConfig = TrainingConfig()) -> None:
        """Initialise the TrainArgsBuilder.

        Args:
            config: Training configuration parameters.
        """
        self.config = config

    def build(
        self,
        dataset_path: str,
        run_name: str,
        project: str = "runs",
    ) -> Dict[str, Any]:
        """Build the complete training arguments dictionary.

        Args:
            dataset_path: Path to the dataset root (must contain
                ``dataset.yaml``).
            run_name: Name for this training run.
            project: Project directory for saving run outputs.

        Returns:
            Dictionary of keyword arguments for ``YOLO.train()``.
        """
        yaml_path = (Path(dataset_path) / "dataset.yaml").resolve()

        return {
            # Core
            "data": str(yaml_path),
            "epochs": self.config.epochs,
            "imgsz": self.config.image_size,
            "batch": self.config.batch_size,
            "lr0": self.config.lr0,
            "lrf": self.config.lrf,
            "cos_lr": self.config.cos_lr,
            "weight_decay": self.config.weight_decay,
            "dropout": self.config.dropout,
            "mosaic": self.config.mosaic,
            "mixup": self.config.mixup,
            "patience": self.config.patience,
            "save_period": self.config.save_period,
            "optimizer": self.config.optimizer,
            "name": run_name,
            "project": str(Path(project).resolve()),
            "exist_ok": True,
            "verbose": False,
            "workers": self.config.workers,
            # Detection
            "conf": settings.CONF_THRESHOLD_RECALL,
            "iou": settings.IOU_THRESHOLD,
            "max_det": 500,
            "augment": True,
            "agnostic_nms": False,
            "retina_masks": True,
            "overlap_mask": True,
            "mask_ratio": 4,
            "boxes": True,
            # Loss weights
            "cls": 0.5,
            "box": 7.5,
            "dfl": 1.5,
            "pose": 12.0,
            "kobj": 1.0,
            "label_smoothing": 0.0,
            # Augmentation
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "degrees": 0.0,
            "translate": 0.1,
            "scale": 0.5,
            "shear": 0.0,
            "perspective": 0.0,
            "flipud": 0.0,
            "fliplr": 0.5,
            "copy_paste": 0.0,
            "auto_augment": "randaugment",
            "erasing": 0.4,
            "crop_fraction": 1.0,
        }

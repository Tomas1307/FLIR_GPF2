from pathlib import Path
from typing import Any, Tuple

from app.config import settings
from app.schemas.finetuning_schema import FinetuningConfig
from app.schemas.metrics_schema import MetricsResult
from app.training.gpu_manager import GpuManager
from app.training.metrics_extractor import MetricsExtractor
from app.training.model_factory import ModelFactory
from app.adapters.yolo_adapter import YoloAdapter
from app.utils.logger import get_logger

logger = get_logger(__name__)


class Finetuner:
    """Fine-tunes a pretrained YOLO model by freezing backbone layers and
    training the detection head with a low learning rate.

    Attributes:
        config: Fine-tuning configuration.
        gpu: Singleton GPU manager.
        factory: Model factory.
        extractor: Metrics extractor.
    """

    def __init__(self, config: FinetuningConfig) -> None:
        """Initialise the Finetuner.

        Args:
            config: Fine-tuning configuration parameters.
        """
        self.config = config
        self.gpu = GpuManager()
        self.factory = ModelFactory()
        self.extractor = MetricsExtractor()

    def finetune(
        self,
        save_dir: str = "finetuning_results",
        run_name: str = "finetuning_run",
    ) -> Tuple[YoloAdapter, MetricsResult]:
        """Run the fine-tuning procedure.

        Steps:
          1. Load the pretrained model.
          2. Train with frozen backbone and Adam optimiser.
          3. Evaluate by class.

        Args:
            save_dir: Directory to save training outputs.
            run_name: Name for this fine-tuning run.

        Returns:
            A tuple of ``(yolo_adapter, metrics_result)``.

        Raises:
            FileNotFoundError: If the model or dataset YAML is missing.
        """
        if not Path(self.config.model_path).exists():
            raise FileNotFoundError(
                f"Model not found: {self.config.model_path}"
            )
        if not Path(self.config.dataset_yaml).exists():
            raise FileNotFoundError(
                f"YAML not found: {self.config.dataset_yaml}"
            )

        self.gpu.clear_cache()

        adapter = self.factory.create_from_checkpoint(self.config.model_path)

        logger.info(
            "Fine-tuning -- LR: %s, Epochs: %d, Freeze: %d layers",
            self.config.lr,
            self.config.epochs,
            self.config.freeze_layers,
        )

        adapter.train(
            data=self.config.dataset_yaml,
            epochs=self.config.epochs,
            patience=self.config.patience,
            lr0=self.config.lr,
            imgsz=self.config.image_size,
            batch=self.config.batch_size,
            project=save_dir,
            name=run_name,
            optimizer="Adam",
            cos_lr=True,
            verbose=True,
            seed=settings.RANDOM_SEED,
            deterministic=True,
            single_cls=False,
            rect=False,
            freeze=self.config.freeze_layers,
        )

        metrics = self.extractor.evaluate_by_class(
            adapter,
            self.config.dataset_yaml,
        )

        logger.info("Fine-tuning complete")
        return adapter, metrics

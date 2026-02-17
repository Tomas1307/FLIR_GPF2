import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from app.config import settings
from app.schemas.training_schema import TrainingConfig
from app.training.gpu_manager import GpuManager
from app.training.metrics_extractor import MetricsExtractor
from app.training.model_factory import ModelFactory
from app.training.results_analyzer import ResultsAnalyzer
from app.training.train_args_builder import TrainArgsBuilder
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ConservativeTrainer:
    """Trains a YOLO model using the conservative configuration that
    achieved the best recall during hyperparameter search.

    The trainer handles model creation, training argument assembly,
    multi-threshold evaluation, and results persistence.

    Attributes:
        config: Training configuration schema.
        gpu: Singleton GPU manager.
        factory: Model factory.
        extractor: Metrics extractor.
        analyzer: Results analyzer.
        builder: Training arguments builder.
    """

    def __init__(
        self, config: TrainingConfig = TrainingConfig()
    ) -> None:
        """Initialise the ConservativeTrainer.

        Args:
            config: Training configuration parameters.
        """
        self.config = config
        self.gpu = GpuManager()
        self.factory = ModelFactory()
        self.extractor = MetricsExtractor()
        self.analyzer = ResultsAnalyzer()
        self.builder = TrainArgsBuilder(config)

    def train(
        self,
        dataset_path: str,
        output_name: str = "conservative_mining_detector",
        resume_from: Optional[str] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Run the full conservative training pipeline.

        Steps:
          1. Build training arguments.
          2. Create or resume the model.
          3. Train with early stopping.
          4. Evaluate with recall-optimised and normal thresholds.
          5. Persist metrics and a copy of the best weights.

        Args:
            dataset_path: Path to the dataset root.
            output_name: Base name for the output directory.
            resume_from: Optional checkpoint path to resume from.

        Returns:
            A tuple of ``(yolo_adapter, metrics_dict)``.

        Raises:
            FileNotFoundError: If *dataset_path* does not exist.
        """
        if not Path(dataset_path).exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        output_dir = Path(f"conservative_final_{output_name}_{timestamp}")
        output_dir.mkdir(exist_ok=True)

        run_name = f"conservative_final_{Path(dataset_path).name}"
        train_args = self.builder.build(
            dataset_path, run_name, project=str(output_dir)
        )

        logger.info("Starting conservative training on %s", dataset_path)
        logger.info(
            "Config -- Batch: %d, LR: %s, Mosaic: %s, Dropout: %s",
            self.config.batch_size,
            self.config.lr0,
            self.config.mosaic,
            self.config.dropout,
        )

        try:
            if resume_from and Path(resume_from).exists():
                logger.info("Resuming from checkpoint: %s", resume_from)
                adapter = self.factory.create_from_checkpoint(resume_from)
            else:
                adapter = self.factory.create(self.config.model_weights)

            adapter.train(**train_args)

            best_model_path = (
                output_dir / run_name / "weights" / "best.pt"
            )
            if not best_model_path.exists():
                raise FileNotFoundError(
                    f"Trained model not found: {best_model_path}"
                )

            final_adapter = self.factory.create_from_checkpoint(
                str(best_model_path)
            )

            recall_metrics = self.extractor.extract_class_metrics(
                final_adapter.validate(
                    data=train_args["data"],
                    imgsz=self.config.image_size,
                    batch=self.config.batch_size,
                    conf=settings.CONF_THRESHOLD_RECALL,
                    iou=0.5,
                    max_det=1000,
                    verbose=True,
                    plots=True,
                    save_json=True,
                ),
                "recall_optimised",
            )

            normal_metrics = self.extractor.extract_class_metrics(
                final_adapter.validate(
                    data=train_args["data"],
                    imgsz=self.config.image_size,
                    batch=self.config.batch_size,
                    conf=settings.CONF_THRESHOLD_NORMAL,
                    iou=0.7,
                    max_det=300,
                    verbose=False,
                ),
                "normal_threshold",
            )

            recommendations = self.analyzer.generate_recommendations(
                recall_metrics
            )

            all_metrics = {
                "dataset": dataset_path,
                "training_epochs": self.config.epochs,
                "model_path": str(best_model_path),
                "recall_optimised": recall_metrics.model_dump(),
                "normal_threshold": normal_metrics.model_dump(),
                "recommendations": recommendations,
            }

            metrics_file = output_dir / "conservative_mining_metrics.json"
            with open(metrics_file, "w") as f:
                json.dump(all_metrics, f, indent=2)

            final_model_path = (
                output_dir
                / f"conservative_mining_detector_recall_{recall_metrics.recall:.3f}.pt"
            )
            shutil.copy2(best_model_path, final_model_path)

            logger.info("Training complete. Model: %s", final_model_path)
            logger.info("Metrics saved to: %s", metrics_file)
            logger.info(
                "Recall (optimised): %.4f | Precision: %.4f",
                recall_metrics.recall,
                recall_metrics.precision,
            )

            return final_adapter, all_metrics

        finally:
            self.gpu.clear_cache()

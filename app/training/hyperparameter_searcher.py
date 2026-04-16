import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from app.config import settings
from app.schemas.hyperparameter_schema import HyperparameterConfig
from app.training.gpu_manager import GpuManager
from app.training.model_factory import ModelFactory
from app.training.results_analyzer import ResultsAnalyzer
from app.training.train_args_builder import TrainArgsBuilder
from app.schemas.training_schema import TrainingConfig
from app.utils.logger import get_logger

logger = get_logger(__name__)


class HyperparameterSearcher:
    """Performs a grid search over hyperparameter configurations to maximise
    recall on the target class.

    Attributes:
        dataset_paths: List of dataset root paths to evaluate.
        target_class: The class id to optimise for.
        base_epochs: Maximum epochs per run.
        gpu: Singleton GPU manager.
        factory: Model factory.
        analyzer: Results analyzer.
        output_dir: Directory for persisting search results.
        results: Accumulated search results.
    """

    def __init__(
        self,
        dataset_paths: List[str],
        target_class: int = settings.TARGET_CLASS,
        base_epochs: int = settings.EPOCHS,
        gpu_memory_gb: int = settings.GPU_MEMORY_GB,
    ) -> None:
        """Initialise the HyperparameterSearcher.

        Args:
            dataset_paths: Paths to dataset roots.
            target_class: Class id to optimise recall for.
            base_epochs: Maximum training epochs per configuration.
            gpu_memory_gb: Available GPU memory.
        """
        self.dataset_paths = dataset_paths
        self.target_class = target_class
        self.base_epochs = base_epochs
        self.gpu = GpuManager(gpu_memory_gb)
        self.factory = ModelFactory()
        self.analyzer = ResultsAnalyzer()
        self.output_dir = Path(
            f"hyperparameter_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.output_dir.mkdir(exist_ok=True)
        self.results: List[Dict[str, Any]] = []

        logger.info("Hyperparameter search initialised")
        logger.info("Target class: %d | Datasets: %d", target_class, len(dataset_paths))

    @staticmethod
    def get_default_configurations() -> List[HyperparameterConfig]:
        """Return the default set of hyperparameter configurations.

        Returns:
            List of ``HyperparameterConfig`` schemas.
        """
        # -- yolo11m configs ---------------------------------------------------
        yolo11m = [
            HyperparameterConfig(
                name="y11m_baseline",
                batch_size=64, lr0=0.01, lrf=0.01, cos_lr=True,
                weight_decay=0.001, dropout=0.1, mosaic=0.8, mixup=0.0,
                image_size=640,
            ),
            HyperparameterConfig(
                name="y11m_high_lr",
                batch_size=64, lr0=0.02, lrf=0.1, cos_lr=True,
                weight_decay=0.001, dropout=0.0, mosaic=1.0, mixup=0.2,
                image_size=640,
            ),
            HyperparameterConfig(
                name="y11m_low_lr",
                batch_size=64, lr0=0.001, lrf=0.01, cos_lr=True,
                weight_decay=0.001, dropout=0.1, mosaic=0.8, mixup=0.0,
                image_size=640,
            ),
            HyperparameterConfig(
                name="y11m_medium_lr",
                batch_size=64, lr0=0.005, lrf=0.01, cos_lr=True,
                weight_decay=0.001, dropout=0.1, mosaic=0.8, mixup=0.0,
                image_size=640,
            ),
            HyperparameterConfig(
                name="y11m_heavy_aug",
                batch_size=64, lr0=0.01, lrf=0.01, cos_lr=True,
                weight_decay=0.001, dropout=0.2, mosaic=1.0, mixup=0.3,
                image_size=640,
            ),
            HyperparameterConfig(
                name="y11m_minimal_aug",
                batch_size=64, lr0=0.01, lrf=0.01, cos_lr=True,
                weight_decay=0.001, dropout=0.0, mosaic=0.3, mixup=0.0,
                image_size=640,
            ),
            HyperparameterConfig(
                name="y11m_high_dropout",
                batch_size=48, lr0=0.01, lrf=0.01, cos_lr=True,
                weight_decay=0.001, dropout=0.3, mosaic=0.8, mixup=0.1,
                image_size=640,
            ),
            HyperparameterConfig(
                name="y11m_high_res",
                batch_size=48, lr0=0.01, lrf=0.01, cos_lr=True,
                weight_decay=0.001, dropout=0.1, mosaic=0.8, mixup=0.0,
                image_size=800,
            ),
            HyperparameterConfig(
                name="y11m_high_res_aug",
                batch_size=48, lr0=0.01, lrf=0.01, cos_lr=True,
                weight_decay=0.001, dropout=0.1, mosaic=1.0, mixup=0.2,
                image_size=800,
            ),
            HyperparameterConfig(
                name="y11m_cosine_off",
                batch_size=64, lr0=0.01, lrf=0.01, cos_lr=False,
                weight_decay=0.001, dropout=0.1, mosaic=0.8, mixup=0.0,
                image_size=640,
            ),
            HyperparameterConfig(
                name="y11m_high_wd",
                batch_size=64, lr0=0.01, lrf=0.01, cos_lr=True,
                weight_decay=0.005, dropout=0.1, mosaic=0.8, mixup=0.0,
                image_size=640,
            ),
            HyperparameterConfig(
                name="y11m_large_batch",
                batch_size=80, lr0=0.01, lrf=0.01, cos_lr=True,
                weight_decay=0.001, dropout=0.1, mosaic=0.8, mixup=0.0,
                image_size=640,
            ),
            HyperparameterConfig(
                name="y11m_adam",
                batch_size=64, lr0=0.001, lrf=0.01, cos_lr=True,
                weight_decay=0.001, dropout=0.1, mosaic=0.8, mixup=0.0,
                image_size=640, optimizer="Adam",
            ),
            HyperparameterConfig(
                name="y11m_adamw",
                batch_size=64, lr0=0.001, lrf=0.01, cos_lr=True,
                weight_decay=0.01, dropout=0.1, mosaic=0.8, mixup=0.0,
                image_size=640, optimizer="AdamW",
            ),
            HyperparameterConfig(
                name="y11m_warm_restart",
                batch_size=64, lr0=0.015, lrf=0.05, cos_lr=True,
                weight_decay=0.0005, dropout=0.1, mosaic=0.8, mixup=0.1,
                image_size=640,
            ),
            # -- yolo11l configs -----------------------------------------------
            HyperparameterConfig(
                name="y11l_baseline",
                batch_size=32, lr0=0.01, lrf=0.01, cos_lr=True,
                weight_decay=0.001, dropout=0.1, mosaic=0.8, mixup=0.0,
                image_size=640, model_weights="yolo11l.pt",
            ),
            HyperparameterConfig(
                name="y11l_high_res",
                batch_size=24, lr0=0.01, lrf=0.01, cos_lr=True,
                weight_decay=0.001, dropout=0.1, mosaic=0.8, mixup=0.0,
                image_size=800, model_weights="yolo11l.pt",
            ),
            HyperparameterConfig(
                name="y11l_adam",
                batch_size=32, lr0=0.001, lrf=0.01, cos_lr=True,
                weight_decay=0.001, dropout=0.1, mosaic=0.8, mixup=0.0,
                image_size=640, model_weights="yolo11l.pt", optimizer="Adam",
            ),
        ]
        # -- yolo26m configs ---------------------------------------------------
        yolo26m = [
            HyperparameterConfig(
                name="y26m_baseline",
                batch_size=64, lr0=0.01, lrf=0.01, cos_lr=True,
                weight_decay=0.001, dropout=0.1, mosaic=0.8, mixup=0.0,
                image_size=640, model_weights="yolo26m.pt",
            ),
            HyperparameterConfig(
                name="y26m_high_lr",
                batch_size=64, lr0=0.02, lrf=0.1, cos_lr=True,
                weight_decay=0.001, dropout=0.0, mosaic=1.0, mixup=0.2,
                image_size=640, model_weights="yolo26m.pt",
            ),
            HyperparameterConfig(
                name="y26m_low_lr",
                batch_size=64, lr0=0.001, lrf=0.01, cos_lr=True,
                weight_decay=0.001, dropout=0.1, mosaic=0.8, mixup=0.0,
                image_size=640, model_weights="yolo26m.pt",
            ),
            HyperparameterConfig(
                name="y26m_medium_lr",
                batch_size=64, lr0=0.005, lrf=0.01, cos_lr=True,
                weight_decay=0.001, dropout=0.1, mosaic=0.8, mixup=0.0,
                image_size=640, model_weights="yolo26m.pt",
            ),
            HyperparameterConfig(
                name="y26m_heavy_aug",
                batch_size=64, lr0=0.01, lrf=0.01, cos_lr=True,
                weight_decay=0.001, dropout=0.2, mosaic=1.0, mixup=0.3,
                image_size=640, model_weights="yolo26m.pt",
            ),
            HyperparameterConfig(
                name="y26m_minimal_aug",
                batch_size=64, lr0=0.01, lrf=0.01, cos_lr=True,
                weight_decay=0.001, dropout=0.0, mosaic=0.3, mixup=0.0,
                image_size=640, model_weights="yolo26m.pt",
            ),
            HyperparameterConfig(
                name="y26m_high_dropout",
                batch_size=48, lr0=0.01, lrf=0.01, cos_lr=True,
                weight_decay=0.001, dropout=0.3, mosaic=0.8, mixup=0.1,
                image_size=640, model_weights="yolo26m.pt",
            ),
            HyperparameterConfig(
                name="y26m_high_res",
                batch_size=48, lr0=0.01, lrf=0.01, cos_lr=True,
                weight_decay=0.001, dropout=0.1, mosaic=0.8, mixup=0.0,
                image_size=800, model_weights="yolo26m.pt",
            ),
            HyperparameterConfig(
                name="y26m_high_res_aug",
                batch_size=48, lr0=0.01, lrf=0.01, cos_lr=True,
                weight_decay=0.001, dropout=0.1, mosaic=1.0, mixup=0.2,
                image_size=800, model_weights="yolo26m.pt",
            ),
            HyperparameterConfig(
                name="y26m_cosine_off",
                batch_size=64, lr0=0.01, lrf=0.01, cos_lr=False,
                weight_decay=0.001, dropout=0.1, mosaic=0.8, mixup=0.0,
                image_size=640, model_weights="yolo26m.pt",
            ),
            HyperparameterConfig(
                name="y26m_high_wd",
                batch_size=64, lr0=0.01, lrf=0.01, cos_lr=True,
                weight_decay=0.005, dropout=0.1, mosaic=0.8, mixup=0.0,
                image_size=640, model_weights="yolo26m.pt",
            ),
            HyperparameterConfig(
                name="y26m_large_batch",
                batch_size=80, lr0=0.01, lrf=0.01, cos_lr=True,
                weight_decay=0.001, dropout=0.1, mosaic=0.8, mixup=0.0,
                image_size=640, model_weights="yolo26m.pt",
            ),
            HyperparameterConfig(
                name="y26m_adam",
                batch_size=64, lr0=0.001, lrf=0.01, cos_lr=True,
                weight_decay=0.001, dropout=0.1, mosaic=0.8, mixup=0.0,
                image_size=640, model_weights="yolo26m.pt", optimizer="Adam",
            ),
            HyperparameterConfig(
                name="y26m_adamw",
                batch_size=64, lr0=0.001, lrf=0.01, cos_lr=True,
                weight_decay=0.01, dropout=0.1, mosaic=0.8, mixup=0.0,
                image_size=640, model_weights="yolo26m.pt", optimizer="AdamW",
            ),
            HyperparameterConfig(
                name="y26m_warm_restart",
                batch_size=64, lr0=0.015, lrf=0.05, cos_lr=True,
                weight_decay=0.0005, dropout=0.1, mosaic=0.8, mixup=0.1,
                image_size=640, model_weights="yolo26m.pt",
            ),
            # -- yolo26l configs -----------------------------------------------
            HyperparameterConfig(
                name="y26l_baseline",
                batch_size=32, lr0=0.01, lrf=0.01, cos_lr=True,
                weight_decay=0.001, dropout=0.1, mosaic=0.8, mixup=0.0,
                image_size=640, model_weights="yolo26l.pt",
            ),
            HyperparameterConfig(
                name="y26l_high_res",
                batch_size=24, lr0=0.01, lrf=0.01, cos_lr=True,
                weight_decay=0.001, dropout=0.1, mosaic=0.8, mixup=0.0,
                image_size=800, model_weights="yolo26l.pt",
            ),
            HyperparameterConfig(
                name="y26l_adam",
                batch_size=32, lr0=0.001, lrf=0.01, cos_lr=True,
                weight_decay=0.001, dropout=0.1, mosaic=0.8, mixup=0.0,
                image_size=640, model_weights="yolo26l.pt", optimizer="Adam",
            ),
        ]
        return yolo11m + yolo26m

    def train_single_config(
        self,
        hp_config: HyperparameterConfig,
        dataset_path: str,
        run_id: int,
    ) -> Optional[Dict[str, Any]]:
        """Train a single configuration and collect metrics.

        Args:
            hp_config: Hyperparameter configuration to evaluate.
            dataset_path: Path to the dataset root.
            run_id: Unique identifier for this run.

        Returns:
            A results dictionary, or ``None`` if the run could not execute.
        """
        if not self.gpu.fits_in_memory(
            hp_config.batch_size, hp_config.image_size, hp_config.model_weights
        ):
            logger.warning(
                "Config %d skipped: estimated memory exceeds limit", run_id
            )
            return None

        training_config = TrainingConfig(
            model_weights=hp_config.model_weights,
            batch_size=hp_config.batch_size,
            lr0=hp_config.lr0,
            lrf=hp_config.lrf,
            cos_lr=hp_config.cos_lr,
            weight_decay=hp_config.weight_decay,
            dropout=hp_config.dropout,
            mosaic=hp_config.mosaic,
            mixup=hp_config.mixup,
            image_size=hp_config.image_size,
            optimizer=hp_config.optimizer,
            epochs=70,
            patience=30,
        )

        builder = TrainArgsBuilder(training_config)
        run_name = (
            f"hp_search_{run_id}_{hp_config.name}_{Path(dataset_path).name}"
        )
        train_args = builder.build(dataset_path, run_name)

        logger.info(
            "Run %d: %s | Batch=%d LR=%s Img=%d",
            run_id,
            hp_config.name,
            hp_config.batch_size,
            hp_config.lr0,
            hp_config.image_size,
        )

        try:
            adapter = self.factory.create(hp_config.model_weights)
            start = time.time()
            adapter.train(**train_args)
            elapsed = time.time() - start

            val_results = adapter.validate(
                data=str((Path(dataset_path) / "dataset.yaml").resolve()), verbose=False
            )

            target_recall = target_precision = target_ap50 = 0.0
            overall_map50 = overall_map = 0.0

            if hasattr(val_results, "box") and val_results.box is not None:
                r = val_results.box.r
                p = val_results.box.p
                ap50 = val_results.box.ap50

                if len(r) > self.target_class:
                    target_recall = float(r[self.target_class])
                    target_precision = float(p[self.target_class])
                    target_ap50 = float(ap50[self.target_class])

                overall_map50 = float(val_results.box.map50)
                overall_map = float(val_results.box.map)

            result = {
                "run_id": run_id,
                "config_name": hp_config.name,
                "dataset": Path(dataset_path).name,
                "config": hp_config.model_dump(),
                "target_class_recall": target_recall,
                "target_class_precision": target_precision,
                "target_class_ap50": target_ap50,
                "overall_map50": overall_map50,
                "overall_map": overall_map,
                "training_time_minutes": elapsed / 60,
            }

            logger.info(
                "Run %d complete -- Recall: %.4f, Precision: %.4f, Time: %.1f min",
                run_id,
                target_recall,
                target_precision,
                elapsed / 60,
            )

            return result

        except Exception as e:
            logger.error("Run %d failed: %s", run_id, e)
            return {
                "run_id": run_id,
                "config_name": hp_config.name,
                "dataset": Path(dataset_path).name,
                "config": hp_config.model_dump(),
                "error": str(e),
                "target_class_recall": 0.0,
                "target_class_precision": 0.0,
                "target_class_ap50": 0.0,
                "overall_map50": 0.0,
                "overall_map": 0.0,
                "training_time_minutes": 0.0,
            }
        finally:
            self.gpu.clear_cache()

    def _save_partial(self) -> None:
        """Persist current results to disk."""
        df = pd.DataFrame(self.results)
        df.to_csv(self.output_dir / "partial_results.csv", index=False)
        with open(self.output_dir / "partial_results.json", "w") as f:
            json.dump(self.results, f, indent=2)

    def run_search(
        self,
        configurations: Optional[List[HyperparameterConfig]] = None,
    ) -> List[Dict[str, Any]]:
        """Execute the full hyperparameter search.

        Args:
            configurations: Configurations to evaluate.  Defaults to
                ``get_default_configurations()``.

        Returns:
            List of result dictionaries.
        """
        if configurations is None:
            configurations = self.get_default_configurations()

        total = len(configurations) * len(self.dataset_paths)
        logger.info("Starting search: %d experiments", total)

        run_id = 0
        for dataset_path in self.dataset_paths:
            logger.info("Dataset: %s", Path(dataset_path).name)
            for hp_config in configurations:
                run_id += 1
                result = self.train_single_config(
                    hp_config, dataset_path, run_id
                )
                if result:
                    self.results.append(result)
                    self._save_partial()
                time.sleep(2)

        self.analyzer.summarise_search_results(self.results, self.target_class)
        return self.results

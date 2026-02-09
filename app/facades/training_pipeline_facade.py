from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from app.config import settings
from app.preprocessing.dataset_preprocessor import DatasetPreprocessor
from app.schemas.finetuning_schema import FinetuningConfig
from app.schemas.preprocessing_schema import PreprocessingConfig
from app.schemas.training_schema import TrainingConfig
from app.training.conservative_trainer import ConservativeTrainer
from app.training.finetuner import Finetuner
from app.training.metrics_extractor import MetricsExtractor
from app.utils.logger import get_logger

logger = get_logger(__name__)


class TrainingPipelineFacade:
    """Facade orchestrating the full training pipeline from preprocessing
    through training and optional fine-tuning.

    Pipeline stages:
      1. Preprocess the dataset (noise filter + CLAHE).
      2. Train a conservative model.
      3. (Optional) Fine-tune the trained model.
      4. Evaluate the final model.

    Attributes:
        dataset_path: Path to the YOLO dataset root.
        preprocessed_path: Path for the preprocessed output.
    """

    def __init__(
        self,
        dataset_path: str,
        preprocessed_path: Optional[str] = None,
    ) -> None:
        """Initialise the TrainingPipelineFacade.

        Args:
            dataset_path: Root of the YOLO dataset.
            preprocessed_path: Destination for the preprocessed dataset.
                Defaults to ``settings.PREPROCESSED_ROOT``.
        """
        self.dataset_path = dataset_path
        self.preprocessed_path = preprocessed_path or str(
            settings.PREPROCESSED_ROOT
        )

    def run(
        self,
        skip_preprocessing: bool = False,
        skip_finetuning: bool = False,
    ) -> Dict[str, Any]:
        """Execute the training pipeline.

        Args:
            skip_preprocessing: If ``True``, use the dataset as-is.
            skip_finetuning: If ``True``, skip the fine-tuning stage.

        Returns:
            Dictionary containing paths and metrics from each stage.
        """
        results: Dict[str, Any] = {}

        # 1. Preprocess
        if skip_preprocessing:
            effective_dataset = self.dataset_path
            logger.info("Skipping preprocessing")
        else:
            logger.info("Stage 1: Preprocessing dataset")
            preprocessor = DatasetPreprocessor(PreprocessingConfig())
            effective_dataset = str(
                preprocessor.preprocess(
                    Path(self.dataset_path),
                    Path(self.preprocessed_path),
                )
            )
            results["preprocessed_path"] = effective_dataset

        # 2. Train
        logger.info("Stage 2: Conservative training")
        trainer = ConservativeTrainer(TrainingConfig())
        adapter, metrics = trainer.train(effective_dataset)
        results["training_metrics"] = metrics
        best_model = metrics.get("model_path", "")

        # 3. Fine-tune
        if not skip_finetuning and best_model:
            logger.info("Stage 3: Fine-tuning")
            yaml_path = str(Path(effective_dataset) / "dataset.yaml")
            ft_config = FinetuningConfig(
                model_path=best_model,
                dataset_yaml=yaml_path,
            )
            finetuner = Finetuner(ft_config)
            ft_adapter, ft_metrics = finetuner.finetune()
            results["finetuning_metrics"] = ft_metrics.model_dump()
        else:
            logger.info("Skipping fine-tuning")

        logger.info("Training pipeline complete")
        return results

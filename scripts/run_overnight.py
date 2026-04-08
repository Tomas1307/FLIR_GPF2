"""Run the full pipeline overnight: data -> validate -> train -> hyperparam search.

Designed to run unattended in a tmux session. All results are saved
incrementally so nothing is lost if the process is interrupted.

Usage::

    python -m scripts.run_overnight           # full run
    python -m scripts.run_overnight --dry-run  # verify setup only
"""

import argparse
import sys

import torch

from app.config import settings
from app.core.dataset_validator import DatasetValidator
from app.facades.data_pipeline_facade import DataPipelineFacade
from app.schemas.training_schema import TrainingConfig
from app.training.conservative_trainer import ConservativeTrainer
from app.training.gpu_manager import GpuManager
from app.training.hyperparameter_searcher import HyperparameterSearcher
from app.utils.logger import get_logger

logger = get_logger(__name__)


def dry_run() -> None:
    """Verify imports, dataset, GPU, and memory estimates."""
    logger.info("=== DRY RUN ===")

    logger.info("CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
        logger.info(
            "GPU memory: %.1f GB",
            torch.cuda.get_device_properties(0).total_memory / 1024**3,
        )

    dataset_yaml = settings.YOLO_ROOT / "dataset.yaml"
    if dataset_yaml.exists():
        logger.info("Dataset YAML found: %s", dataset_yaml)
    else:
        logger.warning("Dataset YAML not found: %s (will be created by data pipeline)", dataset_yaml)

    gpu = GpuManager()
    configs = HyperparameterSearcher.get_default_configurations()
    logger.info("Hyperparameter configs: %d", len(configs))
    for cfg in configs:
        fits = gpu.fits_in_memory(cfg.batch_size, cfg.image_size, cfg.model_weights)
        est = gpu.estimate_usage(cfg.batch_size, cfg.image_size, cfg.model_weights)
        status = "OK" if fits else "SKIP (OOM)"
        logger.info(
            "  %s: batch=%d img=%d est=%.1fGB — %s",
            cfg.name, cfg.batch_size, cfg.image_size, est, status,
        )

    logger.info("=== DRY RUN COMPLETE — ready for overnight ===")


def main() -> None:
    """Execute the overnight pipeline."""
    parser = argparse.ArgumentParser(description="Overnight training pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Verify setup only")
    args = parser.parse_args()

    if args.dry_run:
        dry_run()
        return

    logger.info("=== OVERNIGHT PIPELINE START ===")

    # Stage 1: Data pipeline
    logger.info("=== STAGE 1: DATA PREPARATION ===")
    try:
        DataPipelineFacade().run()
    except Exception:
        logger.exception("Data pipeline failed — aborting")
        sys.exit(1)

    # Stage 2: Validate
    logger.info("=== STAGE 2: DATASET VALIDATION ===")
    validator = DatasetValidator(settings.YOLO_ROOT)
    if not validator.validate_all():
        logger.error("Validation failed — aborting")
        sys.exit(1)

    # Stage 3: Conservative training
    logger.info("=== STAGE 3: CONSERVATIVE TRAINING ===")
    recall = 0.0
    try:
        trainer = ConservativeTrainer(TrainingConfig())
        _, metrics = trainer.train(str(settings.PREPROCESSED_ROOT))
        recall_data = metrics.get("recall_optimised", {})
        recall = recall_data.get("recall", 0.0) if isinstance(recall_data, dict) else 0.0
        logger.info("Conservative training recall: %.4f", recall)
    except Exception:
        logger.exception("Conservative training failed — continuing to search")

    # Stage 4: Recall gate
    if recall < settings.RECALL_GATE:
        logger.warning("Recall %.4f < %.2f — below target", recall, settings.RECALL_GATE)
    else:
        logger.info("Recall %.4f >= %.2f — above target", recall, settings.RECALL_GATE)

    # Stage 5: Hyperparameter search (always runs)
    logger.info("=== STAGE 5: HYPERPARAMETER SEARCH ===")
    try:
        searcher = HyperparameterSearcher(
            dataset_paths=[str(settings.PREPROCESSED_ROOT)],
            target_class=settings.TARGET_CLASS,
            base_epochs=settings.EPOCHS,
            gpu_memory_gb=settings.GPU_MEMORY_GB,
        )
        results = searcher.run_search()
        logger.info("Search complete: %d experiments", len(results))
    except Exception:
        logger.exception("Hyperparameter search failed")

    logger.info("=== OVERNIGHT PIPELINE COMPLETE ===")


if __name__ == "__main__":
    main()

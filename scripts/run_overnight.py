"""Run the full pipeline overnight: data -> validate -> train -> hyperparam search.

Designed to run unattended in a tmux session. All results are saved
incrementally so nothing is lost if the process is interrupted.

Resilience logic:
  - If a stage's output already exists it is skipped (unless ``--fresh``).
  - If validation fails the data pipeline is re-run once and validation
    retried before aborting.
  - Preprocessing (CLAHE + noise filter) is integrated into the data
    pipeline *before* augmentation so that augmented images inherit the
    enhancement.  A standalone preprocessing stage is kept as fallback
    for datasets built without the integrated step.
  - Training and search failures are logged but do not abort the pipeline.

Usage::

    python -m scripts.run_overnight                    # full run
    python -m scripts.run_overnight --dry-run          # verify setup only
    python -m scripts.run_overnight --fresh            # force re-run from scratch
    python -m scripts.run_overnight --resume           # auto-detect latest checkpoint
    python -m scripts.run_overnight --resume path/to/last.pt  # explicit checkpoint
"""

import argparse
import sys
from pathlib import Path

import torch

from app.config import settings
from app.core.dataset_validator import DatasetValidator
from app.facades.data_pipeline_facade import DataPipelineFacade
from app.preprocessing.dataset_preprocessor import DatasetPreprocessor
from app.schemas.preprocessing_schema import PreprocessingConfig
from app.schemas.training_schema import TrainingConfig
from app.training.conservative_trainer import ConservativeTrainer
from app.training.gpu_manager import GpuManager
from app.training.hyperparameter_searcher import HyperparameterSearcher
from app.utils.logger import get_logger

logger = get_logger(__name__)

MAX_VALIDATION_RETRIES = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_has_images(root: Path) -> bool:
    """Return True if *root*/train/images/ contains at least one file."""
    train_images = root / "train" / "images"
    return train_images.exists() and any(train_images.iterdir())


def _find_latest_checkpoint() -> Path | None:
    """Return the most recent last.pt checkpoint from conservative training runs."""
    candidates = sorted(
        Path(".").glob("conservative_final_*/*/weights/last.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------

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

    for label, root in [("YOLO_ROOT", settings.YOLO_ROOT), ("PREPROCESSED_ROOT", settings.PREPROCESSED_ROOT)]:
        if _split_has_images(root):
            logger.info("%s: exists (%s)", label, root)
        else:
            logger.info("%s: empty or missing (%s)", label, root)

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


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def stage_data_pipeline(fresh: bool) -> bool:
    """Stage 1: Build YOLO dataset from raw data.

    The data pipeline now includes preprocessing (noise filter + CLAHE)
    before augmentation, so augmented images inherit the enhancement.

    Returns True on success.
    """
    if not fresh and _split_has_images(settings.YOLO_ROOT):
        logger.info("=== STAGE 1: SKIPPED (dataset exists at %s) ===", settings.YOLO_ROOT)
        return True

    logger.info("=== STAGE 1: DATA PREPARATION (includes preprocessing) ===")
    try:
        DataPipelineFacade().run()
        return True
    except Exception:
        logger.exception("Data pipeline failed")
        return False


def stage_validate() -> bool:
    """Stage 2: Validate the YOLO dataset.

    Returns True if all checks pass.
    """
    logger.info("=== STAGE 2: DATASET VALIDATION ===")
    validator = DatasetValidator(settings.YOLO_ROOT)
    return validator.validate_all()


def stage_preprocess_fallback() -> Path:
    """Stage 2.5 (fallback): Standalone preprocessing for legacy datasets.

    If the data pipeline was skipped (dataset already existed) and it was
    built *before* preprocessing was integrated into the data pipeline,
    this stage applies CLAHE + noise filter to a copy in PREPROCESSED_ROOT.

    Returns the effective dataset path to use for training.
    """
    if _split_has_images(settings.PREPROCESSED_ROOT):
        logger.info("=== STAGE 2.5: SKIPPED (preprocessed dataset exists at %s) ===", settings.PREPROCESSED_ROOT)
        return settings.PREPROCESSED_ROOT

    logger.info("=== STAGE 2.5: STANDALONE PREPROCESSING (fallback) ===")
    try:
        preprocessor = DatasetPreprocessor(PreprocessingConfig())
        preprocessor.preprocess(settings.YOLO_ROOT, settings.PREPROCESSED_ROOT)

        # Update dataset.yaml path to point to preprocessed root
        yaml_path = settings.PREPROCESSED_ROOT / "dataset.yaml"
        if yaml_path.exists():
            text = yaml_path.read_text()
            new_lines = []
            for line in text.splitlines():
                if line.strip().startswith("path:"):
                    new_lines.append(f"path: {settings.PREPROCESSED_ROOT.resolve()}")
                else:
                    new_lines.append(line)
            yaml_path.write_text("\n".join(new_lines) + "\n")
            logger.info("Updated dataset.yaml path → %s", settings.PREPROCESSED_ROOT.resolve())

        return settings.PREPROCESSED_ROOT
    except Exception:
        logger.exception("Standalone preprocessing failed — using YOLO_ROOT as-is")
        return settings.YOLO_ROOT


def stage_training(dataset_root: Path, resume_from: Path | None = None) -> float:
    """Stage 3: Conservative training.

    Args:
        dataset_root: Path to the YOLO dataset.
        resume_from: Optional checkpoint path to resume from.

    Returns the mining-class recall (0.0 on failure).
    """
    logger.info("=== STAGE 3: CONSERVATIVE TRAINING ===")
    logger.info("Training dataset: %s", dataset_root)
    if resume_from:
        logger.info("Resuming from checkpoint: %s", resume_from)
    try:
        trainer = ConservativeTrainer(TrainingConfig())
        _, metrics = trainer.train(
            str(dataset_root),
            resume_from=str(resume_from) if resume_from else None,
        )
        recall_data = metrics.get("recall_optimised", {})
        recall = recall_data.get("recall", 0.0) if isinstance(recall_data, dict) else 0.0
        logger.info("Conservative training recall: %.4f", recall)
        return recall
    except Exception:
        logger.exception("Conservative training failed — continuing to search")
        return 0.0


def stage_hyperparameter_search(dataset_root: Path) -> None:
    """Stage 5: Hyperparameter grid search."""
    logger.info("=== STAGE 5: HYPERPARAMETER SEARCH ===")
    logger.info("Search dataset: %s", dataset_root)
    try:
        searcher = HyperparameterSearcher(
            dataset_paths=[str(dataset_root)],
            target_class=settings.TARGET_CLASS,
            base_epochs=settings.EPOCHS,
            gpu_memory_gb=settings.GPU_MEMORY_GB,
        )
        results = searcher.run_search()
        logger.info("Search complete: %d experiments", len(results))
    except Exception:
        logger.exception("Hyperparameter search failed")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main() -> None:
    """Execute the overnight pipeline with resilience logic."""
    parser = argparse.ArgumentParser(description="Overnight training pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Verify setup only")
    parser.add_argument("--fresh", action="store_true", help="Force re-run all stages from scratch")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        metavar="CHECKPOINT",
        help="Resume conservative training. Omit path to auto-detect latest checkpoint.",
    )
    args = parser.parse_args()

    if args.dry_run:
        dry_run()
        return

    logger.info("=== OVERNIGHT PIPELINE START ===")

    # -- Stage 1: Data pipeline (includes preprocessing before augmentation) ---
    if not stage_data_pipeline(args.fresh):
        logger.error("Data pipeline failed — aborting")
        sys.exit(1)

    # -- Stage 2: Validation (with retry) --------------------------------------
    for attempt in range(1 + MAX_VALIDATION_RETRIES):
        if stage_validate():
            break

        if attempt < MAX_VALIDATION_RETRIES:
            logger.warning(
                "Validation failed — re-running data pipeline (attempt %d/%d)",
                attempt + 1, MAX_VALIDATION_RETRIES,
            )
            if not stage_data_pipeline(fresh=True):
                logger.error("Data pipeline retry failed — aborting")
                sys.exit(1)
        else:
            logger.error("Validation failed after %d retries — aborting", MAX_VALIDATION_RETRIES)
            sys.exit(1)

    # -- Determine effective dataset -------------------------------------------
    # If the data pipeline ran fresh, preprocessing is already integrated
    # (Stage 5/9 inside DataPipelineFacade).  If the dataset was skipped
    # (pre-existing), fall back to standalone preprocessing.
    effective_dataset = settings.YOLO_ROOT
    if not args.fresh and _split_has_images(settings.YOLO_ROOT):
        # Dataset was pre-existing — may lack integrated preprocessing.
        # Run standalone preprocessing as fallback.
        effective_dataset = stage_preprocess_fallback()

    logger.info("Effective training dataset: %s", effective_dataset)

    # -- Resolve resume checkpoint ---------------------------------------------
    resume_ckpt: Path | None = None
    if args.resume:
        if args.resume == "auto":
            resume_ckpt = _find_latest_checkpoint()
            if resume_ckpt:
                logger.info("Auto-detected checkpoint: %s", resume_ckpt)
            else:
                logger.warning("No checkpoint found for --resume auto — starting fresh training")
        else:
            resume_ckpt = Path(args.resume)
            if not resume_ckpt.exists():
                logger.error("Checkpoint not found: %s — aborting", resume_ckpt)
                sys.exit(1)

    # -- Stage 3: Conservative training ----------------------------------------
    recall = stage_training(effective_dataset, resume_from=resume_ckpt)

    # -- Stage 4: Recall gate --------------------------------------------------
    if recall < settings.RECALL_GATE:
        logger.warning("Recall %.4f < %.2f — below target", recall, settings.RECALL_GATE)
    else:
        logger.info("Recall %.4f >= %.2f — above target", recall, settings.RECALL_GATE)

    # -- Stage 5: Hyperparameter search (always runs) --------------------------
    stage_hyperparameter_search(effective_dataset)

    logger.info("=== OVERNIGHT PIPELINE COMPLETE ===")


if __name__ == "__main__":
    main()

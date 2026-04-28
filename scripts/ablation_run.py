"""Run ablation study on a given dataset.

Usage::

    CUDA_VISIBLE_DEVICES=2 python -m scripts.ablation_run --config y11m_heavy_aug
    CUDA_VISIBLE_DEVICES=2 python -m scripts.ablation_run --config y11m_heavy_aug --dataset /path/to/dataset
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

from app.training.hyperparameter_searcher import HyperparameterSearcher
from app.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_DATASET = "/home/tacosta/FLIR_GPF2/Dataset_Ablation_NoAug"
EPOCHS = 150
PATIENCE = 20


def main():
    parser = argparse.ArgumentParser(description="Ablation study")
    parser.add_argument("--config", type=str, required=True, help="Config name from HP search (e.g. y11m_heavy_aug)")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help="Path to dataset root (must contain dataset.yaml)")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    args = parser.parse_args()

    dataset_yaml = str((Path(args.dataset) / "dataset.yaml").resolve())

    configs = {c.name: c for c in HyperparameterSearcher.get_default_configurations()}
    if args.config not in configs:
        logger.error("Config '%s' not found. Available: %s", args.config, list(configs.keys()))
        return

    hp = configs[args.config]
    dataset_name = Path(args.dataset).name
    run_name = f"ablation_{dataset_name}_{hp.name}"

    logger.info("=== ABLATION STUDY ===")
    logger.info("Config: %s", hp.name)
    logger.info("Model: %s | Batch: %d | ImgSz: %d", hp.model_weights, hp.batch_size, hp.image_size)
    logger.info("Epochs: %d | Patience: %d", args.epochs, args.patience)
    logger.info("Dataset: %s", dataset_yaml)

    model = YOLO(hp.model_weights)
    model.train(
        data=dataset_yaml,
        epochs=args.epochs,
        patience=args.patience,
        batch=hp.batch_size,
        imgsz=hp.image_size,
        lr0=hp.lr0,
        lrf=hp.lrf,
        cos_lr=hp.cos_lr,
        weight_decay=hp.weight_decay,
        dropout=hp.dropout,
        mosaic=hp.mosaic,
        mixup=hp.mixup,
        optimizer=hp.optimizer,
        name=run_name,
        project="runs",
        exist_ok=True,
        verbose=True,
    )

    logger.info("=== ABLATION COMPLETE: %s ===", run_name)


if __name__ == "__main__":
    main()

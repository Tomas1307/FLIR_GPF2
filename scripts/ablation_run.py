"""Run ablation study on Dataset_Ablation_NoAug.

Usage::

    CUDA_VISIBLE_DEVICES=2 python -m scripts.ablation_run --config y11m_warm_restart
    CUDA_VISIBLE_DEVICES=2 python -m scripts.ablation_run --config y11m_baseline
"""

import argparse

from ultralytics import YOLO

from app.training.hyperparameter_searcher import HyperparameterSearcher
from app.utils.logger import get_logger

logger = get_logger(__name__)

DATASET = "/home/tacosta/FLIR_GPF2/Dataset_Ablation_NoAug/dataset.yaml"
EPOCHS = 150
PATIENCE = 20


def main():
    parser = argparse.ArgumentParser(description="Ablation study (no augmentation)")
    parser.add_argument("--config", type=str, required=True, help="Config name from HP search (e.g. y11m_warm_restart)")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    args = parser.parse_args()

    configs = {c.name: c for c in HyperparameterSearcher.get_default_configurations()}
    if args.config not in configs:
        logger.error("Config '%s' not found. Available: %s", args.config, list(configs.keys()))
        return

    hp = configs[args.config]
    run_name = f"ablation_noaug_{hp.name}"

    logger.info("=== ABLATION STUDY ===")
    logger.info("Config: %s", hp.name)
    logger.info("Model: %s | Batch: %d | ImgSz: %d", hp.model_weights, hp.batch_size, hp.image_size)
    logger.info("Epochs: %d | Patience: %d", args.epochs, args.patience)
    logger.info("Dataset: %s", DATASET)

    model = YOLO(hp.model_weights)
    model.train(
        data=DATASET,
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

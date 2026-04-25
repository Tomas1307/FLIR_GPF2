"""Extract comprehensive HP search results for documentation.

Re-validates each best.pt to get per-class metrics (especially class 4)
and combines with overall metrics, best epoch, and HP configuration.

Usage::

    CUDA_VISIBLE_DEVICES=2 python -m scripts.extract_hp_results
"""

import csv
import json
from pathlib import Path

import pandas as pd

from app.config import settings
from app.training.hyperparameter_searcher import HyperparameterSearcher
from app.training.model_factory import ModelFactory
from app.utils.logger import get_logger

logger = get_logger(__name__)

DATASET = "Dataset_Balanceado_Preproc"
RUNS_DIR = Path("runs")
TARGET_CLASS = settings.TARGET_CLASS  # 4
NUM_CLASSES = settings.NUM_CLASSES    # 5


def _get_config_map():
    """Map (run_id, config_name) -> HyperparameterConfig."""
    configs = HyperparameterSearcher.get_default_configurations()
    return {(i + 1, c.name): c for i, c in enumerate(configs)}


def _best_epoch_from_csv(run_dir: Path):
    """Find best epoch and its overall metrics from results.csv.

    YOLO fitness = 0.1 * mAP50 + 0.9 * mAP50-95.
    """
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return None

    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Strip whitespace from keys
            row = {k.strip(): v.strip() for k, v in row.items()}
            rows.append(row)

    if not rows:
        return None

    best = None
    best_fitness = -1
    for row in rows:
        try:
            map50 = float(row.get("metrics/mAP50(B)", 0))
            map50_95 = float(row.get("metrics/mAP50-95(B)", 0))
            fitness = 0.1 * map50 + 0.9 * map50_95
            if fitness > best_fitness:
                best_fitness = fitness
                best = row
        except (ValueError, TypeError):
            continue

    if best is None:
        return None

    return {
        "best_epoch": int(best.get("epoch", 0)) + 1,
        "total_epochs": len(rows),
        "overall_precision": float(best.get("metrics/precision(B)", 0)),
        "overall_recall": float(best.get("metrics/recall(B)", 0)),
        "overall_map50": float(best.get("metrics/mAP50(B)", 0)),
        "overall_map50_95": float(best.get("metrics/mAP50-95(B)", 0)),
        "fitness": best_fitness,
    }


def _validate_best_pt(run_dir: Path, dataset_path: str, model_weights: str):
    """Validate best.pt and extract per-class metrics."""
    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        return None

    factory = ModelFactory()
    adapter = factory.create(model_weights)
    adapter.model = adapter.model.__class__(str(best_pt))

    val_results = adapter.validate(
        data=str((Path(dataset_path) / "dataset.yaml").resolve()),
        verbose=False,
    )

    per_class = {}
    if hasattr(val_results, "box") and val_results.box is not None:
        r = val_results.box.r
        p = val_results.box.p
        ap50 = val_results.box.ap50
        ap = val_results.box.ap

        for cls_id in range(min(NUM_CLASSES, len(r))):
            per_class[cls_id] = {
                "recall": float(r[cls_id]),
                "precision": float(p[cls_id]),
                "ap50": float(ap50[cls_id]),
            }

    return per_class


def main():
    dataset_path = str(Path(DATASET).resolve())
    config_map = _get_config_map()
    factory = ModelFactory()

    all_results = []

    run_dirs = sorted(
        RUNS_DIR.glob(f"hp_search_*_{DATASET}"),
        key=lambda d: int(d.name.split("_")[2]),
    )

    logger.info("Found %d runs to analyse", len(run_dirs))

    for run_dir in run_dirs:
        name = run_dir.name.replace(f"_{DATASET}", "")
        parts = name.split("_", 3)  # hp_search_{id}_{config_name}
        run_id = int(parts[2])
        config_name = "_".join(parts[3:])

        logger.info("Processing %d/%d: %s", run_id, len(run_dirs), config_name)

        # Get HP config
        hp_config = config_map.get((run_id, config_name))
        if hp_config is None:
            logger.warning("No config found for run %d: %s", run_id, config_name)
            continue

        # Get best epoch + overall metrics from results.csv
        epoch_info = _best_epoch_from_csv(run_dir)
        if epoch_info is None:
            logger.warning("No results.csv for run %d", run_id)
            continue

        # Validate best.pt for per-class metrics
        per_class = _validate_best_pt(run_dir, dataset_path, hp_config.model_weights)

        cls4 = per_class.get(TARGET_CLASS, {}) if per_class else {}

        row = {
            "run_id": run_id,
            "config_name": config_name,
            "model": hp_config.model_weights,
            "batch_size": hp_config.batch_size,
            "lr0": hp_config.lr0,
            "lrf": hp_config.lrf,
            "optimizer": hp_config.optimizer,
            "cos_lr": hp_config.cos_lr,
            "weight_decay": hp_config.weight_decay,
            "dropout": hp_config.dropout,
            "mosaic": hp_config.mosaic,
            "mixup": hp_config.mixup,
            "image_size": hp_config.image_size,
            "best_epoch": epoch_info["best_epoch"],
            "total_epochs": epoch_info["total_epochs"],
            "overall_recall": epoch_info["overall_recall"],
            "overall_precision": epoch_info["overall_precision"],
            "overall_map50": epoch_info["overall_map50"],
            "overall_map50_95": epoch_info["overall_map50_95"],
            "cls4_recall": cls4.get("recall", 0.0),
            "cls4_precision": cls4.get("precision", 0.0),
            "cls4_ap50": cls4.get("ap50", 0.0),
        }
        all_results.append(row)

        # Clear GPU between validations
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save full CSV
    df = pd.DataFrame(all_results)
    df.to_csv("hp_search_full_results.csv", index=False)
    logger.info("Saved full results to hp_search_full_results.csv")

    # Print summary table sorted by class 4 recall
    df_sorted = df.sort_values("cls4_recall", ascending=False)

    # Print full table sorted by class 4 recall
    df_sorted = df.sort_values("cls4_recall", ascending=False)

    cols = [
        ("#",       3),
        ("Config",  25),
        ("Model",   12),
        ("Batch",   5),
        ("LR0",     7),
        ("LRF",     7),
        ("Optim",   6),
        ("CosLR",   5),
        ("WD",      7),
        ("Drop",    5),
        ("Mosaic",  6),
        ("Mixup",   5),
        ("ImgSz",   5),
        ("BstEp",   5),
        ("TotEp",   5),
        ("OvR",     7),
        ("OvP",     7),
        ("OvmAP50", 8),
        ("OvmAP95", 8),
        ("C4R",     7),
        ("C4P",     7),
        ("C4AP50",  7),
    ]
    header = " ".join(f"{name:>{w}}" for name, w in cols)
    sep = "-" * len(header)

    print(f"\n{'=' * len(header)}")
    print("HYPERPARAMETER SEARCH RESULTS — sorted by Illegal Mining (class 4) recall")
    print(f"{'=' * len(header)}")
    print(header)
    print(sep)
    for i, (_, r) in enumerate(df_sorted.iterrows(), 1):
        print(
            f"{i:>3} {r['config_name']:<25} {r['model']:<12} {r['batch_size']:>5} "
            f"{r['lr0']:>7.4f} {r['lrf']:>7.4f} {r['optimizer']:>6} "
            f"{'Y' if r['cos_lr'] else 'N':>5} {r['weight_decay']:>7.4f} "
            f"{r['dropout']:>5.2f} {r['mosaic']:>6.2f} {r['mixup']:>5.2f} "
            f"{r['image_size']:>5} {r['best_epoch']:>5} {r['total_epochs']:>5} "
            f"{r['overall_recall']:>7.4f} {r['overall_precision']:>7.4f} "
            f"{r['overall_map50']:>8.4f} {r['overall_map50_95']:>8.4f} "
            f"{r['cls4_recall']:>7.4f} {r['cls4_precision']:>7.4f} {r['cls4_ap50']:>7.4f}"
        )

    # Top 5
    print(f"\n{'=' * 90}")
    print("TOP 5 by Class 4 (Illegal Mining) Recall")
    print(f"{'=' * 90}")
    for i, (_, r) in enumerate(df_sorted.head(5).iterrows(), 1):
        print(
            f"  #{i} {r['config_name']} ({r['model']}) — "
            f"C4 Recall: {r['cls4_recall']:.4f}, C4 Precision: {r['cls4_precision']:.4f}, "
            f"C4 AP50: {r['cls4_ap50']:.4f} | Best epoch: {r['best_epoch']}/{r['total_epochs']}"
        )

    # Save JSON for documentation
    with open("hp_search_full_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Saved JSON to hp_search_full_results.json")


if __name__ == "__main__":
    main()

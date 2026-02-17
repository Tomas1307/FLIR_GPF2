# Architecture - Illegal Mining Detection Pipeline

## Overview

YOLOv11-based object detection pipeline for identifying illegal mining operations in FLIR/satellite imagery. The system achieved **93% recall** on the target class (illegal mining). The `app/` package follows enterprise-grade standards: GoF design patterns, Pydantic schemas, one-class-per-file, Singleton configuration.

**5 object classes** + background:

| ID | Name | Role |
|----|------|------|
| -1 | Background | Images with no annotations |
| 0 | Vehicles | Supporting indicator |
| 1 | Warehouses | Supporting indicator |
| 2 | Roads | Supporting indicator |
| 3 | Rivers | Supporting indicator |
| **4** | **Illegal Mining** | **Primary detection target** |

---

## Directory Layout

```
app/
|-- config.py                          # Singleton settings (Pydantic BaseSettings)
|
|-- schemas/                           # Pydantic data models (no business logic)
|   |-- class_schema.py                #   ClassDefinition
|   |-- dataset_schema.py              #   DatasetConfig
|   |-- augmentation_schema.py         #   AugmentationTarget
|   |-- preprocessing_schema.py        #   PreprocessingConfig
|   |-- training_schema.py             #   TrainingConfig
|   |-- hyperparameter_schema.py       #   HyperparameterConfig
|   |-- finetuning_schema.py           #   FinetuningConfig
|   |-- metrics_schema.py              #   MiningMetrics, MetricsResult
|
|-- utils/                             # Pure functions (no classes here)
|   |-- logger.py                      #   get_logger()
|   |-- file_utils.py                  #   copy_file(), ensure_dir(), glob_images()
|   |-- image_utils.py                 #   load_image(), save_image(), bgr_to_rgb()
|   |-- label_utils.py                 #   parse_yolo_label(), write_yolo_label()
|   |-- hash_utils.py                  #   compute_image_hash()
|   |-- dataset_stats.py               #   count_classes(), compute_distribution(), images_by_class()
|
|-- core/                              # Dataset manipulation (one class per file)
|   |-- dataset_unifier.py             #   DatasetUnifier
|   |-- strategic_splitter.py          #   StrategicSplitter
|   |-- label_cleaner.py               #   LabelCleaner
|   |-- duplicate_remover.py           #   DuplicateRemover
|   |-- class_balancer.py              #   ClassBalancer
|
|-- preprocessing/                     # Image-level transforms
|   |-- noise_filter.py                #   NoiseFilter
|   |-- contrast_enhancer.py           #   ContrastEnhancer
|   |-- preprocessing_pipeline.py      #   PreprocessingPipeline  [Facade]
|   |-- dataset_preprocessor.py        #   DatasetPreprocessor
|
|-- augmentation/                      # Data augmentation
|   |-- mining_augmentation.py         #   MiningAugmentationPipeline
|   |-- normal_augmentation.py         #   NormalAugmentationPipeline
|   |-- background_augmentation.py     #   BackgroundAugmentation
|   |-- augmentation_factory.py        #   AugmentationFactory    [Factory]
|   |-- class_augmentor.py             #   ClassAugmentor
|
|-- training/                          # Model training and evaluation
|   |-- gpu_manager.py                 #   GpuManager             [Singleton]
|   |-- model_factory.py               #   ModelFactory            [Factory]
|   |-- train_args_builder.py          #   TrainArgsBuilder
|   |-- conservative_trainer.py        #   ConservativeTrainer
|   |-- hyperparameter_searcher.py     #   HyperparameterSearcher
|   |-- finetuner.py                   #   Finetuner
|   |-- metrics_extractor.py           #   MetricsExtractor
|   |-- results_analyzer.py            #   ResultsAnalyzer
|
|-- adapters/                          # External API wrappers
|   |-- yolo_adapter.py                #   YoloAdapter            [Adapter]
|   |-- dataset_adapter.py             #   DatasetAdapter          [Adapter]
|
|-- facades/                           # Pipeline orchestrators
|   |-- data_pipeline_facade.py        #   DataPipelineFacade     [Facade]
|   |-- training_pipeline_facade.py    #   TrainingPipelineFacade [Facade]
|
|-- visualization/                     # Plotting and display
|   |-- bbox_visualizer.py             #   BBoxVisualizer
|   |-- class_distribution_chart.py    #   ClassDistributionChart
|   |-- prediction_grid.py             #   PredictionGrid
|   |-- preprocessing_comparison.py    #   PreprocessingComparison
|   |-- histogram_plotter.py           #   HistogramPlotter

scripts/
|-- run_data_pipeline.py               # Stage 1: unify -> split -> augment -> clean
|-- run_preprocessing.py               # Stage 2: CLAHE + median filter
|-- run_training.py                    # Stage 3: conservative training
|-- run_hyperparameter_search.py       # Stage 4: HP grid search
|-- run_finetuning.py                  # Stage 5: fine-tune with frozen backbone
|-- run_evaluation.py                  # Standalone model evaluation
|-- run_full_pipeline.py               # End-to-end: data + training
```

---

## Design Patterns

| Pattern | Class | File | Purpose |
|---------|-------|------|---------|
| **Singleton** | `Settings` | `app/config.py` | One global config accessed as `settings.VARIABLE_NAME` |
| **Singleton** | `GpuManager` | `app/training/gpu_manager.py` | Single point of GPU memory management via `__new__` |
| **Factory** | `AugmentationFactory` | `app/augmentation/augmentation_factory.py` | Returns `BackgroundAugmentation` / `MiningAugmentationPipeline` / `NormalAugmentationPipeline` by class id |
| **Factory** | `ModelFactory` | `app/training/model_factory.py` | Creates `YoloAdapter` instances with automatic GPU cleanup |
| **Facade** | `DataPipelineFacade` | `app/facades/data_pipeline_facade.py` | Orchestrates 8-stage data preparation behind `run()` |
| **Facade** | `TrainingPipelineFacade` | `app/facades/training_pipeline_facade.py` | Orchestrates preprocess + train + finetune behind `run()` |
| **Facade** | `PreprocessingPipeline` | `app/preprocessing/preprocessing_pipeline.py` | Chains `NoiseFilter` + `ContrastEnhancer` behind `process()` |
| **Adapter** | `YoloAdapter` | `app/adapters/yolo_adapter.py` | Wraps `ultralytics.YOLO` so the codebase is decoupled from Ultralytics internals |
| **Adapter** | `DatasetAdapter` | `app/adapters/dataset_adapter.py` | Resolves `val`/`validation`, `Imagenes`/`images`, `Etiquetas`/`labels` path inconsistencies |

---

## Dependency Graph

Layers are strictly ordered -- lower layers never import from higher ones.

```
                        scripts/
                           |
                      facades/ (orchestration)
                     /         \
             core/              training/
               |                /    |    \
         preprocessing/   adapters/  |   augmentation/
               |              |      |        |
            schemas/       schemas/  |     schemas/
               \              |      |       /
                -------- utils/ + config.py --------
```

### Per-module dependencies (imports from `app/`)

| Module | Depends on |
|--------|-----------|
| `config` | (none -- leaf) |
| `schemas/*` | (none -- leaf) |
| `utils/logger` | (none -- leaf) |
| `utils/file_utils` | `utils.logger` |
| `utils/image_utils` | (none -- only cv2/numpy) |
| `utils/label_utils` | (none -- only pathlib) |
| `utils/hash_utils` | (none -- only cv2/hashlib) |
| `utils/dataset_stats` | `utils.label_utils`, `utils.file_utils`, `utils.logger` |
| `core/dataset_unifier` | `config`, `utils.file_utils`, `utils.logger` |
| `core/strategic_splitter` | `config`, `utils.file_utils`, `utils.label_utils`, `utils.logger` |
| `core/label_cleaner` | `config`, `utils.logger` |
| `core/duplicate_remover` | `config`, `utils.file_utils`, `utils.hash_utils`, `utils.logger` |
| `core/class_balancer` | `config`, `utils.file_utils`, `utils.label_utils`, `utils.logger` |
| `preprocessing/noise_filter` | `config` |
| `preprocessing/contrast_enhancer` | `config` |
| `preprocessing/preprocessing_pipeline` | `preprocessing.noise_filter`, `preprocessing.contrast_enhancer`, `schemas.preprocessing_schema` |
| `preprocessing/dataset_preprocessor` | `preprocessing.preprocessing_pipeline`, `schemas.preprocessing_schema`, `utils.file_utils`, `utils.image_utils`, `utils.logger` |
| `augmentation/mining_augmentation` | (none -- only albumentations) |
| `augmentation/normal_augmentation` | (none -- only albumentations) |
| `augmentation/background_augmentation` | (none -- only cv2/numpy) |
| `augmentation/augmentation_factory` | `augmentation.*`, `config` |
| `augmentation/class_augmentor` | `augmentation.augmentation_factory`, `config`, `utils.*` |
| `adapters/yolo_adapter` | `utils.logger` |
| `adapters/dataset_adapter` | `config`, `utils.logger` |
| `training/gpu_manager` | `config`, `utils.logger` |
| `training/model_factory` | `adapters.yolo_adapter`, `training.gpu_manager`, `utils.logger` |
| `training/train_args_builder` | `config`, `schemas.training_schema` |
| `training/metrics_extractor` | `adapters.yolo_adapter`, `config`, `schemas.metrics_schema`, `utils.logger` |
| `training/results_analyzer` | `schemas.metrics_schema`, `utils.logger` |
| `training/conservative_trainer` | `config`, `schemas.training_schema`, `training.*`, `utils.logger` |
| `training/hyperparameter_searcher` | `config`, `schemas.*`, `training.*`, `utils.logger` |
| `training/finetuner` | `config`, `schemas.finetuning_schema`, `training.*`, `adapters.yolo_adapter`, `utils.logger` |
| `facades/data_pipeline_facade` | `config`, `core.*`, `augmentation.class_augmentor`, `utils.*` |
| `facades/training_pipeline_facade` | `config`, `preprocessing.*`, `training.*`, `schemas.*`, `utils.logger` |
| `visualization/*` | `config`, `utils.*`, `adapters.yolo_adapter`, `preprocessing.*` (varies) |

---

## Data Flow

### Full Pipeline (`scripts/run_full_pipeline.py`)

```
RAW DATA (data/)
  |
  v
[Stage 1] DataPipelineFacade.run()
  |-- DatasetUnifier.unify()              data/ -> data_unified/
  |-- dataset_stats.images_by_class()     analyse unified set
  |-- StrategicSplitter.split()           guarantee class 4 in all splits
  |-- StrategicSplitter.copy_to_yolo()    data_unified/ -> yolo_dataset/{train,val,test}/
  |-- ClassAugmentor.augment_all()        balance classes to target counts
  |-- LabelCleaner.clean()               fix "4.0" -> "4" in labels
  |-- DuplicateRemover.clean()           remove perceptual-hash duplicates
  |-- ClassBalancer.remove_excess(1)      remove 80% augmented warehouses
  |
  v
YOLO DATASET (yolo_dataset/)
  |
  v
[Stage 2] TrainingPipelineFacade.run()
  |-- DatasetPreprocessor.preprocess()
  |     |-- NoiseFilter.apply()           median blur if salt-and-pepper detected
  |     |-- ContrastEnhancer.apply()      CLAHE on L channel of LAB
  |     -> preprocessed/
  |
  |-- ConservativeTrainer.train()
  |     |-- ModelFactory.create()         GpuManager.clear_cache() + YOLO(weights)
  |     |-- TrainArgsBuilder.build()      ~50-key dict from TrainingConfig
  |     |-- YoloAdapter.train()           ultralytics training
  |     |-- MetricsExtractor              evaluate at conf=0.15 and conf=0.25
  |     |-- ResultsAnalyzer               generate recommendations
  |     -> conservative_final_*/
  |
  |-- Finetuner.finetune()               (optional)
        |-- freeze 10 backbone layers
        |-- Adam optimizer, lr=0.0005
        |-- 15 epochs
        -> finetuning_results/
```

### Augmentation Decision Tree (`AugmentationFactory.create()`)

```
class_id == -1  -->  BackgroundAugmentation   (OpenCV pixel transforms, no bboxes)
class_id == 4   -->  MiningAugmentationPipeline  (Albumentations with bbox_params)
class_id in {0,1,2,3} --> NormalAugmentationPipeline (Albumentations with bbox_params)
```

---

## Configuration (`app/config.py`)

All tunable values live in `Settings` and are accessed as `settings.VARIABLE_NAME`. Override via environment variables prefixed with `FLIR_` or a `.env` file.

| Category | Key settings |
|----------|-------------|
| **Paths** | `DATA_ROOT`, `UNIFIED_ROOT`, `YOLO_ROOT`, `PREPROCESSED_ROOT` |
| **Classes** | `CLASS_NAMES`, `TARGET_CLASS=4`, `NUM_CLASSES=5` |
| **Augmentation** | `AUG_TARGET_MINING=3500`, `AUG_TARGET_VEHICLES=2000`, `AUG_TARGET_WAREHOUSES=3000`, `AUG_TARGET_ROADS=3000`, `AUG_TARGET_RIVERS=3000`, `AUG_TARGET_BACKGROUND=2500` |
| **Splits** | `TEST_SIZE=0.15`, `VAL_SIZE=0.15` |
| **Preprocessing** | `NOISE_THRESHOLD=0.01`, `MEDIAN_KERNEL=3`, `CLAHE_CLIP=2.0`, `CLAHE_TILE_GRID=(8,8)` |
| **Training** | `MODEL_WEIGHTS=yolo11m.pt`, `BATCH_SIZE=40`, `LR0=0.005`, `EPOCHS=100`, `PATIENCE=3`, `DROPOUT=0.1`, `MOSAIC=0.8` |
| **Fine-tuning** | `FINETUNE_LR=0.0005`, `FINETUNE_EPOCHS=15`, `FREEZE_LAYERS=10` |
| **Inference** | `CONF_THRESHOLD_RECALL=0.15`, `CONF_THRESHOLD_NORMAL=0.25`, `IOU_THRESHOLD=0.6` |
| **Hardware** | `GPU_MEMORY_GB=24`, `RANDOM_SEED=42`, `WORKERS=16` |

---

## Schemas (`app/schemas/`)

All data structures use Pydantic `BaseModel` -- no `@dataclass` anywhere.

| Schema | Fields | Used by |
|--------|--------|---------|
| `ClassDefinition` | `class_id`, `name`, `is_background` | visualization |
| `DatasetConfig` | `root`, `image_dirs`, `label_dirs`, `yaml_path`, `splits` | adapters |
| `AugmentationTarget` | `class_id`, `current_count`, `target_count`, `has_bboxes` | augmentation |
| `PreprocessingConfig` | `apply_noise_filter`, `apply_contrast_enhancement`, noise/CLAHE params | preprocessing |
| `TrainingConfig` | all training hyperparameters | conservative_trainer, train_args_builder |
| `HyperparameterConfig` | named config with HP values | hyperparameter_searcher |
| `FinetuningConfig` | `model_path`, `dataset_yaml`, `lr`, `epochs`, `freeze_layers` | finetuner |
| `MiningMetrics` | `recall`, `precision`, `ap50`, `overall_map50`, `overall_map` | metrics_extractor |
| `MetricsResult` | overall stats + optional `class_metrics` + `recommendations` | metrics_extractor, finetuner |

---

## Key Classes -- Quick Reference

### Core Data Processing

| Class | Method | Input | Output |
|-------|--------|-------|--------|
| `DatasetUnifier` | `unify()` | `data/{Imagenes,Etiquetas}/` | `data_unified/{images,labels}/` |
| `StrategicSplitter` | `split(images_by_class)` | `Dict[int, List[str]]` | `Dict[str, List[str]]` (train/val/test stems) |
| `StrategicSplitter` | `copy_to_yolo(splits)` | split dict | `yolo_dataset/{train,val,test}/` |
| `ClassAugmentor` | `augment_all(targets)` | `Dict[int, int]` | `Dict[int, int]` (generated counts) |
| `LabelCleaner` | `clean()` | label files in-place | `(files_modified, lines_corrected)` |
| `DuplicateRemover` | `clean()` | images in-place | `Dict[str, int]` (per-split removal counts) |
| `ClassBalancer` | `remove_excess(class_id)` | augmented images in-place | `(images_removed, objects_removed)` |

### Preprocessing

| Class | Method | Input | Output |
|-------|--------|-------|--------|
| `NoiseFilter` | `apply(image)` | BGR ndarray | `(filtered, noise_detected)` |
| `ContrastEnhancer` | `apply(image)` | BGR ndarray | enhanced BGR ndarray |
| `PreprocessingPipeline` | `process(image)` | BGR ndarray | `(processed, noise_detected)` |
| `DatasetPreprocessor` | `preprocess(src, dst)` | dataset paths | preprocessed dataset path |

### Training

| Class | Method | Input | Output |
|-------|--------|-------|--------|
| `GpuManager` | `clear_cache()` | -- | frees GPU memory |
| `GpuManager` | `estimate_usage(batch, img, weights)` | config values | estimated GB (float) |
| `ModelFactory` | `create(weights)` | weight name/path | `YoloAdapter` |
| `TrainArgsBuilder` | `build(dataset, run_name)` | paths | `Dict[str, Any]` (~50 keys) |
| `ConservativeTrainer` | `train(dataset_path)` | dataset path | `(YoloAdapter, metrics_dict)` |
| `HyperparameterSearcher` | `run_search(configs)` | HP configs list | `List[Dict]` results |
| `Finetuner` | `finetune(save_dir, run_name)` | -- | `(YoloAdapter, MetricsResult)` |
| `MetricsExtractor` | `extract_class_metrics(val_results)` | YOLO val output | `MiningMetrics` |
| `MetricsExtractor` | `evaluate_by_class(model, yaml)` | model + dataset | `MetricsResult` |
| `ResultsAnalyzer` | `generate_recommendations(metrics)` | `MiningMetrics` | `List[str]` |

---

## Coding Conventions

1. **One class per file.** No multiple classes in a single file. No classes inside methods.
2. **No global functions in class files.** Standalone functions belong in `utils/`.
3. **Pydantic only.** No `@dataclass`. All data structures are `BaseModel` or `BaseSettings`.
4. **All imports at the top.** No imports inside `try/except` or conditional blocks.
5. **English identifiers.** All code, docstrings, log messages, and variable names in English.
6. **No emojis in source code.** Structured logging via `get_logger()` replaces all emoji `print()` calls.
7. **Config via settings.** Every tunable parameter accessed as `settings.VARIABLE_NAME` from `app/config.py`.
8. **Comprehensive docstrings.** All classes and functions documented with parameters, return types, and exceptions.

---

## Running the Pipeline

```bash
# Full end-to-end
python scripts/run_full_pipeline.py

# Individual stages
python scripts/run_data_pipeline.py          # data preparation
python scripts/run_preprocessing.py          # CLAHE + noise filter
python scripts/run_training.py               # conservative training
python scripts/run_hyperparameter_search.py  # HP search
python scripts/run_finetuning.py --model path/to/best.pt
python scripts/run_evaluation.py --model path/to/best.pt

# Quick import verification
python -c "from app.config import settings; print(settings.TARGET_CLASS)"
```

---

## External Dependencies

| Package | Used for |
|---------|----------|
| `ultralytics` | YOLO model training, validation, inference |
| `torch` | GPU management, tensor operations |
| `pydantic` + `pydantic-settings` | Schema validation, settings management |
| `albumentations` | Bbox-aware image augmentation |
| `opencv-python` | Image I/O, preprocessing filters, background augmentation |
| `numpy` | Array operations |
| `pandas` | Results analysis and CSV export |
| `matplotlib` + `Pillow` | Visualization and plotting |
| `scikit-learn` | (available for splits, currently manual) |
| `tqdm` | Progress bars |

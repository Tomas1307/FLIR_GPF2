"""Entry point for dataset preprocessing (CLAHE + median filter).

Usage::

    python scripts/run_preprocessing.py
"""

from pathlib import Path

from app.config import settings
from app.preprocessing.dataset_preprocessor import DatasetPreprocessor
from app.schemas.preprocessing_schema import PreprocessingConfig
from app.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Run preprocessing on the YOLO dataset."""
    source = settings.YOLO_ROOT
    output = settings.PREPROCESSED_ROOT

    logger.info("Preprocessing %s -> %s", source, output)
    preprocessor = DatasetPreprocessor(PreprocessingConfig())
    preprocessor.preprocess(Path(source), Path(output))
    logger.info("Preprocessing complete")


if __name__ == "__main__":
    main()

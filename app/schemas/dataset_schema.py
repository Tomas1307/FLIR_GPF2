from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class DatasetConfig(BaseModel):
    """Schema describing the paths and structure of a YOLO dataset.

    Attributes:
        root: Root directory of the dataset.
        image_dirs: Mapping of split name to image directory path.
        label_dirs: Mapping of split name to label directory path.
        yaml_path: Optional path to the dataset YAML file.
        splits: List of split names present in this dataset.
    """

    root: Path = Field(..., description="Root directory of the dataset.")
    image_dirs: Dict[str, Path] = Field(
        default_factory=dict,
        description="Mapping of split name to image directory.",
    )
    label_dirs: Dict[str, Path] = Field(
        default_factory=dict,
        description="Mapping of split name to label directory.",
    )
    yaml_path: Optional[Path] = Field(
        default=None,
        description="Path to the dataset YAML configuration file.",
    )
    splits: List[str] = Field(
        default_factory=lambda: ["train", "val", "test"],
        description="List of split names.",
    )

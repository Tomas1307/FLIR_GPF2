from pydantic import BaseModel, Field


class FinetuningConfig(BaseModel):
    """Schema for fine-tuning configuration parameters.

    Attributes:
        model_path: Path to the pretrained YOLO model weights.
        dataset_yaml: Path to the dataset YAML file.
        lr: Learning rate for fine-tuning.
        epochs: Number of fine-tuning epochs.
        image_size: Image resolution during fine-tuning.
        batch_size: Batch size for fine-tuning.
        freeze_layers: Number of backbone layers to freeze.
        patience: Early stopping patience.
    """

    model_path: str = Field(..., description="Path to pretrained model weights.")
    dataset_yaml: str = Field(..., description="Path to dataset YAML file.")
    lr: float = Field(default=0.0005)
    epochs: int = Field(default=15)
    image_size: int = Field(default=640)
    batch_size: int = Field(default=40)
    freeze_layers: int = Field(default=10)
    patience: int = Field(default=3)

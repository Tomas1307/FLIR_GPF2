from typing import Optional

from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    """Schema for conservative training configuration.

    Attributes:
        model_weights: Pretrained model weights filename.
        batch_size: Training batch size.
        lr0: Initial learning rate.
        lrf: Final learning rate ratio.
        cos_lr: Use cosine learning rate schedule.
        weight_decay: L2 regularisation coefficient.
        dropout: Dropout probability.
        mosaic: Mosaic augmentation probability.
        mixup: MixUp augmentation probability.
        image_size: Training image resolution in pixels.
        epochs: Maximum number of training epochs.
        patience: Early stopping patience (epochs without improvement).
        workers: Number of data-loader workers.
        resume_from: Optional path to a checkpoint to resume training from.
    """

    model_weights: str = Field(default="yolo11m.pt")
    batch_size: int = Field(default=40)
    lr0: float = Field(default=0.005)
    lrf: float = Field(default=0.01)
    cos_lr: bool = Field(default=True)
    weight_decay: float = Field(default=0.001)
    dropout: float = Field(default=0.1)
    mosaic: float = Field(default=0.8)
    mixup: float = Field(default=0.0)
    image_size: int = Field(default=640)
    epochs: int = Field(default=100)
    patience: int = Field(default=3)
    workers: int = Field(default=16)
    resume_from: Optional[str] = Field(
        default=None,
        description="Path to a checkpoint to resume training from.",
    )

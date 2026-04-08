from pydantic import BaseModel, Field


class HyperparameterConfig(BaseModel):
    """Schema for a single hyperparameter search configuration.

    Attributes:
        name: Human-readable name for this configuration.
        batch_size: Batch size for training.
        lr0: Initial learning rate.
        lrf: Final learning rate ratio.
        cos_lr: Use cosine learning rate schedule.
        weight_decay: L2 regularisation coefficient.
        dropout: Dropout probability.
        mosaic: Mosaic augmentation probability.
        mixup: MixUp augmentation probability.
        image_size: Training image resolution in pixels.
        model_weights: Pretrained model weights filename.
    """

    name: str = Field(..., description="Configuration name.")
    batch_size: int = Field(default=40)
    lr0: float = Field(default=0.005)
    lrf: float = Field(default=0.01)
    cos_lr: bool = Field(default=True)
    weight_decay: float = Field(default=0.001)
    dropout: float = Field(default=0.1)
    mosaic: float = Field(default=0.8)
    mixup: float = Field(default=0.0)
    image_size: int = Field(default=640)
    model_weights: str = Field(default="yolo11m.pt")
    optimizer: str = Field(
        default="auto",
        description="Optimizer for training (auto, SGD, Adam, AdamW).",
    )

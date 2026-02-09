from pathlib import Path
from typing import Any, Dict, Optional

from ultralytics import YOLO

from app.utils.logger import get_logger

logger = get_logger(__name__)


class YoloAdapter:
    """Adapter wrapping the Ultralytics YOLO API.

    Provides a simplified interface for loading models, training, and
    validation so that the rest of the codebase is decoupled from the
    Ultralytics implementation details.

    Attributes:
        model: The underlying ``ultralytics.YOLO`` model instance.
    """

    def __init__(self, weights: str) -> None:
        """Initialise the YoloAdapter by loading a model.

        Args:
            weights: Path to pretrained YOLO weights or a model name
                (e.g. ``yolo11m.pt``).
        """
        logger.info("Loading YOLO model: %s", weights)
        self.model = YOLO(weights)

    def train(self, **kwargs: Any) -> Any:
        """Train the model with the given keyword arguments.

        All arguments are forwarded directly to ``YOLO.train()``.

        Args:
            **kwargs: Training arguments accepted by the Ultralytics API.

        Returns:
            The training results object.
        """
        logger.info("Starting YOLO training")
        return self.model.train(**kwargs)

    def validate(self, **kwargs: Any) -> Any:
        """Run validation with the given keyword arguments.

        All arguments are forwarded directly to ``YOLO.val()``.

        Args:
            **kwargs: Validation arguments accepted by the Ultralytics API.

        Returns:
            The validation results object.
        """
        return self.model.val(**kwargs)

    def predict(
        self,
        source: str,
        imgsz: int = 640,
        device: int = 0,
        verbose: bool = False,
    ) -> Any:
        """Run inference on the given source.

        Args:
            source: Path to the image or directory to predict on.
            imgsz: Image size for inference.
            device: Device index.
            verbose: Whether to print verbose output.

        Returns:
            The prediction results.
        """
        return self.model(source, imgsz=imgsz, device=device, verbose=verbose)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str) -> "YoloAdapter":
        """Create an adapter from a trained checkpoint.

        Args:
            checkpoint_path: Path to the ``best.pt`` or ``last.pt`` weights.

        Returns:
            A new ``YoloAdapter`` instance.

        Raises:
            FileNotFoundError: If the checkpoint does not exist.
        """
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )
        return cls(checkpoint_path)

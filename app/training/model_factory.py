from pathlib import Path

from app.adapters.yolo_adapter import YoloAdapter
from app.training.gpu_manager import GpuManager
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ModelFactory:
    """Factory for creating YOLO model instances with automatic GPU cleanup.

    Before loading a new model the factory clears the GPU cache to prevent
    out-of-memory errors during successive training runs.
    """

    def __init__(self) -> None:
        """Initialise the ModelFactory."""
        self._gpu = GpuManager()

    def create(self, weights: str) -> YoloAdapter:
        """Create a new YOLO model from pretrained weights.

        GPU memory is cleared before the model is loaded.

        Args:
            weights: Model weights path or name (e.g. ``yolo11m.pt``).

        Returns:
            A ``YoloAdapter`` wrapping the loaded model.
        """
        self._gpu.clear_cache()
        logger.info("Creating model from: %s", weights)
        return YoloAdapter(weights)

    def create_from_checkpoint(self, checkpoint_path: str) -> YoloAdapter:
        """Create a YOLO model from a trained checkpoint.

        GPU memory is cleared before loading.

        Args:
            checkpoint_path: Path to the ``best.pt`` or ``last.pt`` file.

        Returns:
            A ``YoloAdapter`` wrapping the loaded checkpoint.

        Raises:
            FileNotFoundError: If *checkpoint_path* does not exist.
        """
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )
        self._gpu.clear_cache()
        logger.info("Loading checkpoint: %s", checkpoint_path)
        return YoloAdapter(checkpoint_path)

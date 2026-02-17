import gc

import torch

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class GpuManager:
    """Singleton that manages GPU memory and provides memory estimation
    utilities.

    Only one instance is ever created regardless of how many times the
    class is instantiated, ensuring consistent memory management across
    the entire pipeline.

    Attributes:
        gpu_memory_gb: Total GPU memory available (in GB).
    """

    _instance = None

    def __new__(cls, *args, **kwargs):  # noqa: D102
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, gpu_memory_gb: int = settings.GPU_MEMORY_GB) -> None:
        """Initialise the GpuManager.

        Subsequent calls with different arguments are ignored because the
        singleton instance is already created.

        Args:
            gpu_memory_gb: Available GPU memory in gigabytes.
        """
        if not hasattr(self, "_initialized"):
            self.gpu_memory_gb = gpu_memory_gb
            self._initialized = True

    def clear_cache(self) -> None:
        """Release cached GPU memory and run garbage collection.

        Calls ``torch.cuda.empty_cache()`` and ``gc.collect()`` to free
        unused GPU memory between training runs.
        """
        logger.info("Clearing GPU memory cache")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024 ** 3
            reserved = torch.cuda.memory_reserved(0) / 1024 ** 3
            logger.info(
                "GPU memory -- allocated: %.2f GB, reserved: %.2f GB",
                allocated,
                reserved,
            )

    def estimate_usage(
        self, batch_size: int, image_size: int, model_weights: str
    ) -> float:
        """Estimate GPU memory usage for a given configuration.

        Args:
            batch_size: Training batch size.
            image_size: Image resolution in pixels.
            model_weights: Model weights filename (e.g. ``yolo11m.pt``).

        Returns:
            Estimated GPU memory consumption in gigabytes.
        """
        base_memory = {
            "yolo11m.pt": 4.0,
            "yolo11l.pt": 6.0,
        }
        base = base_memory.get(model_weights, 4.0)
        per_batch = batch_size * (image_size / 640) ** 2 * 0.1
        return base + per_batch

    def fits_in_memory(
        self,
        batch_size: int,
        image_size: int,
        model_weights: str,
        safety_margin: float = 0.9,
    ) -> bool:
        """Check whether a configuration fits within available GPU memory.

        Args:
            batch_size: Training batch size.
            image_size: Image resolution.
            model_weights: Model weights filename.
            safety_margin: Fraction of total memory considered usable.

        Returns:
            ``True`` if the estimated usage is within the safety margin.
        """
        estimated = self.estimate_usage(batch_size, image_size, model_weights)
        return estimated <= self.gpu_memory_gb * safety_margin

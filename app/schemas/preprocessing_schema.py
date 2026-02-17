from typing import Tuple

from pydantic import BaseModel, Field


class PreprocessingConfig(BaseModel):
    """Schema for image preprocessing parameters.

    Attributes:
        apply_noise_filter: Whether to apply salt-and-pepper noise reduction.
        apply_contrast_enhancement: Whether to apply CLAHE contrast enhancement.
        noise_threshold: Minimum ratio of extreme pixels to trigger median filter.
        median_kernel: Kernel size for the median blur filter.
        clahe_clip: Clip limit for CLAHE.
        clahe_tile_grid: Tile grid size for CLAHE.
    """

    apply_noise_filter: bool = Field(
        default=True, description="Enable salt-and-pepper noise reduction."
    )
    apply_contrast_enhancement: bool = Field(
        default=True, description="Enable CLAHE contrast enhancement."
    )
    noise_threshold: float = Field(
        default=0.01,
        description="Minimum ratio of extreme pixels to trigger noise filter.",
    )
    median_kernel: int = Field(
        default=3, description="Kernel size for the median blur filter."
    )
    clahe_clip: float = Field(
        default=2.0, description="Clip limit for CLAHE."
    )
    clahe_tile_grid: Tuple[int, int] = Field(
        default=(8, 8), description="Tile grid size for CLAHE."
    )

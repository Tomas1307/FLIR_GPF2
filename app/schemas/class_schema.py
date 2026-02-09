from pydantic import BaseModel, Field


class ClassDefinition(BaseModel):
    """Schema representing a single object-detection class.

    Attributes:
        class_id: Integer identifier for the class (-1 for background).
        name: Human-readable name of the class.
        is_background: Whether this class represents pure background.
    """

    class_id: int = Field(..., description="Integer identifier for the class.")
    name: str = Field(..., description="Human-readable class name.")
    is_background: bool = Field(
        default=False,
        description="True if this class represents background (no objects).",
    )

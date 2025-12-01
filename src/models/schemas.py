"""
Pydantic models for request/response validation.

This module defines all the data models used for API request validation
and response serialization using Pydantic.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class FaceEmbedding(BaseModel):
    """
    Extracted face data including embedding and metadata.

    This model represents a single detected face with its embedding vector
    and associated metadata like bounding box and detection confidence.

    Attributes:
        embedding: 512-dimensional face embedding vector
        bbox: Bounding box coordinates [x1, y1, x2, y2]
        keypoints: Facial landmark coordinates (5 points: eyes, nose, mouth corners)
        detection_score: Confidence score of face detection (0.0 to 1.0)
        gender: Predicted gender (0=female, 1=male) if available
        age: Predicted age if available

    Example:
        >>> face = FaceEmbedding(
        ...     embedding=[0.1] * 512,
        ...     bbox=[100, 100, 200, 200],
        ...     keypoints=[[120, 130], [180, 130], [150, 160], [130, 190], [170, 190]],
        ...     detection_score=0.95,
        ...     gender=1,
        ...     age=30
        ... )
    """

    embedding: list[float] = Field(description="512-dimensional face embedding")
    bbox: list[int] = Field(description="Bounding box [x1, y1, x2, y2]")
    keypoints: list[list[float]] = Field(description="Facial landmark coordinates")
    detection_score: float = Field(ge=0.0, le=1.0, description="Detection confidence")
    gender: Optional[int] = Field(default=None, description="Gender (0=F, 1=M)")
    age: Optional[int] = Field(default=None, ge=0, le=120, description="Predicted age")


class RepresentResponse(BaseModel):
    """
    Response from the /represent endpoint.

    Contains all detected faces with their embeddings and request metadata.

    Attributes:
        embeddings: List of detected faces with embeddings
        faces_detected: Number of faces found in the image
        processing_time_ms: Time taken to process the image in milliseconds
        request_id: Unique identifier for this request (for debugging)

    Example:
        >>> response = RepresentResponse(
        ...     embeddings=[],
        ...     faces_detected=0,
        ...     processing_time_ms=150.5,
        ...     request_id="abc12345"
        ... )
    """

    embeddings: list[FaceEmbedding] = Field(default_factory=list)
    faces_detected: int = Field(ge=0)
    processing_time_ms: float = Field(ge=0)
    request_id: str


class ErrorResponse(BaseModel):
    """
    Standardized error response format.

    Used for all API error responses to provide consistent error information.

    Attributes:
        error: Human-readable error message
        error_code: Machine-readable error code for programmatic handling
        request_id: Unique identifier for debugging (if available)

    Example:
        >>> error = ErrorResponse(
        ...     error="No image file provided",
        ...     error_code="RequestValidationError",
        ...     request_id="abc12345"
        ... )
    """

    error: str
    error_code: str
    request_id: Optional[str] = None


class HealthStatus(str, Enum):
    """
    Health check status values.

    Attributes:
        HEALTHY: All systems operational
        DEGRADED: Partial functionality available
        UNHEALTHY: Service is not operational
    """

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class HealthResponse(BaseModel):
    """
    Response from the /health endpoint.

    Provides detailed information about the service health and model status.

    Attributes:
        status: Overall health status
        model_loaded: Whether the ML model is ready for inference
        model_name: Name of the loaded model (if loaded)
        version: API version string
        uptime_seconds: Seconds since the model was loaded

    Example:
        >>> health = HealthResponse(
        ...     status=HealthStatus.HEALTHY,
        ...     model_loaded=True,
        ...     model_name="buffalo_l",
        ...     uptime_seconds=3600.0
        ... )
    """

    status: HealthStatus
    model_loaded: bool
    model_name: Optional[str] = None
    version: str = "2.0.0"
    uptime_seconds: float


class ImageInput(BaseModel):
    """
    Validated image input parameters.

    Used internally to validate uploaded image files before processing.

    Attributes:
        file_size: Size of the uploaded file in bytes
        content_type: MIME type of the uploaded file (if provided)
    """

    file_size: int = Field(ge=1)
    content_type: Optional[str] = None

    @field_validator("content_type")
    @classmethod
    def validate_content_type(cls, v: Optional[str]) -> Optional[str]:
        """
        Validate that content type is an accepted image format.

        Args:
            v: Content type string to validate

        Returns:
            Validated content type

        Raises:
            ValueError: If content type is not a supported image format
        """
        if v is None:
            return v
        allowed = {"image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp"}
        if v.lower() not in allowed:
            raise ValueError(f"Unsupported content type: {v}")
        return v

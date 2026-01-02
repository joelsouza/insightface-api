"""Custom exceptions for the InsightFace API."""

from src.exceptions.errors import (
    APIError,
    ImageDecodeError,
    ImageValidationError,
    InferenceTimeoutError,
    ModelNotReadyError,
    RequestValidationError,
)

__all__ = [
    "APIError",
    "ImageDecodeError",
    "ImageValidationError",
    "InferenceTimeoutError",
    "ModelNotReadyError",
    "RequestValidationError",
]

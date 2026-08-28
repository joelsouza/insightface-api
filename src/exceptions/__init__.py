"""Custom exceptions for the InsightFace API."""

from src.exceptions.errors import (
    APIError,
    ImageDecodeError,
    ImageDownloadError,
    ImageValidationError,
    InferenceTimeoutError,
    ModelNotReadyError,
    RequestValidationError,
    ServiceOverloadedError,
)

__all__ = [
    "APIError",
    "ImageDecodeError",
    "ImageDownloadError",
    "ImageValidationError",
    "InferenceTimeoutError",
    "ModelNotReadyError",
    "RequestValidationError",
    "ServiceOverloadedError",
]

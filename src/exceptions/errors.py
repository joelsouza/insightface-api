"""
Custom exception classes for the InsightFace API.

This module defines a hierarchy of exceptions used throughout the application
to handle various error conditions with appropriate HTTP status codes.

Exception Hierarchy:
    APIError (base)
    ├── ImageDecodeError (400) - Image decoding failures
    ├── ImageValidationError (400) - Image validation failures
    ├── RequestValidationError (400) - Request validation failures
    └── ModelNotReadyError (503) - Model not available
"""

from __future__ import annotations

from typing import Optional


class APIError(Exception):
    """
    Base exception for all API errors.

    Provides a consistent interface for error handling with HTTP status codes
    and machine-readable error codes.

    Attributes:
        message: Human-readable error description
        status_code: HTTP status code to return
        error_code: Machine-readable error identifier

    Example:
        >>> raise APIError("Something went wrong", status_code=400, error_code="BAD_INPUT")
    """

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: Optional[str] = None,
    ) -> None:
        """
        Initialize an API error.

        Args:
            message: Human-readable error description
            status_code: HTTP status code (default: 500)
            error_code: Machine-readable code (default: class name)
        """
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code or self.__class__.__name__


class ImageDecodeError(APIError):
    """
    Raised when image decoding fails.

    This typically occurs when the uploaded file is not a valid image
    or is corrupted.

    HTTP Status: 400 Bad Request
    """

    def __init__(self, message: str = "Failed to decode image") -> None:
        """
        Initialize an image decode error.

        Args:
            message: Error description (default: "Failed to decode image")
        """
        super().__init__(message, status_code=400)


class ImageValidationError(APIError):
    """
    Raised when image validation fails.

    This occurs when the image is valid but doesn't meet requirements
    (e.g., too small, wrong dimensions).

    HTTP Status: 400 Bad Request
    """

    def __init__(self, message: str) -> None:
        """
        Initialize an image validation error.

        Args:
            message: Description of what validation failed
        """
        super().__init__(message, status_code=400)


class ModelNotReadyError(APIError):
    """
    Raised when the ML model is not available for inference.

    This can occur during startup before the model is loaded,
    or if model initialization failed.

    HTTP Status: 503 Service Unavailable
    """

    def __init__(self, message: str = "Model not initialized") -> None:
        """
        Initialize a model not ready error.

        Args:
            message: Error description (default: "Model not initialized")
        """
        super().__init__(message, status_code=503)


class RequestValidationError(APIError):
    """
    Raised when request validation fails.

    This occurs when required fields are missing or have invalid values
    (e.g., no image file provided, file too large).

    HTTP Status: 400 Bad Request
    """

    def __init__(self, message: str) -> None:
        """
        Initialize a request validation error.

        Args:
            message: Description of what validation failed
        """
        super().__init__(message, status_code=400)

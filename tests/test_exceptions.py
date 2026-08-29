"""Tests for custom exception classes."""

from __future__ import annotations

import pytest

from src.exceptions import (
    APIError,
    ImageDecodeError,
    ImageValidationError,
    ModelNotReadyError,
    RequestValidationError,
)


class TestAPIError:
    """Tests for base APIError class."""

    def test_defaults(self) -> None:
        """Test APIError with default values."""
        error = APIError("Test error")
        assert error.message == "Test error"
        assert error.status_code == 500
        assert error.error_code == "APIError"

    def test_custom_values(self) -> None:
        """Test APIError with custom values."""
        error = APIError("Custom error", status_code=400, error_code="CUSTOM")
        assert error.message == "Custom error"
        assert error.status_code == 400
        assert error.error_code == "CUSTOM"

    def test_str_representation(self) -> None:
        """Test string representation."""
        error = APIError("Test message")
        assert str(error) == "Test message"


class TestImageDecodeError:
    """Tests for ImageDecodeError class."""

    def test_defaults(self) -> None:
        """Test ImageDecodeError defaults."""
        error = ImageDecodeError()
        assert error.status_code == 400
        assert error.error_code == "IMAGE_DECODE_FAILED"
        assert "decode" in error.message.lower()

    def test_custom_message(self) -> None:
        """Test ImageDecodeError with custom message."""
        error = ImageDecodeError("Custom decode error")
        assert error.message == "Custom decode error"


class TestImageValidationError:
    """Tests for ImageValidationError class."""

    def test_creation(self) -> None:
        """Test ImageValidationError creation."""
        error = ImageValidationError("Image too small")
        assert error.status_code == 400
        assert error.message == "Image too small"
        assert error.error_code == "IMAGE_VALIDATION_FAILED"


class TestModelNotReadyError:
    """Tests for ModelNotReadyError class."""

    def test_defaults(self) -> None:
        """Test ModelNotReadyError defaults."""
        error = ModelNotReadyError()
        assert error.status_code == 503
        assert error.error_code == "MODEL_NOT_READY"
        assert "not initialized" in error.message.lower()

    def test_custom_message(self) -> None:
        """Test ModelNotReadyError with custom message."""
        error = ModelNotReadyError("GPU not available")
        assert error.message == "GPU not available"


class TestRequestValidationError:
    """Tests for RequestValidationError class."""

    def test_creation(self) -> None:
        """Test RequestValidationError creation."""
        error = RequestValidationError("No file provided")
        assert error.status_code == 400
        assert error.message == "No file provided"
        assert error.error_code == "REQUEST_INVALID"

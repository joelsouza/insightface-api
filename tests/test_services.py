"""Tests for service layer."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from src.config import Settings
from src.exceptions import (
    ImageDecodeError,
    ImageValidationError,
    ModelNotReadyError,
    RequestValidationError,
)
from src.services import ModelManager
from src.services.image import (
    decode_image,
    extract_face_data,
    validate_image_dimensions,
    validate_image_file,
)


class TestValidateImageFile:
    """Tests for validate_image_file function."""

    def test_missing_file(self) -> None:
        """Test validation with missing file."""
        with pytest.raises(RequestValidationError) as exc_info:
            validate_image_file(None, 1024 * 1024)
        assert "No image file" in str(exc_info.value.message)

    def test_empty_filename(self) -> None:
        """Test validation with empty filename."""
        mock_file = MagicMock()
        mock_file.filename = ""
        with pytest.raises(RequestValidationError):
            validate_image_file(mock_file, 1024 * 1024)

    def test_empty_content(self) -> None:
        """Test validation with empty file content."""
        mock_file = MagicMock()
        mock_file.filename = "test.jpg"
        mock_file.seek = MagicMock()
        mock_file.tell = MagicMock(return_value=0)
        with pytest.raises(RequestValidationError) as exc_info:
            validate_image_file(mock_file, 1024 * 1024)
        assert "Empty file" in str(exc_info.value.message)

    def test_file_too_large(self) -> None:
        """Test validation with oversized file."""
        mock_file = MagicMock()
        mock_file.filename = "large.jpg"
        mock_file.seek = MagicMock()
        mock_file.tell = MagicMock(return_value=20 * 1024 * 1024)  # 20MB
        with pytest.raises(RequestValidationError) as exc_info:
            validate_image_file(mock_file, 16 * 1024 * 1024)
        assert "too large" in str(exc_info.value.message).lower()

    def test_valid_file(self) -> None:
        """Test validation with valid file."""
        mock_file = MagicMock()
        mock_file.filename = "test.jpg"
        mock_file.content_type = "image/jpeg"
        mock_file.seek = MagicMock()
        mock_file.tell = MagicMock(return_value=1024)
        result = validate_image_file(mock_file, 16 * 1024 * 1024)
        assert result.file_size == 1024


class TestDecodeImage:
    """Tests for decode_image function."""

    def test_invalid_bytes(self) -> None:
        """Test decoding invalid image bytes."""
        with pytest.raises(ImageDecodeError):
            decode_image(b"not an image")

    def test_valid_jpeg(self) -> None:
        """Test decoding valid JPEG bytes."""
        # Create a real image in memory
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        _, encoded = cv2.imencode(".jpg", img)
        result = decode_image(encoded.tobytes())
        assert result is not None
        assert len(result.shape) == 3


class TestValidateImageDimensions:
    """Tests for validate_image_dimensions function."""

    def test_valid_dimensions(self, valid_image_array: np.ndarray) -> None:
        """Test dimension validation with valid image."""
        # Should not raise
        validate_image_dimensions(valid_image_array, 640)

    def test_too_small(self, small_image_array: np.ndarray) -> None:
        """Test dimension validation with too-small image."""
        with pytest.raises(ImageValidationError) as exc_info:
            validate_image_dimensions(small_image_array, 640)
        assert "too small" in str(exc_info.value.message).lower()


class TestExtractFaceData:
    """Tests for extract_face_data function."""

    def test_full_face_data(self, mock_face: MagicMock) -> None:
        """Test face data extraction with all attributes."""
        result = extract_face_data(mock_face)
        assert len(result.embedding) == 512
        assert result.detection_score == 0.95
        assert result.gender == 1
        assert result.age == 30

    def test_face_without_optional_attrs(
        self, mock_face_no_attributes: MagicMock
    ) -> None:
        """Test face data extraction without optional attributes."""
        result = extract_face_data(mock_face_no_attributes)
        assert result.gender is None
        assert result.age is None


class TestModelManager:
    """Tests for ModelManager class."""

    def test_initial_state(
        self, settings: Settings, mock_logger: MagicMock
    ) -> None:
        """Test ModelManager initial state."""
        manager = ModelManager(settings=settings, logger=mock_logger)
        assert manager.is_loaded is False
        assert manager.model is None
        assert manager.uptime == 0.0

    def test_get_faces_not_loaded(
        self, settings: Settings, mock_logger: MagicMock
    ) -> None:
        """Test get_faces when model not loaded."""
        manager = ModelManager(settings=settings, logger=mock_logger)
        with pytest.raises(ModelNotReadyError):
            manager.get_faces(np.zeros((100, 100, 3), dtype=np.uint8))

    def test_get_faces_loaded(
        self,
        mock_model_manager: ModelManager,
        valid_image_array: np.ndarray,
        mock_face: MagicMock,
    ) -> None:
        """Test get_faces when model is loaded."""
        mock_model_manager.model.get.return_value = [mock_face]
        result = mock_model_manager.get_faces(valid_image_array)
        assert len(result) == 1

    def test_unload(self, mock_model_manager: ModelManager) -> None:
        """Test model unloading."""
        mock_model_manager.unload()
        assert mock_model_manager.model is None
        assert mock_model_manager.is_loaded is False

    def test_uptime(self, mock_model_manager: ModelManager) -> None:
        """Test uptime calculation."""
        mock_model_manager.load_time = time.time() - 100
        assert mock_model_manager.uptime >= 100

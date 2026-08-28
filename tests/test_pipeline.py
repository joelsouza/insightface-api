"""
Tests for the single-job image pipeline.

The pipeline runs decode, validation, inference, and extraction in one
thread-pool job, and returns face coordinates in original image pixels even
when the image was decoded at half resolution.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.config import Settings
from src.exceptions import ImageValidationError
from src.services import extract_face_data, process_image


@pytest.fixture
def model_manager(mock_face: MagicMock) -> MagicMock:
    """Model manager stub that always finds one face."""
    manager = MagicMock()
    manager.get_faces.return_value = [mock_face]
    return manager


class TestProcessImage:
    """Tests for process_image."""

    def test_small_jpeg_is_decoded_at_full_size(
        self, jpeg_bytes: bytes, settings: Settings, model_manager: MagicMock
    ) -> None:
        """Test that a small JPEG keeps full resolution."""
        result = process_image(jpeg_bytes, model_manager, settings)

        assert result.downscaled is False
        assert (result.width, result.height) == (200, 200)
        assert len(result.faces) == 1
        assert result.faces[0].bbox == [100, 100, 200, 200]

        image = model_manager.get_faces.call_args[0][0]
        assert image.shape == (200, 200, 3)

    def test_large_jpeg_is_decoded_at_half_size(
        self,
        large_jpeg_bytes: bytes,
        settings: Settings,
        model_manager: MagicMock,
    ) -> None:
        """Test that a large JPEG is decoded at half size and scaled back."""
        result = process_image(large_jpeg_bytes, model_manager, settings)

        assert result.downscaled is True
        # Reported size stays in original pixels
        assert (result.width, result.height) == (2560, 1440)

        # The model saw the half-size image
        image = model_manager.get_faces.call_args[0][0]
        assert image.shape == (720, 1280, 3)

        # Coordinates are scaled back to original pixels
        assert result.faces[0].bbox == [200, 200, 400, 400]
        assert result.faces[0].keypoints[0] == [240.0, 260.0]

    def test_png_is_never_downscaled(
        self, png_bytes: bytes, settings: Settings, model_manager: MagicMock
    ) -> None:
        """Test that the half-size shortcut only applies to JPEGs."""
        result = process_image(png_bytes, model_manager, settings)
        assert result.downscaled is False

    def test_rejects_non_image_bytes(
        self, settings: Settings, model_manager: MagicMock
    ) -> None:
        """Test that a non-image payload is rejected before decoding."""
        with pytest.raises(ImageValidationError):
            process_image(b"not an image at all", model_manager, settings)

    def test_reports_timings(
        self, jpeg_bytes: bytes, settings: Settings, model_manager: MagicMock
    ) -> None:
        """Test that decode and detect timings are recorded."""
        result = process_image(jpeg_bytes, model_manager, settings)
        assert result.decode_ms >= 0.0
        assert result.detect_ms >= 0.0


class TestEmbeddingRounding:
    """Tests for embedding size reduction."""

    def test_rounds_to_six_decimals_by_default(
        self, mock_face: MagicMock
    ) -> None:
        """Test that embeddings are rounded to 6 decimals."""
        result = extract_face_data(mock_face)

        assert len(result.embedding) == 512
        for value in result.embedding:
            assert value == round(value, 6)

    def test_rounding_preserves_the_vector(self, mock_face: MagicMock) -> None:
        """Test that rounding does not change the embedding materially."""
        original = mock_face.embedding.astype(np.float64)
        rounded = np.array(extract_face_data(mock_face).embedding)

        cosine = float(
            original @ rounded
            / (np.linalg.norm(original) * np.linalg.norm(rounded))
        )
        assert cosine > 0.9999999

    def test_decimals_is_configurable(self, mock_face: MagicMock) -> None:
        """Test that the decimal count can be changed."""
        result = extract_face_data(mock_face, decimals=2)
        for value in result.embedding:
            assert value == round(value, 2)

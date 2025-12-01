"""
Test Suite for InsightFace API v2

Comprehensive tests covering:
- Unit tests for image processing functions
- Unit tests for model manager
- Integration tests for API endpoints
- Error handling tests
- Edge cases and validation

Run with: pytest test_api_v2.py -v
"""

from __future__ import annotations

import io
import sys
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from flask import Flask
from flask.testing import FlaskClient

# Mock insightface before importing api_v2 to avoid import errors
sys.modules["insightface"] = MagicMock()
sys.modules["insightface.app"] = MagicMock()

# Import the module under test (after mocking insightface)
from api_v2 import (
    APIError,
    FaceEmbedding,
    HealthResponse,
    HealthStatus,
    ImageDecodeError,
    ImageValidationError,
    ModelManager,
    ModelNotReadyError,
    RepresentResponse,
    RequestValidationError,
    Settings,
    create_app,
    decode_image,
    extract_face_data,
    validate_image_dimensions,
    validate_image_file,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def settings() -> Settings:
    """Create test settings with defaults."""
    return Settings(
        port=5001,
        max_content_length=16 * 1024 * 1024,
        max_image_dimension=640,
        detection_threshold=0.5,
        model_name="buffalo_l",
        log_level="WARNING",  # Reduce log noise in tests
    )


@pytest.fixture
def mock_logger() -> MagicMock:
    """Create a mock logger."""
    return MagicMock()


@pytest.fixture
def mock_face() -> MagicMock:
    """Create a mock InsightFace Face object."""
    face = MagicMock()
    face.embedding = np.random.rand(512).astype(np.float32)
    face.bbox = np.array([100, 100, 200, 200], dtype=np.float32)
    face.kps = np.array([
        [120.0, 130.0],
        [180.0, 130.0],
        [150.0, 160.0],
        [130.0, 190.0],
        [170.0, 190.0],
    ])
    face.det_score = 0.95
    face.gender = 1
    face.age = 30
    return face


@pytest.fixture
def mock_face_no_attributes() -> MagicMock:
    """Create a mock face without gender/age attributes."""
    face = MagicMock()
    face.embedding = np.random.rand(512).astype(np.float32)
    face.bbox = np.array([50, 50, 100, 100], dtype=np.float32)
    face.kps = np.array([
        [60.0, 65.0],
        [90.0, 65.0],
        [75.0, 80.0],
        [65.0, 95.0],
        [85.0, 95.0],
    ])
    face.det_score = 0.88
    # Remove gender and age attributes
    del face.gender
    del face.age
    return face


@pytest.fixture
def mock_model_manager(settings: Settings, mock_logger: MagicMock) -> ModelManager:
    """Create a model manager with mocked model."""
    manager = ModelManager(settings=settings, logger=mock_logger)
    manager.model = MagicMock()
    manager.is_loaded = True
    manager.load_time = 1000.0
    return manager


@pytest.fixture
def valid_jpeg_bytes() -> bytes:
    """Create valid JPEG image bytes (1x1 red pixel)."""
    # Minimal valid JPEG - 1x1 red pixel
    return bytes([
        0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
        0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43,
        0x00, 0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08, 0x07, 0x07, 0x07, 0x09,
        0x09, 0x08, 0x0A, 0x0C, 0x14, 0x0D, 0x0C, 0x0B, 0x0B, 0x0C, 0x19, 0x12,
        0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E, 0x1D, 0x1A, 0x1C, 0x1C, 0x20,
        0x24, 0x2E, 0x27, 0x20, 0x22, 0x2C, 0x23, 0x1C, 0x1C, 0x28, 0x37, 0x29,
        0x2C, 0x30, 0x31, 0x34, 0x34, 0x34, 0x1F, 0x27, 0x39, 0x3D, 0x38, 0x32,
        0x3C, 0x2E, 0x33, 0x34, 0x32, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01,
        0x00, 0x01, 0x01, 0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00, 0x1F, 0x00, 0x00,
        0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
        0x09, 0x0A, 0x0B, 0xFF, 0xC4, 0x00, 0xB5, 0x10, 0x00, 0x02, 0x01, 0x03,
        0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00, 0x01, 0x7D,
        0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06,
        0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08,
        0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0, 0x24, 0x33, 0x62, 0x72,
        0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28,
        0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45,
        0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
        0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73, 0x74, 0x75,
        0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
        0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3,
        0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6,
        0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9,
        0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
        0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4,
        0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA, 0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01,
        0x00, 0x00, 0x3F, 0x00, 0xFB, 0xD5, 0xDB, 0x20, 0xA8, 0xF0, 0x00, 0x00,
        0x00, 0x00, 0xFF, 0xD9
    ])


@pytest.fixture
def valid_image_array() -> np.ndarray:
    """Create a valid test image array (100x100 RGB)."""
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def small_image_array() -> np.ndarray:
    """Create a too-small image array (5x5)."""
    return np.random.randint(0, 255, (5, 5, 3), dtype=np.uint8)


@pytest.fixture
def app(settings: Settings) -> Flask:
    """Create test Flask application with mocked model."""
    with patch("api_v2.ModelManager.load") as mock_load:
        mock_load.return_value = True
        test_app = create_app(settings)
        test_app.config["model_manager"].is_loaded = True
        test_app.config["model_manager"].load_time = 1000.0
        yield test_app


@pytest.fixture
def client(app: Flask) -> FlaskClient:
    """Create test client."""
    return app.test_client()


# =============================================================================
# Unit Tests: Custom Exceptions
# =============================================================================


class TestExceptions:
    """Tests for custom exception classes."""

    def test_api_error_defaults(self) -> None:
        """Test APIError with default values."""
        error = APIError("Test error")
        assert error.message == "Test error"
        assert error.status_code == 500
        assert error.error_code == "APIError"

    def test_api_error_custom_values(self) -> None:
        """Test APIError with custom values."""
        error = APIError("Custom error", status_code=400, error_code="CUSTOM")
        assert error.message == "Custom error"
        assert error.status_code == 400
        assert error.error_code == "CUSTOM"

    def test_image_decode_error(self) -> None:
        """Test ImageDecodeError defaults."""
        error = ImageDecodeError()
        assert error.status_code == 400
        assert error.error_code == "ImageDecodeError"

    def test_model_not_ready_error(self) -> None:
        """Test ModelNotReadyError defaults."""
        error = ModelNotReadyError()
        assert error.status_code == 503
        assert "not initialized" in error.message.lower()

    def test_request_validation_error(self) -> None:
        """Test RequestValidationError."""
        error = RequestValidationError("Invalid input")
        assert error.status_code == 400
        assert error.message == "Invalid input"


# =============================================================================
# Unit Tests: Pydantic Models
# =============================================================================


class TestPydanticModels:
    """Tests for Pydantic request/response models."""

    def test_face_embedding_valid(self) -> None:
        """Test FaceEmbedding with valid data."""
        embedding = FaceEmbedding(
            embedding=[0.1] * 512,
            bbox=[100, 100, 200, 200],
            keypoints=[[1.0, 2.0]] * 5,
            detection_score=0.95,
            gender=1,
            age=30,
        )
        assert len(embedding.embedding) == 512
        assert embedding.detection_score == 0.95

    def test_face_embedding_optional_fields(self) -> None:
        """Test FaceEmbedding with optional fields as None."""
        embedding = FaceEmbedding(
            embedding=[0.1] * 512,
            bbox=[100, 100, 200, 200],
            keypoints=[[1.0, 2.0]] * 5,
            detection_score=0.8,
        )
        assert embedding.gender is None
        assert embedding.age is None

    def test_represent_response(self) -> None:
        """Test RepresentResponse model."""
        response = RepresentResponse(
            embeddings=[],
            faces_detected=0,
            processing_time_ms=50.5,
            request_id="abc123",
        )
        assert response.faces_detected == 0
        assert response.request_id == "abc123"

    def test_health_response(self) -> None:
        """Test HealthResponse model."""
        response = HealthResponse(
            status=HealthStatus.HEALTHY,
            model_loaded=True,
            model_name="buffalo_l",
            uptime_seconds=100.0,
        )
        assert response.status == HealthStatus.HEALTHY
        assert response.version == "2.0.0"


# =============================================================================
# Unit Tests: Image Processing
# =============================================================================


class TestImageProcessing:
    """Tests for image processing functions."""

    def test_validate_image_file_missing(self) -> None:
        """Test validation with missing file."""
        with pytest.raises(RequestValidationError) as exc_info:
            validate_image_file(None, 1024 * 1024)
        assert "No image file" in str(exc_info.value.message)

    def test_validate_image_file_empty_filename(self) -> None:
        """Test validation with empty filename."""
        mock_file = MagicMock()
        mock_file.filename = ""
        with pytest.raises(RequestValidationError):
            validate_image_file(mock_file, 1024 * 1024)

    def test_validate_image_file_empty_content(self) -> None:
        """Test validation with empty file content."""
        mock_file = MagicMock()
        mock_file.filename = "test.jpg"
        mock_file.seek = MagicMock()
        mock_file.tell = MagicMock(return_value=0)
        with pytest.raises(RequestValidationError) as exc_info:
            validate_image_file(mock_file, 1024 * 1024)
        assert "Empty file" in str(exc_info.value.message)

    def test_validate_image_file_too_large(self) -> None:
        """Test validation with oversized file."""
        mock_file = MagicMock()
        mock_file.filename = "large.jpg"
        mock_file.seek = MagicMock()
        mock_file.tell = MagicMock(return_value=20 * 1024 * 1024)  # 20MB
        with pytest.raises(RequestValidationError) as exc_info:
            validate_image_file(mock_file, 16 * 1024 * 1024)
        assert "too large" in str(exc_info.value.message).lower()

    def test_validate_image_file_valid(self) -> None:
        """Test validation with valid file."""
        mock_file = MagicMock()
        mock_file.filename = "test.jpg"
        mock_file.content_type = "image/jpeg"
        mock_file.seek = MagicMock()
        mock_file.tell = MagicMock(return_value=1024)
        result = validate_image_file(mock_file, 16 * 1024 * 1024)
        assert result.file_size == 1024

    def test_decode_image_invalid(self) -> None:
        """Test decoding invalid image bytes."""
        with pytest.raises(ImageDecodeError):
            decode_image(b"not an image")

    def test_decode_image_valid(self, valid_jpeg_bytes: bytes) -> None:
        """Test decoding valid JPEG bytes."""
        # This test may fail with minimal JPEG, using numpy array instead
        import cv2
        # Create a real image in memory
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        _, encoded = cv2.imencode(".jpg", img)
        result = decode_image(encoded.tobytes())
        assert result is not None
        assert len(result.shape) == 3

    def test_validate_image_dimensions_valid(
        self, valid_image_array: np.ndarray
    ) -> None:
        """Test dimension validation with valid image."""
        # Should not raise
        validate_image_dimensions(valid_image_array, 640)

    def test_validate_image_dimensions_too_small(
        self, small_image_array: np.ndarray
    ) -> None:
        """Test dimension validation with too-small image."""
        with pytest.raises(ImageValidationError) as exc_info:
            validate_image_dimensions(small_image_array, 640)
        assert "too small" in str(exc_info.value.message).lower()

    def test_extract_face_data(self, mock_face: MagicMock) -> None:
        """Test face data extraction."""
        result = extract_face_data(mock_face)
        assert len(result.embedding) == 512
        assert result.detection_score == 0.95
        assert result.gender == 1
        assert result.age == 30

    def test_extract_face_data_no_attributes(
        self, mock_face_no_attributes: MagicMock
    ) -> None:
        """Test face data extraction without optional attributes."""
        result = extract_face_data(mock_face_no_attributes)
        assert result.gender is None
        assert result.age is None


# =============================================================================
# Unit Tests: Model Manager
# =============================================================================


class TestModelManager:
    """Tests for ModelManager class."""

    def test_model_manager_initial_state(
        self, settings: Settings, mock_logger: MagicMock
    ) -> None:
        """Test ModelManager initial state."""
        manager = ModelManager(settings=settings, logger=mock_logger)
        assert manager.is_loaded is False
        assert manager.model is None
        assert manager.uptime == 0.0

    def test_model_manager_get_faces_not_loaded(
        self, settings: Settings, mock_logger: MagicMock
    ) -> None:
        """Test get_faces when model not loaded."""
        manager = ModelManager(settings=settings, logger=mock_logger)
        with pytest.raises(ModelNotReadyError):
            manager.get_faces(np.zeros((100, 100, 3), dtype=np.uint8))

    def test_model_manager_get_faces_loaded(
        self,
        mock_model_manager: ModelManager,
        valid_image_array: np.ndarray,
        mock_face: MagicMock,
    ) -> None:
        """Test get_faces when model is loaded."""
        mock_model_manager.model.get.return_value = [mock_face]
        result = mock_model_manager.get_faces(valid_image_array)
        assert len(result) == 1

    def test_model_manager_unload(self, mock_model_manager: ModelManager) -> None:
        """Test model unloading."""
        mock_model_manager.unload()
        assert mock_model_manager.model is None
        assert mock_model_manager.is_loaded is False

    def test_model_manager_uptime(self, mock_model_manager: ModelManager) -> None:
        """Test uptime calculation."""
        import time
        mock_model_manager.load_time = time.time() - 100
        assert mock_model_manager.uptime >= 100


# =============================================================================
# Integration Tests: API Endpoints
# =============================================================================


class TestAPIEndpoints:
    """Integration tests for API endpoints."""

    def test_index_endpoint(self, client: FlaskClient) -> None:
        """Test GET / returns API info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.get_json()
        assert data["name"] == "InsightFace API"
        assert "version" in data

    def test_health_endpoint_healthy(self, client: FlaskClient) -> None:
        """Test GET /health when model is loaded."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

    def test_health_endpoint_unhealthy(self, app: Flask) -> None:
        """Test GET /health when model is not loaded."""
        app.config["model_manager"].is_loaded = False
        app.config["model_manager"]._initialization_error = "Test error"
        client = app.test_client()
        response = client.get("/health")
        assert response.status_code == 503
        data = response.get_json()
        assert data["status"] == "unhealthy"

    def test_up_endpoint(self, client: FlaskClient) -> None:
        """Test GET /up backwards compatibility."""
        response = client.get("/up")
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "ok"

    def test_represent_no_file(self, client: FlaskClient) -> None:
        """Test POST /represent without file."""
        response = client.post("/represent")
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data
        assert "No image" in data["error"]

    def test_represent_empty_file(self, client: FlaskClient) -> None:
        """Test POST /represent with empty file."""
        response = client.post(
            "/represent",
            data={"image_file": (io.BytesIO(b""), "empty.jpg")},
            content_type="multipart/form-data",
        )
        assert response.status_code == 400

    def test_represent_invalid_image(self, client: FlaskClient) -> None:
        """Test POST /represent with invalid image data."""
        response = client.post(
            "/represent",
            data={"image_file": (io.BytesIO(b"not an image"), "test.jpg")},
            content_type="multipart/form-data",
        )
        assert response.status_code == 400
        data = response.get_json()
        assert "decode" in data["error"].lower() or "error" in data

    def test_represent_valid_image(
        self, app: Flask, mock_face: MagicMock
    ) -> None:
        """Test POST /represent with valid image."""
        import cv2

        # Create a real test image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, encoded = cv2.imencode(".jpg", img)
        image_bytes = encoded.tobytes()

        # Mock the model to return a face
        app.config["model_manager"].model = MagicMock()
        app.config["model_manager"].model.get.return_value = [mock_face]

        client = app.test_client()
        response = client.post(
            "/represent",
            data={"image_file": (io.BytesIO(image_bytes), "test.jpg")},
            content_type="multipart/form-data",
        )

        assert response.status_code == 200
        data = response.get_json()
        assert "embeddings" in data
        assert data["faces_detected"] == 1
        assert "request_id" in data
        assert "processing_time_ms" in data

    def test_represent_no_faces(self, app: Flask) -> None:
        """Test POST /represent with image containing no faces."""
        import cv2

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, encoded = cv2.imencode(".jpg", img)
        image_bytes = encoded.tobytes()

        # Mock the model to return no faces
        app.config["model_manager"].model = MagicMock()
        app.config["model_manager"].model.get.return_value = []

        client = app.test_client()
        response = client.post(
            "/represent",
            data={"image_file": (io.BytesIO(image_bytes), "test.jpg")},
            content_type="multipart/form-data",
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data["faces_detected"] == 0
        assert data["embeddings"] == []

    def test_request_id_header(self, client: FlaskClient) -> None:
        """Test that X-Request-ID header is returned."""
        response = client.get("/health")
        assert "X-Request-ID" in response.headers


# =============================================================================
# Integration Tests: Error Handling
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_model_not_ready_error(self, app: Flask) -> None:
        """Test error when model is not ready."""
        import cv2

        app.config["model_manager"].is_loaded = False

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, encoded = cv2.imencode(".jpg", img)

        client = app.test_client()
        response = client.post(
            "/represent",
            data={"image_file": (io.BytesIO(encoded.tobytes()), "test.jpg")},
            content_type="multipart/form-data",
        )

        assert response.status_code == 503
        data = response.get_json()
        assert "error" in data

    def test_unexpected_error_handling(self, app: Flask) -> None:
        """Test handling of unexpected errors."""
        import cv2

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, encoded = cv2.imencode(".jpg", img)

        # Make model.get raise an unexpected error
        app.config["model_manager"].model = MagicMock()
        app.config["model_manager"].model.get.side_effect = RuntimeError(
            "Unexpected"
        )

        client = app.test_client()
        response = client.post(
            "/represent",
            data={"image_file": (io.BytesIO(encoded.tobytes()), "test.jpg")},
            content_type="multipart/form-data",
        )

        assert response.status_code == 500
        data = response.get_json()
        assert "error" in data
        assert data["error_code"] == "INTERNAL_ERROR"


# =============================================================================
# Settings Tests
# =============================================================================


class TestSettings:
    """Tests for Settings configuration."""

    def test_default_settings(self) -> None:
        """Test default settings values."""
        settings = Settings()
        assert settings.port == 5001
        assert settings.max_image_dimension == 640
        assert settings.detection_threshold == 0.5

    def test_settings_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test settings loaded from environment."""
        monkeypatch.setenv("PORT", "8080")
        monkeypatch.setenv("MAX_IMAGE_DIMENSION", "1024")
        settings = Settings()
        assert settings.port == 8080
        assert settings.max_image_dimension == 1024

    def test_settings_validation(self) -> None:
        """Test settings validation."""
        with pytest.raises(ValueError):
            Settings(port=0)  # Invalid port

        with pytest.raises(ValueError):
            Settings(detection_threshold=1.5)  # Invalid threshold


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

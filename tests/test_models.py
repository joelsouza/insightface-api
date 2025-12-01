"""Tests for Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models import (
    ErrorResponse,
    FaceEmbedding,
    HealthResponse,
    HealthStatus,
    ImageInput,
    RepresentResponse,
)


class TestFaceEmbedding:
    """Tests for FaceEmbedding model."""

    def test_valid_creation(self) -> None:
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
        assert embedding.gender == 1
        assert embedding.age == 30

    def test_optional_fields(self) -> None:
        """Test FaceEmbedding with optional fields as None."""
        embedding = FaceEmbedding(
            embedding=[0.1] * 512,
            bbox=[100, 100, 200, 200],
            keypoints=[[1.0, 2.0]] * 5,
            detection_score=0.8,
        )
        assert embedding.gender is None
        assert embedding.age is None

    def test_detection_score_bounds(self) -> None:
        """Test detection_score validation."""
        with pytest.raises(ValidationError):
            FaceEmbedding(
                embedding=[0.1] * 512,
                bbox=[100, 100, 200, 200],
                keypoints=[[1.0, 2.0]] * 5,
                detection_score=1.5,  # Invalid
            )


class TestRepresentResponse:
    """Tests for RepresentResponse model."""

    def test_creation(self) -> None:
        """Test RepresentResponse creation."""
        response = RepresentResponse(
            embeddings=[],
            faces_detected=0,
            processing_time_ms=50.5,
            request_id="abc123",
        )
        assert response.faces_detected == 0
        assert response.request_id == "abc123"
        assert response.processing_time_ms == 50.5

    def test_with_embeddings(self) -> None:
        """Test RepresentResponse with embeddings."""
        face = FaceEmbedding(
            embedding=[0.1] * 512,
            bbox=[100, 100, 200, 200],
            keypoints=[[1.0, 2.0]] * 5,
            detection_score=0.9,
        )
        response = RepresentResponse(
            embeddings=[face],
            faces_detected=1,
            processing_time_ms=100.0,
            request_id="xyz789",
        )
        assert len(response.embeddings) == 1


class TestErrorResponse:
    """Tests for ErrorResponse model."""

    def test_creation(self) -> None:
        """Test ErrorResponse creation."""
        response = ErrorResponse(
            error="Something went wrong",
            error_code="INTERNAL_ERROR",
            request_id="abc123",
        )
        assert response.error == "Something went wrong"
        assert response.error_code == "INTERNAL_ERROR"

    def test_optional_request_id(self) -> None:
        """Test ErrorResponse without request_id."""
        response = ErrorResponse(
            error="Error message",
            error_code="ERROR",
        )
        assert response.request_id is None


class TestHealthResponse:
    """Tests for HealthResponse model."""

    def test_healthy(self) -> None:
        """Test HealthResponse for healthy status."""
        response = HealthResponse(
            status=HealthStatus.HEALTHY,
            model_loaded=True,
            model_name="buffalo_l",
            uptime_seconds=100.0,
        )
        assert response.status == HealthStatus.HEALTHY
        assert response.version == "2.0.0"

    def test_unhealthy(self) -> None:
        """Test HealthResponse for unhealthy status."""
        response = HealthResponse(
            status=HealthStatus.UNHEALTHY,
            model_loaded=False,
            uptime_seconds=0.0,
        )
        assert response.status == HealthStatus.UNHEALTHY
        assert response.model_name is None


class TestImageInput:
    """Tests for ImageInput model."""

    def test_valid_creation(self) -> None:
        """Test ImageInput with valid data."""
        input_data = ImageInput(
            file_size=1024,
            content_type="image/jpeg",
        )
        assert input_data.file_size == 1024
        assert input_data.content_type == "image/jpeg"

    def test_invalid_content_type(self) -> None:
        """Test ImageInput with invalid content type."""
        with pytest.raises(ValidationError):
            ImageInput(
                file_size=1024,
                content_type="application/pdf",
            )

    def test_none_content_type(self) -> None:
        """Test ImageInput with None content type."""
        input_data = ImageInput(file_size=1024)
        assert input_data.content_type is None

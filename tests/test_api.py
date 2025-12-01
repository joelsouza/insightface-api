"""Tests for API endpoints."""

from __future__ import annotations

import io
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from flask import Flask
from flask.testing import FlaskClient


class TestIndexEndpoint:
    """Tests for GET / endpoint."""

    def test_returns_api_info(self, client: FlaskClient) -> None:
        """Test GET / returns API info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.get_json()
        assert data["name"] == "InsightFace API"
        assert "version" in data
        assert "endpoints" in data


class TestHealthEndpoint:
    """Tests for GET /health endpoint."""

    def test_healthy(self, client: FlaskClient) -> None:
        """Test GET /health when model is loaded."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

    def test_unhealthy(self, app: Flask) -> None:
        """Test GET /health when model is not loaded."""
        app.config["model_manager"].is_loaded = False
        app.config["model_manager"]._initialization_error = "Test error"
        client = app.test_client()
        response = client.get("/health")
        assert response.status_code == 503
        data = response.get_json()
        assert data["status"] == "unhealthy"


class TestUpEndpoint:
    """Tests for GET /up endpoint."""

    def test_returns_ok(self, client: FlaskClient) -> None:
        """Test GET /up backwards compatibility."""
        response = client.get("/up")
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "ok"


class TestRepresentEndpoint:
    """Tests for POST /represent endpoint."""

    def test_no_file(self, client: FlaskClient) -> None:
        """Test POST /represent without file."""
        response = client.post("/represent")
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data
        assert "No image" in data["error"]

    def test_empty_file(self, client: FlaskClient) -> None:
        """Test POST /represent with empty file."""
        response = client.post(
            "/represent",
            data={"image_file": (io.BytesIO(b""), "empty.jpg")},
            content_type="multipart/form-data",
        )
        assert response.status_code == 400

    def test_invalid_image(self, client: FlaskClient) -> None:
        """Test POST /represent with invalid image data."""
        response = client.post(
            "/represent",
            data={"image_file": (io.BytesIO(b"not an image"), "test.jpg")},
            content_type="multipart/form-data",
        )
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data

    def test_valid_image(self, app: Flask, mock_face: MagicMock) -> None:
        """Test POST /represent with valid image."""
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

    def test_no_faces_detected(self, app: Flask) -> None:
        """Test POST /represent with image containing no faces."""
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


class TestRequestHeaders:
    """Tests for request/response headers."""

    def test_request_id_header(self, client: FlaskClient) -> None:
        """Test that X-Request-ID header is returned."""
        response = client.get("/health")
        assert "X-Request-ID" in response.headers


class TestErrorHandling:
    """Tests for error handling."""

    def test_model_not_ready(self, app: Flask) -> None:
        """Test error when model is not ready."""
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

    def test_unexpected_error(self, app: Flask) -> None:
        """Test handling of unexpected errors."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, encoded = cv2.imencode(".jpg", img)

        # Make model.get raise an unexpected error
        app.config["model_manager"].model = MagicMock()
        app.config["model_manager"].model.get.side_effect = RuntimeError("Unexpected")

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

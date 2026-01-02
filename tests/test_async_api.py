"""
Tests for async API endpoints (Quart version).

These tests verify that the async routes work correctly with the
Quart application and InferenceExecutor thread pool.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from werkzeug.datastructures import FileStorage

# Mark all tests in this module as async
pytestmark = pytest.mark.asyncio


class TestAsyncIndexEndpoint:
    """Tests for GET / endpoint."""

    async def test_index_returns_api_info(self, async_client):
        """Test that index returns API information."""
        response = await async_client.get("/")
        assert response.status_code == 200

        data = await response.get_json()
        assert data["name"] == "InsightFace API"
        assert data["version"] == "2.0.0"
        assert "/represent" in data["endpoints"]
        assert "/health" in data["endpoints"]


class TestAsyncHealthEndpoint:
    """Tests for GET /health endpoint."""

    async def test_health_returns_status(self, async_client):
        """Test that health endpoint returns model status."""
        response = await async_client.get("/health")
        assert response.status_code == 200

        data = await response.get_json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert "uptime_seconds" in data

    async def test_health_returns_503_when_model_not_loaded(self, async_app):
        """Test health returns 503 when model isn't loaded."""
        async_app.config["model_manager"].is_loaded = False

        async with async_app.test_client() as client:
            response = await client.get("/health")
            assert response.status_code == 503

            data = await response.get_json()
            assert data["status"] in ["unhealthy", "degraded"]


class TestAsyncUpEndpoint:
    """Tests for GET /up endpoint."""

    async def test_up_returns_ok(self, async_client):
        """Test simple health check returns ok."""
        response = await async_client.get("/up")
        assert response.status_code == 200

        data = await response.get_json()
        assert data["status"] == "ok"


class TestAsyncRepresentEndpoint:
    """Tests for POST /represent endpoint."""

    async def test_represent_no_file_returns_400(self, async_client):
        """Test that missing file returns 400."""
        response = await async_client.post("/represent")
        assert response.status_code == 400

        data = await response.get_json()
        assert "error" in data

    async def test_represent_empty_file_returns_400(self, async_client):
        """Test that empty file returns 400."""
        # Quart test client requires FileStorage objects
        file_storage = FileStorage(
            stream=io.BytesIO(b""),
            filename="empty.jpg",
            content_type="image/jpeg",
        )
        response = await async_client.post(
            "/represent",
            files={"image_file": file_storage},
        )
        assert response.status_code == 400

    async def test_represent_invalid_image_returns_400(self, async_client):
        """Test that invalid image data returns 400."""
        file_storage = FileStorage(
            stream=io.BytesIO(b"not an image"),
            filename="test.jpg",
            content_type="image/jpeg",
        )
        response = await async_client.post(
            "/represent",
            files={"image_file": file_storage},
        )
        assert response.status_code == 400

    async def test_represent_model_not_ready_returns_503(self, async_app):
        """Test that request fails when model isn't ready."""
        async_app.config["model_manager"].is_loaded = False

        # Create a minimal valid JPEG
        jpeg_header = b"\xff\xd8\xff\xe0\x00\x10JFIF"
        file_storage = FileStorage(
            stream=io.BytesIO(jpeg_header),
            filename="test.jpg",
            content_type="image/jpeg",
        )

        async with async_app.test_client() as client:
            response = await client.post(
                "/represent",
                files={"image_file": file_storage},
            )
            assert response.status_code == 503

    async def test_represent_with_valid_image(self, async_app, mock_face):
        """Test successful face detection with valid image."""
        # Create a valid PNG image
        png_header = b"\x89PNG\r\n\x1a\n"
        png_data = png_header + b"\x00" * 100
        file_storage = FileStorage(
            stream=io.BytesIO(png_data),
            filename="test.png",
            content_type="image/png",
        )

        # Mock the model manager to return faces
        async_app.config["model_manager"].model = MagicMock()
        async_app.config["model_manager"].model.get.return_value = [mock_face]

        # Mock decode_image to return a valid array
        with patch("src.api.routes_async.decode_image") as mock_decode:
            mock_decode.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

            async with async_app.test_client() as client:
                response = await client.post(
                    "/represent",
                    files={"image_file": file_storage},
                )

                assert response.status_code == 200
                data = await response.get_json()
                assert "embeddings" in data
                assert "faces_detected" in data
                assert "processing_time_ms" in data
                assert "request_id" in data


class TestAsyncRequestId:
    """Tests for request ID handling in async routes."""

    async def test_request_id_in_response_header(self, async_client):
        """Test that X-Request-ID header is present in response."""
        response = await async_client.get("/")
        assert "X-Request-ID" in response.headers
        assert len(response.headers["X-Request-ID"]) == 8

"""
Tests for async API endpoints (Quart version).

These tests verify that the async routes work correctly with the
Quart application and InferenceExecutor thread pool.
"""

from __future__ import annotations

import io
from unittest.mock import AsyncMock, MagicMock, patch

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

    async def test_represent_with_valid_image(self, async_app, mock_face, png_bytes):
        """Test successful face detection with valid image."""
        file_storage = FileStorage(
            stream=io.BytesIO(png_bytes),
            filename="test.png",
            content_type="image/png",
        )

        # Mock the model manager to return faces
        async_app.config["model_manager"].model = MagicMock()
        async_app.config["model_manager"].model.get.return_value = [mock_face]

        async with async_app.test_client() as client:
            response = await client.post(
                "/represent",
                files={"image_file": file_storage},
            )

            assert response.status_code == 200
            data = await response.get_json()
            assert data["faces_detected"] == 1
            assert len(data["embeddings"]) == 1
            assert len(data["embeddings"][0]["embedding"]) == 512
            assert "processing_time_ms" in data
            assert "request_id" in data

    async def test_represent_rejects_file_and_url_together(
        self, async_app, png_bytes
    ):
        """Test that sending both inputs is a 400."""
        file_storage = FileStorage(
            stream=io.BytesIO(png_bytes),
            filename="test.png",
            content_type="image/png",
        )

        async with async_app.test_client() as client:
            response = await client.post(
                "/represent",
                files={"image_file": file_storage},
                form={"image_url": "https://example.com/a.jpg"},
            )
            assert response.status_code == 400
            data = await response.get_json()
            assert "not both" in data["error"]

    async def test_represent_rejects_empty_json_body(self, async_app):
        """Test that a JSON body with no image_url is a 400."""
        async with async_app.test_client() as client:
            response = await client.post("/represent", json={})
            assert response.status_code == 400

    async def test_represent_accepts_image_url(
        self, settings, mock_face, png_bytes
    ):
        """Test that an allowlisted image_url is downloaded and processed."""
        from src.app_async import create_async_app
        from src.services import ModelManager

        settings.image_url_allowed_hosts = "*.example.com"

        with patch.object(ModelManager, "load", return_value=True):
            app = create_async_app(settings)

        app.config["model_manager"].is_loaded = True
        app.config["model_manager"].model = MagicMock()
        app.config["model_manager"].model.get.return_value = [mock_face]

        # test_app() runs the before_serving hooks that open the HTTP client.
        with patch(
            "src.api.routes_async.fetch_image",
            new=AsyncMock(return_value=png_bytes),
        ) as mock_fetch:
            async with app.test_app() as test_app:
                response = await test_app.test_client().post(
                    "/represent",
                    json={"image_url": "https://cdn.example.com/a.png"},
                )

        assert response.status_code == 200
        data = await response.get_json()
        assert data["faces_detected"] == 1
        assert mock_fetch.await_count == 1

    async def test_represent_returns_503_when_overloaded(
        self, async_app, png_bytes
    ):
        """Test that a full inference queue returns 503 with Retry-After."""
        from src.exceptions import ServiceOverloadedError

        file_storage = FileStorage(
            stream=io.BytesIO(png_bytes),
            filename="test.png",
            content_type="image/png",
        )

        executor = async_app.config["inference_executor"]
        with patch.object(
            executor, "run", side_effect=ServiceOverloadedError("full")
        ):
            async with async_app.test_client() as client:
                response = await client.post(
                    "/represent",
                    files={"image_file": file_storage},
                )

        assert response.status_code == 503
        assert response.headers["Retry-After"] == "1"
        data = await response.get_json()
        assert data["error_code"] == "OVERLOADED"


class TestAsyncRequestId:
    """Tests for request ID handling in async routes."""

    async def test_request_id_in_response_header(self, async_client):
        """Test that X-Request-ID header is present in response."""
        response = await async_client.get("/")
        assert "X-Request-ID" in response.headers
        assert len(response.headers["X-Request-ID"]) == 8

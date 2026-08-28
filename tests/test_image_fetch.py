"""
Tests for the image_url input.

The URL is caller-controlled, so the checks that matter are the ones that stop
the API from being used as a proxy into a private network.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

import httpx
import pytest

from src.config import Settings
from src.exceptions import ImageDownloadError, RequestValidationError
from src.services import fetch_image, validate_image_url

ALLOWED_URL = "https://bucket.r2.cloudflarestorage.com/photo.jpg"


@pytest.fixture
def url_settings() -> Settings:
    """Settings with URL input enabled for one host pattern."""
    return Settings(
        image_url_allowed_hosts="*.r2.cloudflarestorage.com",
        max_content_length=1024,
        log_level="WARNING",
    )


def make_client(handler) -> httpx.AsyncClient:
    """Build a client whose requests are answered by `handler`."""
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class TestValidateImageUrl:
    """Tests for URL allowlisting."""

    def test_disabled_when_allowlist_is_empty(self) -> None:
        """Test that URL input is off unless hosts are configured."""
        settings = Settings(log_level="WARNING")
        with pytest.raises(RequestValidationError) as exc_info:
            validate_image_url(ALLOWED_URL, settings)
        assert "disabled" in str(exc_info.value)

    def test_rejects_http(self, url_settings: Settings) -> None:
        """Test that plain HTTP is rejected."""
        with pytest.raises(RequestValidationError) as exc_info:
            validate_image_url(
                "http://bucket.r2.cloudflarestorage.com/a.jpg", url_settings
            )
        assert "https" in str(exc_info.value)

    def test_rejects_other_schemes(self, url_settings: Settings) -> None:
        """Test that file:// and friends are rejected."""
        with pytest.raises(RequestValidationError):
            validate_image_url("file:///etc/passwd", url_settings)

    @pytest.mark.parametrize(
        "host", ["169.254.169.254", "127.0.0.1", "10.0.0.5", "[::1]"]
    )
    def test_rejects_ip_literals(self, host: str, url_settings: Settings) -> None:
        """Test that IP hosts cannot bypass the allowlist."""
        with pytest.raises(RequestValidationError) as exc_info:
            validate_image_url(f"https://{host}/a.jpg", url_settings)
        assert "IP address" in str(exc_info.value)

    def test_rejects_host_outside_allowlist(self, url_settings: Settings) -> None:
        """Test that an unlisted host is rejected."""
        with pytest.raises(RequestValidationError) as exc_info:
            validate_image_url("https://evil.example.com/a.jpg", url_settings)
        assert "not allowed" in str(exc_info.value)

    def test_accepts_allowlisted_host(self, url_settings: Settings) -> None:
        """Test that a matching host passes."""
        assert (
            validate_image_url(ALLOWED_URL, url_settings)
            == "bucket.r2.cloudflarestorage.com"
        )


class TestFetchImage:
    """Tests for the download itself."""

    @pytest.fixture
    def semaphore(self) -> asyncio.Semaphore:
        """Download concurrency limiter."""
        return asyncio.Semaphore(4)

    @pytest.mark.asyncio
    async def test_returns_body(self, url_settings, semaphore) -> None:
        """Test that a normal response is returned as bytes."""
        payload = b"\xff\xd8\xff" + b"x" * 100

        async with make_client(
            lambda request: httpx.Response(200, content=payload)
        ) as client:
            data = await fetch_image(ALLOWED_URL, client, semaphore, url_settings)

        assert data == payload

    @pytest.mark.asyncio
    async def test_rejects_non_200(self, url_settings, semaphore) -> None:
        """Test that an error response becomes a 502."""
        async with make_client(
            lambda request: httpx.Response(404, content=b"nope")
        ) as client:
            with pytest.raises(ImageDownloadError) as exc_info:
                await fetch_image(ALLOWED_URL, client, semaphore, url_settings)

        assert exc_info.value.status_code == 502
        assert "404" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_rejects_declared_oversize(self, url_settings, semaphore) -> None:
        """Test that a too-large Content-Length is refused."""
        async with make_client(
            lambda request: httpx.Response(200, content=b"x" * 4096)
        ) as client:
            with pytest.raises(ImageDownloadError) as exc_info:
                await fetch_image(ALLOWED_URL, client, semaphore, url_settings)

        assert "too large" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_rejects_streamed_oversize(self, url_settings, semaphore) -> None:
        """Test that a chunked body over the limit is refused mid-stream."""

        async def chunks() -> AsyncIterator[bytes]:
            for _ in range(10):
                yield b"x" * 512

        async with make_client(
            lambda request: httpx.Response(200, content=chunks())
        ) as client:
            with pytest.raises(ImageDownloadError) as exc_info:
                await fetch_image(ALLOWED_URL, client, semaphore, url_settings)

        assert "too large" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_rejects_empty_body(self, url_settings, semaphore) -> None:
        """Test that an empty response is refused."""
        async with make_client(
            lambda request: httpx.Response(200, content=b"")
        ) as client:
            with pytest.raises(ImageDownloadError) as exc_info:
                await fetch_image(ALLOWED_URL, client, semaphore, url_settings)

        assert "empty" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_network_error_becomes_502(self, url_settings, semaphore) -> None:
        """Test that a connection failure becomes a 502."""

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("no route to host", request=request)

        async with make_client(handler) as client:
            with pytest.raises(ImageDownloadError) as exc_info:
                await fetch_image(ALLOWED_URL, client, semaphore, url_settings)

        assert exc_info.value.status_code == 502

    @pytest.mark.asyncio
    async def test_does_not_fetch_disallowed_host(
        self, url_settings, semaphore
    ) -> None:
        """Test that a rejected URL never reaches the network."""
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request.url)
            return httpx.Response(200, content=b"x")

        async with make_client(handler) as client:
            with pytest.raises(RequestValidationError):
                await fetch_image(
                    "https://evil.example.com/a.jpg", client, semaphore, url_settings
                )

        assert calls == []

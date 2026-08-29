"""
Tests for the image_url input.

The URL is caller-controlled, so the checks that matter are the ones that stop
the API from being used as a proxy into a private network.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from typing import AsyncIterator
from unittest.mock import patch

import httpx
import pytest

from src.config import Settings
from src.exceptions import ImageDownloadError, RequestValidationError
from src.services import (
    fetch_image,
    is_public_address,
    resolve_public_addresses,
    validate_image_url,
)

ALLOWED_URL = "https://bucket.r2.cloudflarestorage.com/photo.jpg"
PUBLIC_IP = "93.184.216.34"


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


def resolves_to(*literals: str):
    """Patch name resolution so a host answers with the given addresses."""

    async def fake_getaddrinfo(self, host, port, **kwargs):
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (literal, port))
            for literal in literals
        ]

    return patch(
        "asyncio.selector_events.BaseSelectorEventLoop.getaddrinfo",
        new=fake_getaddrinfo,
    )


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
        """Test that a matching host passes and its parts come back."""
        assert validate_image_url(ALLOWED_URL, url_settings) == (
            "bucket.r2.cloudflarestorage.com",
            443,
            "/photo.jpg",
        )

    def test_keeps_the_query_string(self, url_settings: Settings) -> None:
        """Test that a presigned URL keeps its query string."""
        host, port, path = validate_image_url(
            "https://bucket.r2.cloudflarestorage.com/a.jpg?sig=abc&x=1", url_settings
        )
        assert path == "/a.jpg?sig=abc&x=1"

    def test_keeps_an_explicit_port(self, url_settings: Settings) -> None:
        """Test that a non-default port is carried through."""
        host, port, path = validate_image_url(
            "https://bucket.r2.cloudflarestorage.com:8443/a.jpg", url_settings
        )
        assert port == 8443


class TestPublicAddressCheck:
    """Tests for address classification."""

    @pytest.mark.parametrize(
        "literal",
        [
            "127.0.0.1",
            "10.0.0.5",
            "172.16.0.1",
            "192.168.1.1",
            "169.254.169.254",  # cloud metadata
            "100.64.0.1",  # carrier-grade NAT
            "0.0.0.0",
            "224.0.0.1",
            "::1",
            "fd00::1",
            "fe80::1",
            "::ffff:127.0.0.1",  # IPv4-mapped loopback
            "::ffff:10.0.0.1",
        ],
    )
    def test_rejects_non_public(self, literal: str) -> None:
        """Test that every non-routable address is refused."""
        assert is_public_address(ipaddress.ip_address(literal)) is False

    @pytest.mark.parametrize(
        "literal", ["8.8.8.8", PUBLIC_IP, "2606:4700:4700::1111", "::ffff:8.8.8.8"]
    )
    def test_accepts_public(self, literal: str) -> None:
        """Test that public addresses pass."""
        assert is_public_address(ipaddress.ip_address(literal)) is True


class TestResolvePublicAddresses:
    """Tests for the DNS step."""

    @pytest.mark.asyncio
    async def test_rejects_localhost(self, url_settings: Settings) -> None:
        """Test the real resolver: localhost is not a public address."""
        with pytest.raises(RequestValidationError) as exc_info:
            await resolve_public_addresses("localhost", 443)
        assert "non-public address" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_rejects_metadata_address(self) -> None:
        """Test that a name pointing at cloud metadata is refused."""
        with resolves_to("169.254.169.254"):
            with pytest.raises(RequestValidationError) as exc_info:
                await resolve_public_addresses("metadata.example.com", 443)
        assert "169.254.169.254" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_rejects_when_any_address_is_private(self) -> None:
        """Test that one bad address in the answer spoils the whole set."""
        with resolves_to(PUBLIC_IP, "10.0.0.5"):
            with pytest.raises(RequestValidationError):
                await resolve_public_addresses("split.example.com", 443)

    @pytest.mark.asyncio
    async def test_returns_public_addresses(self) -> None:
        """Test that an all-public answer is returned."""
        with resolves_to(PUBLIC_IP, "8.8.8.8"):
            addresses = await resolve_public_addresses("cdn.example.com", 443)
        assert [str(a) for a in addresses] == [PUBLIC_IP, "8.8.8.8"]

    @pytest.mark.asyncio
    async def test_resolution_failure_becomes_502(self) -> None:
        """Test that an unresolvable name is a download error, not a crash."""

        async def boom(self, host, port, **kwargs):
            raise socket.gaierror("Name or service not known")

        with patch(
            "asyncio.selector_events.BaseSelectorEventLoop.getaddrinfo", new=boom
        ):
            with pytest.raises(ImageDownloadError) as exc_info:
                await resolve_public_addresses("nope.example.com", 443)
        assert exc_info.value.status_code == 502


class TestFetchImage:
    """Tests for the download itself."""

    @pytest.fixture
    def semaphore(self) -> asyncio.Semaphore:
        """Download concurrency limiter."""
        return asyncio.Semaphore(4)

    @pytest.fixture(autouse=True)
    def public_dns(self):
        """Resolve every host in this class to one public address."""
        with resolves_to(PUBLIC_IP):
            yield

    @pytest.mark.asyncio
    async def test_returns_body(self, url_settings, semaphore) -> None:
        """Test that a normal response is returned as bytes."""
        payload = b"\xff\xd8\xff" + b"x" * 100
        stats = {}

        async with make_client(
            lambda request: httpx.Response(200, content=payload)
        ) as client:
            data = await fetch_image(
                ALLOWED_URL, client, semaphore, url_settings, stats=stats
            )

        assert data == payload
        assert stats["dns_ms"] >= 0.0
        assert stats["semaphore_wait_ms"] >= 0.0
        assert stats["transfer_ms"] >= 0.0

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

    @pytest.mark.asyncio
    async def test_connects_to_the_checked_address(
        self, url_settings, semaphore
    ) -> None:
        """Test that the connection is pinned to the address that was checked.

        Resolving once and connecting by name would leave a window for a
        second DNS answer to point at a private address.
        """
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["host"] = request.url.host
            seen["header"] = request.headers["Host"]
            seen["path"] = request.url.raw_path.decode()
            seen["sni"] = request.extensions.get("sni_hostname")
            return httpx.Response(200, content=b"\xff\xd8\xff" + b"x" * 10)

        async with make_client(handler) as client:
            await fetch_image(
                "https://bucket.r2.cloudflarestorage.com/a.jpg?sig=abc",
                client,
                semaphore,
                url_settings,
            )

        # Connect to the address, but present the name to the server and to TLS
        assert seen["host"] == PUBLIC_IP
        assert seen["header"] == "bucket.r2.cloudflarestorage.com"
        assert seen["sni"] == "bucket.r2.cloudflarestorage.com"
        assert seen["path"] == "/a.jpg?sig=abc"


class TestSsrfRegressions:
    """The holes an allowlist alone does not close."""

    @pytest.fixture
    def semaphore(self) -> asyncio.Semaphore:
        """Download concurrency limiter."""
        return asyncio.Semaphore(4)

    @pytest.fixture
    def wildcard_settings(self) -> Settings:
        """The worst allowlist a user could configure."""
        return Settings(image_url_allowed_hosts="*", log_level="WARNING")

    @pytest.mark.asyncio
    async def test_wildcard_allowlist_still_refuses_localhost(
        self, wildcard_settings, semaphore
    ) -> None:
        """Test that `*` does not open a path to the loopback interface."""
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request.url)
            return httpx.Response(200, content=b"secret")

        async with make_client(handler) as client:
            with pytest.raises(RequestValidationError) as exc_info:
                await fetch_image(
                    "https://localhost/admin", client, semaphore, wildcard_settings
                )

        assert "non-public address" in str(exc_info.value)
        assert calls == []

    @pytest.mark.asyncio
    async def test_allowlisted_name_pointing_at_metadata_is_refused(
        self, wildcard_settings, semaphore
    ) -> None:
        """Test that an allowlisted name cannot reach the metadata service."""
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request.url)
            return httpx.Response(200, content=b"credentials")

        with resolves_to("169.254.169.254"):
            async with make_client(handler) as client:
                with pytest.raises(RequestValidationError):
                    await fetch_image(
                        "https://metadata.example.com/latest/meta-data/iam/",
                        client,
                        semaphore,
                        wildcard_settings,
                    )

        assert calls == []

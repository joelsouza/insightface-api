"""
Fetch images referenced by URL.

Batch jobs read their images from object storage. Letting the API download the
image itself removes one full upload hop per request. The download is I/O, so
it runs on the event loop and never touches the inference thread pool.

The URL is attacker-controlled input, so it is checked against an explicit
allowlist of hosts before any connection is made.
"""

from __future__ import annotations

import asyncio
import fnmatch
import ipaddress
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

import httpx

from src.exceptions import ImageDownloadError, RequestValidationError

if TYPE_CHECKING:
    from src.config import Settings


def validate_image_url(url: str, settings: Settings) -> str:
    """
    Check that a URL is safe to fetch and return its host.

    Only HTTPS URLs whose host matches `image_url_allowed_hosts` are accepted.
    IP literals are rejected: they bypass the host allowlist and are the usual
    route to internal metadata services.

    Args:
        url: The requested image URL
        settings: Application settings holding the allowlist

    Returns:
        The lowercase host of the URL

    Raises:
        RequestValidationError: If URL input is disabled or the URL is not allowed

    Example:
        >>> validate_image_url("https://bucket.r2.cloudflarestorage.com/a.jpg", settings)
        'bucket.r2.cloudflarestorage.com'
    """
    allowed = settings.allowed_image_url_hosts
    if not allowed:
        raise RequestValidationError(
            "image_url input is disabled. Set IMAGE_URL_ALLOWED_HOSTS to enable it."
        )

    parts = urlsplit(url)

    if parts.scheme != "https":
        raise RequestValidationError("image_url must use https")

    host = (parts.hostname or "").lower()
    if not host:
        raise RequestValidationError("image_url has no host")

    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise RequestValidationError(
            "image_url must use a host name, not an IP address"
        )

    if not any(fnmatch.fnmatch(host, pattern) for pattern in allowed):
        raise RequestValidationError(f"image_url host is not allowed: {host}")

    return host


async def fetch_image(
    url: str,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    settings: Settings,
) -> bytes:
    """
    Download an image over HTTPS, bounded in size and concurrency.

    The body is streamed so an oversized response is dropped as soon as the
    limit is passed, instead of being buffered in full.

    Args:
        url: Image URL (validated before the request is made)
        client: Shared HTTP client
        semaphore: Limits parallel downloads
        settings: Application settings

    Returns:
        Raw image bytes

    Raises:
        RequestValidationError: If the URL is not allowed
        ImageDownloadError: If the download fails or the body is too large

    Example:
        >>> data = await fetch_image(url, client, semaphore, settings)
    """
    validate_image_url(url, settings)

    max_bytes = settings.max_content_length
    max_mb = max_bytes / (1024 * 1024)

    async with semaphore:
        try:
            async with client.stream("GET", url) as response:
                if response.status_code != 200:
                    raise ImageDownloadError(
                        f"Image download returned HTTP {response.status_code}"
                    )

                declared = response.headers.get("content-length")
                if declared is not None and declared.isdigit():
                    if int(declared) > max_bytes:
                        raise ImageDownloadError(
                            f"Image too large. Maximum size is {max_mb:.1f}MB"
                        )

                chunks: list[bytes] = []
                total = 0
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > max_bytes:
                        raise ImageDownloadError(
                            f"Image too large. Maximum size is {max_mb:.1f}MB"
                        )
                    chunks.append(chunk)

        except httpx.HTTPError as e:
            raise ImageDownloadError(f"Image download failed: {e}")

    data = b"".join(chunks)
    if not data:
        raise ImageDownloadError("Image download returned an empty body")

    return data

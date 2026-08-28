"""
Fetch images referenced by URL.

Batch jobs read their images from object storage. Letting the API download the
image itself removes one full upload hop per request. The download is I/O, so
it runs on the event loop and never touches the inference thread pool.

The URL is caller-controlled input, so this module is the trust boundary
against server-side request forgery. Three checks work together, and all three
are needed:

1. The URL must be HTTPS and its host must match the configured allowlist.
2. Every address the host resolves to must be a public address. Checking the
   host text alone is not enough: a name in the allowlist can point at
   loopback, a private range, or a cloud metadata service.
3. The connection is pinned to the address that was checked, so a second DNS
   answer cannot swap in a private address after the check (DNS rebinding).

Redirects are not followed, because a redirect target would skip all of this.
"""

from __future__ import annotations

import asyncio
import fnmatch
import ipaddress
import socket
from dataclasses import dataclass
from typing import TYPE_CHECKING, Union
from urllib.parse import urlsplit, urlunsplit

import httpx

from src.exceptions import ImageDownloadError, RequestValidationError

if TYPE_CHECKING:
    from src.config import Settings

IPAddress = Union[ipaddress.IPv4Address, ipaddress.IPv6Address]

DEFAULT_HTTPS_PORT = 443


@dataclass(frozen=True)
class _Target:
    """
    A download target with its address already resolved and checked.

    Attributes:
        url: The URL with the host replaced by the checked address
        host_header: The original host (and port), for the `Host` header
        sni_hostname: The original host, for TLS server name and certificate
            verification
    """

    url: str
    host_header: str
    sni_hostname: str


def _unwrap(address: IPAddress) -> IPAddress:
    """
    Return the IPv4 address behind an IPv4-mapped IPv6 address.

    `::ffff:127.0.0.1` is loopback, and the mapping must be removed before the
    address is classified.

    Args:
        address: Parsed IP address

    Returns:
        The mapped IPv4 address, or the input unchanged
    """
    mapped = getattr(address, "ipv4_mapped", None)
    return mapped if mapped is not None else address


def is_public_address(address: IPAddress) -> bool:
    """
    Report whether an address is routable on the public internet.

    `is_global` already covers every case here. The rest is stated explicitly
    so the intent survives, and so the check does not depend on one property
    behaving identically across Python versions.

    Args:
        address: Parsed IP address

    Returns:
        True if the address is safe to connect to

    Example:
        >>> is_public_address(ipaddress.ip_address("8.8.8.8"))
        True
        >>> is_public_address(ipaddress.ip_address("169.254.169.254"))
        False
    """
    address = _unwrap(address)
    return address.is_global and not (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )


def validate_image_url(url: str, settings: Settings) -> tuple[str, int, str]:
    """
    Check the shape of a URL, before any name is resolved.

    Only HTTPS URLs whose host matches `image_url_allowed_hosts` are accepted.
    IP-literal hosts are refused outright: an allowlist is written in terms of
    names, so a literal has no business matching one.

    This check alone is **not** sufficient. The host must still be resolved and
    every one of its addresses checked; see `resolve_public_addresses`.

    Args:
        url: The requested image URL
        settings: Application settings holding the allowlist

    Returns:
        (host, port, path with query) of the URL

    Raises:
        RequestValidationError: If URL input is disabled or the URL is not allowed

    Example:
        >>> validate_image_url("https://bucket.r2.cloudflarestorage.com/a.jpg", settings)
        ('bucket.r2.cloudflarestorage.com', 443, '/a.jpg')
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

    try:
        port = parts.port or DEFAULT_HTTPS_PORT
    except ValueError:
        raise RequestValidationError("image_url has an invalid port")

    path = parts.path or "/"
    return host, port, urlunsplit(("", "", path, parts.query, ""))


async def resolve_public_addresses(host: str, port: int) -> list[IPAddress]:
    """
    Resolve a host and refuse it unless every address is public.

    Every A and AAAA record is checked, not just the first. A host that answers
    with one public address and one private address is refused: which one a
    later connection would use is not something this code can control.

    Args:
        host: Host name to resolve
        port: Port, passed through to `getaddrinfo`

    Returns:
        Every resolved address, all of them public

    Raises:
        RequestValidationError: If any address is not public
        ImageDownloadError: If the name cannot be resolved

    Example:
        >>> await resolve_public_addresses("example.com", 443)
        [IPv4Address('93.184.216.34')]
    """
    loop = asyncio.get_running_loop()

    try:
        infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise ImageDownloadError(f"Could not resolve image_url host: {host} ({e})")

    addresses = []
    seen = set()
    for info in infos:
        literal = info[4][0]
        if literal in seen:
            continue
        seen.add(literal)
        addresses.append(ipaddress.ip_address(literal))

    if not addresses:
        raise ImageDownloadError(f"Could not resolve image_url host: {host}")

    for address in addresses:
        if not is_public_address(address):
            raise RequestValidationError(
                f"image_url host {host} resolves to a non-public address: {address}"
            )

    return addresses


async def build_target(url: str, settings: Settings) -> _Target:
    """
    Validate a URL and pin it to a checked address.

    The returned URL points at the address, not the name, so the connection
    cannot be redirected to a different address by a second DNS answer. The
    original host still travels in the `Host` header and in the TLS server
    name, so the request works and the certificate is verified against the
    name the caller asked for.

    Args:
        url: The requested image URL
        settings: Application settings

    Returns:
        A target whose address has already been checked

    Raises:
        RequestValidationError: If the URL or any of its addresses is refused
        ImageDownloadError: If the name cannot be resolved
    """
    host, port, path = validate_image_url(url, settings)
    addresses = await resolve_public_addresses(host, port)

    address = addresses[0]
    literal = f"[{address}]" if address.version == 6 else str(address)

    host_header = host if port == DEFAULT_HTTPS_PORT else f"{host}:{port}"

    return _Target(
        url=urlunsplit(("https", f"{literal}:{port}", "", "", "")).rstrip("/") + path,
        host_header=host_header,
        sni_hostname=host,
    )


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
        url: Image URL (checked and pinned before the request is made)
        client: Shared HTTP client
        semaphore: Limits parallel downloads
        settings: Application settings

    Returns:
        Raw image bytes

    Raises:
        RequestValidationError: If the URL or its addresses are refused
        ImageDownloadError: If the download fails or the body is too large

    Example:
        >>> data = await fetch_image(url, client, semaphore, settings)
    """
    target = await build_target(url, settings)

    max_bytes = settings.max_content_length
    max_mb = max_bytes / (1024 * 1024)

    async with semaphore:
        try:
            async with client.stream(
                "GET",
                target.url,
                headers={"Host": target.host_header},
                extensions={"sni_hostname": target.sni_hostname},
            ) as response:
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

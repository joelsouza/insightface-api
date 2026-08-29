"""
Single-job image pipeline for the async API.

Everything CPU-bound for one request happens here: decode, validation,
inference, and result extraction. The async route hands this function to the
inference thread pool as one job, so the event loop stays free to accept
connections and answer health checks while a 12 MP JPEG is being decoded.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np
from PIL import Image

from src.exceptions import ImageDecodeError
from src.models import FaceEmbedding
from src.services.image import (
    extract_face_data,
    validate_image_dimensions,
    verify_magic_bytes,
)

if TYPE_CHECKING:
    from src.config import Settings
    from src.services.model_manager import ModelManager


@dataclass
class PipelineResult:
    """
    Outcome of processing one image.

    Attributes:
        faces: Extracted face data, ready for serialization
        width: Original image width in pixels
        height: Original image height in pixels
        decode_ms: Time spent decoding the image
        detect_ms: Time spent in face detection
        align_ms: Time spent aligning face crops
        embed_ms: Time spent extracting face embeddings
        extract_ms: Time spent converting faces to response data
        downscaled: Whether the image was decoded at half resolution
    """

    faces: list[FaceEmbedding]
    width: int
    height: int
    decode_ms: float
    detect_ms: float
    align_ms: float
    embed_ms: float
    extract_ms: float
    downscaled: bool


def _read_header_size(file_bytes: bytes) -> Optional[tuple[int, int]]:
    """
    Read image dimensions from the file header without decoding pixels.

    `PIL.Image.open` is lazy: it parses the header and stops.

    Args:
        file_bytes: Raw image bytes

    Returns:
        (width, height), or None if the header could not be read
    """
    try:
        with Image.open(io.BytesIO(file_bytes)) as img:
            return img.size
    except Exception:
        return None


def _decode(
    file_bytes: bytes, image_format: str, settings: Settings
) -> tuple[np.ndarray, int, Optional[tuple[int, int]]]:
    """
    Decode image bytes, halving the resolution when the image is far too big.

    Detection resizes the image to `max_image_dimension` anyway. Decoding a
    4000x3000 JPEG at full size costs ~36 MB and ~100 ms for pixels that are
    thrown away, so JPEGs at 4x the detection size or larger are decoded at
    half resolution by the JPEG decoder itself.

    Args:
        file_bytes: Raw image bytes
        image_format: Format name from `verify_magic_bytes`
        settings: Application settings

    Returns:
        (BGR image, scale factor back to original pixels, header size or None)

    Raises:
        ImageDecodeError: If decoding fails or the image is not 3-channel
    """
    header_size = _read_header_size(file_bytes)

    scale = 1
    flags = cv2.IMREAD_COLOR
    if image_format == "jpeg" and header_size is not None:
        if max(header_size) >= 4 * settings.max_image_dimension:
            scale = 2
            flags = cv2.IMREAD_REDUCED_COLOR_2

    nparr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(nparr, flags)

    if image is None:
        raise ImageDecodeError("Failed to decode image. File may be corrupted.")

    # Ensure image has 3 color channels (BGR)
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ImageDecodeError("Image must be a color image with 3 channels (RGB/BGR)")

    return image, scale, header_size


def process_image(
    file_bytes: bytes, mm: ModelManager, settings: Settings
) -> PipelineResult:
    """
    Decode, validate, and run inference on one image.

    Runs entirely in a worker thread. Face coordinates are always returned in
    original image pixels, even when the image was decoded at half resolution.

    Args:
        file_bytes: Raw image bytes
        mm: Model manager holding the loaded face engine
        settings: Application settings

    Returns:
        PipelineResult with extracted faces, image size, and timings

    Raises:
        ImageValidationError: If the format is unsupported or the image is tiny
        ImageDecodeError: If decoding fails
        ModelNotReadyError: If the model is not loaded

    Example:
        >>> result = process_image(jpeg_bytes, model_manager, settings)
        >>> print(len(result.faces), result.width, result.height)
    """
    decode_start = time.perf_counter()

    image_format = verify_magic_bytes(file_bytes)
    image, scale, header_size = _decode(file_bytes, image_format, settings)

    decode_ms = (time.perf_counter() - decode_start) * 1000

    validate_image_dimensions(image, settings.max_image_dimension)

    timings: dict[str, float] = {}
    detect_start = time.perf_counter()
    faces = mm.get_faces(image, timings=timings)
    inference_ms = (time.perf_counter() - detect_start) * 1000

    if scale != 1:
        for face in faces:
            face.bbox = face.bbox * scale
            if getattr(face, "kps", None) is not None:
                face.kps = face.kps * scale

    if header_size is not None:
        width, height = header_size
    else:
        height, width = image.shape[0] * scale, image.shape[1] * scale

    extract_start = time.perf_counter()
    extracted_faces = [
        extract_face_data(f, settings.embedding_decimals) for f in faces
    ]
    extract_ms = (time.perf_counter() - extract_start) * 1000

    return PipelineResult(
        faces=extracted_faces,
        width=width,
        height=height,
        decode_ms=decode_ms,
        detect_ms=timings.get("detect_ms", inference_ms),
        align_ms=timings.get("align_ms", 0.0),
        embed_ms=timings.get("embed_ms", 0.0),
        extract_ms=extract_ms,
        downscaled=scale != 1,
    )

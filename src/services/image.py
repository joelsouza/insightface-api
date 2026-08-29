"""
Image processing utilities for the InsightFace API.

This module provides functions for validating, decoding, and processing
uploaded images before face detection.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
from pydantic import ValidationError
from werkzeug.datastructures import FileStorage

from src.exceptions import ImageDecodeError, ImageValidationError, RequestValidationError
from src.models import FaceEmbedding, ImageInput

# Magic bytes for supported image formats
IMAGE_MAGIC_BYTES = {
    b"\xff\xd8\xff": "jpeg",
    b"\x89PNG\r\n\x1a\n": "png",
    b"GIF87a": "gif",
    b"GIF89a": "gif",
    b"RIFF": "webp",  # WebP starts with RIFF
    b"BM": "bmp",
}


def verify_magic_bytes(file_bytes: bytes) -> str:
    """
    Verify that file bytes match a supported image format.

    Checks the file's magic bytes (file signature) against known image formats
    to prevent processing of non-image files.

    Args:
        file_bytes: Raw file bytes to verify

    Returns:
        Detected image format name (e.g., "jpeg", "png")

    Raises:
        ImageValidationError: If file doesn't match any supported image format

    Example:
        >>> with open("photo.jpg", "rb") as f:
        ...     format_name = verify_magic_bytes(f.read())
        >>> print(f"Detected format: {format_name}")
    """
    for magic, format_name in IMAGE_MAGIC_BYTES.items():
        if file_bytes.startswith(magic):
            return format_name

    raise ImageValidationError(
        "Invalid image format. Supported formats: JPEG, PNG, GIF, WebP, BMP"
    )


def validate_image_file(file: FileStorage, max_size: int) -> ImageInput:
    """
    Validate an uploaded image file before processing.

    Performs the following validations:
    - File exists and has a filename
    - File is not empty
    - File size is within limits
    - Content type is valid (via Pydantic model)

    Args:
        file: Uploaded file from Flask request
        max_size: Maximum allowed file size in bytes

    Returns:
        Validated ImageInput model with file metadata

    Raises:
        RequestValidationError: If file is missing, empty, too large, or invalid type

    Example:
        >>> from flask import request
        >>> file = request.files.get("image_file")
        >>> validated = validate_image_file(file, max_size=16*1024*1024)
        >>> print(f"File size: {validated.file_size} bytes")
    """
    if file is None or file.filename == "":
        raise RequestValidationError("No image file provided")

    # Get file size by seeking to end
    file.seek(0, 2)
    file_size = file.tell()
    file.seek(0)

    if file_size == 0:
        raise RequestValidationError("Empty file provided")

    if file_size > max_size:
        max_mb = max_size / (1024 * 1024)
        raise RequestValidationError(
            f"File too large. Maximum size is {max_mb:.1f}MB"
        )

    try:
        return ImageInput(
            file_size=file_size,
            content_type=file.content_type,
        )
    except ValidationError as e:
        # Extract the first error message from Pydantic validation
        error_msg = e.errors()[0].get("msg", "Invalid content type")
        raise RequestValidationError(f"Invalid file: {error_msg}")


def decode_image(file_bytes: bytes) -> np.ndarray:
    """
    Decode image bytes into a numpy array.

    Uses OpenCV to decode the image from raw bytes into a BGR numpy array
    suitable for processing with InsightFace. Verifies magic bytes before
    decoding to reject non-image files early.

    Args:
        file_bytes: Raw image bytes (JPEG, PNG, etc.)

    Returns:
        BGR image as numpy array with shape (height, width, 3)

    Raises:
        ImageValidationError: If file doesn't match supported image format
        ImageDecodeError: If decoding fails (invalid or corrupted image)

    Example:
        >>> with open("photo.jpg", "rb") as f:
        ...     image = decode_image(f.read())
        >>> print(f"Image shape: {image.shape}")
    """
    # Verify magic bytes before attempting decode
    verify_magic_bytes(file_bytes)

    nparr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise ImageDecodeError("Failed to decode image. File may be corrupted.")

    # Ensure image has 3 color channels (BGR)
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ImageDecodeError("Image must be a color image with 3 channels (RGB/BGR)")

    return image


def validate_image_dimensions(image: np.ndarray, max_dimension: int) -> None:
    """
    Validate that image dimensions are within acceptable limits.

    Checks that the image is large enough for face detection and logs
    a warning if the image is very large (which may impact performance).

    Args:
        image: Decoded image array
        max_dimension: Maximum dimension used for face detection

    Raises:
        ImageValidationError: If image is too small (< 10x10 pixels)

    Note:
        InsightFace handles resizing internally, so large images are allowed
        but a warning is logged for images > 4x the detection size.

    Example:
        >>> image = cv2.imread("photo.jpg")
        >>> validate_image_dimensions(image, max_dimension=640)
    """
    height, width = image.shape[:2]

    if height < 10 or width < 10:
        raise ImageValidationError(
            f"Image too small: {width}x{height}. Minimum is 10x10."
        )

    # Note: InsightFace handles resizing internally, but we log large images
    if height > max_dimension * 4 or width > max_dimension * 4:
        logging.getLogger("insightface-api").warning(
            f"Large image received: {width}x{height}. "
            f"Consider resizing for better performance."
        )


def extract_face_data(face: Any, decimals: int = 6) -> FaceEmbedding:
    """
    Extract structured data from a detected face object.

    Converts the InsightFace Face object into a Pydantic model with
    all available attributes.

    Args:
        face: InsightFace Face object containing detection results
        decimals: Decimal places kept in the embedding. float32 values print
            with 17 digits in JSON, which doubles the response size for no
            gain: 6 decimals keep cosine similarity to ~1e-7.

    Returns:
        FaceEmbedding model with embedding vector and metadata

    Note:
        Gender and age attributes are optional and may not be present
        depending on the InsightFace model configuration.

    Example:
        >>> faces = model_manager.get_faces(image)
        >>> embeddings = [extract_face_data(face) for face in faces]
    """
    return FaceEmbedding(
        embedding=np.round(face.embedding.astype(np.float64), decimals).tolist(),
        bbox=face.bbox.astype(int).tolist(),
        keypoints=face.kps.tolist(),
        det_score=float(face.det_score),
        gender=(
            int(face.gender)
            if hasattr(face, "gender") and face.gender is not None
            else None
        ),
        age=(
            int(face.age)
            if hasattr(face, "age") and face.age is not None
            else None
        ),
    )

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
from werkzeug.datastructures import FileStorage

from src.exceptions import ImageDecodeError, ImageValidationError, RequestValidationError
from src.models import FaceEmbedding, ImageInput


def validate_image_file(file: FileStorage, max_size: int) -> ImageInput:
    """
    Validate an uploaded image file before processing.

    Performs the following validations:
    - File exists and has a filename
    - File is not empty
    - File size is within limits

    Args:
        file: Uploaded file from Flask request
        max_size: Maximum allowed file size in bytes

    Returns:
        Validated ImageInput model with file metadata

    Raises:
        RequestValidationError: If file is missing, empty, or too large

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

    return ImageInput(
        file_size=file_size,
        content_type=file.content_type,
    )


def decode_image(file_bytes: bytes) -> np.ndarray:
    """
    Decode image bytes into a numpy array.

    Uses OpenCV to decode the image from raw bytes into a BGR numpy array
    suitable for processing with InsightFace.

    Args:
        file_bytes: Raw image bytes (JPEG, PNG, etc.)

    Returns:
        BGR image as numpy array with shape (height, width, 3)

    Raises:
        ImageDecodeError: If decoding fails (invalid or corrupted image)

    Example:
        >>> with open("photo.jpg", "rb") as f:
        ...     image = decode_image(f.read())
        >>> print(f"Image shape: {image.shape}")
    """
    nparr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise ImageDecodeError("Failed to decode image. File may be corrupted.")

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


def extract_face_data(face: Any) -> FaceEmbedding:
    """
    Extract structured data from a detected face object.

    Converts the InsightFace Face object into a Pydantic model with
    all available attributes.

    Args:
        face: InsightFace Face object containing detection results

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
        embedding=face.embedding.tolist(),
        bbox=face.bbox.astype(int).tolist(),
        keypoints=face.kps.tolist(),
        detection_score=float(face.det_score),
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

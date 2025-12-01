"""
API route definitions for the InsightFace API.

This module defines a Flask Blueprint with all API endpoints for
face detection and embedding extraction.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from flask import Blueprint, Response, current_app, g, jsonify, request

from src.models import HealthResponse, HealthStatus, RepresentResponse
from src.services import (
    ModelManager,
    decode_image,
    extract_face_data,
    validate_image_dimensions,
    validate_image_file,
)

if TYPE_CHECKING:
    from src.config import Settings

# Create blueprint for API routes
api_blueprint = Blueprint("api", __name__)


@api_blueprint.route("/", methods=["GET"])
def index() -> tuple[Response, int]:
    """
    API information endpoint.

    Returns basic information about the API including version
    and available endpoints.

    Returns:
        JSON response with API info and 200 status

    Example response:
        {
            "name": "InsightFace API",
            "version": "2.0.0",
            "endpoints": {
                "/represent": "POST - Extract face embeddings from image",
                "/health": "GET - Health check"
            }
        }
    """
    return jsonify({
        "name": "InsightFace API",
        "version": "2.0.0",
        "endpoints": {
            "/represent": "POST - Extract face embeddings from image",
            "/health": "GET - Health check",
        },
    }), 200


@api_blueprint.route("/health", methods=["GET"])
def health() -> tuple[Response, int]:
    """
    Health check endpoint with detailed status.

    Provides information about the service health including model
    status, uptime, and version information.

    Returns:
        - 200 with HealthResponse if healthy
        - 503 with HealthResponse if unhealthy/degraded

    Example response:
        {
            "status": "healthy",
            "model_loaded": true,
            "model_name": "buffalo_l",
            "version": "2.0.0",
            "uptime_seconds": 3600.5
        }
    """
    mm: ModelManager = current_app.config["model_manager"]
    settings: Settings = current_app.config["settings"]

    if mm.is_loaded:
        status = HealthStatus.HEALTHY
    elif mm.initialization_error:
        status = HealthStatus.UNHEALTHY
    else:
        status = HealthStatus.DEGRADED

    response = HealthResponse(
        status=status,
        model_loaded=mm.is_loaded,
        model_name=settings.model_name if mm.is_loaded else None,
        uptime_seconds=round(mm.uptime, 2),
    )

    status_code = 200 if status == HealthStatus.HEALTHY else 503
    return jsonify(response.model_dump()), status_code


@api_blueprint.route("/up", methods=["GET"])
def up() -> tuple[Response, int]:
    """
    Simple health check endpoint.

    Provides backwards compatibility with v1 API. Returns a simple
    status response without detailed health information.

    Returns:
        JSON response with status "ok" and 200 status

    Example response:
        {"status": "ok"}
    """
    return jsonify({"status": "ok"}), 200


@api_blueprint.route("/represent", methods=["POST"])
def represent() -> tuple[Response, int]:
    """
    Extract face embeddings from an uploaded image.

    Accepts a multipart form upload with an image file and returns
    detected faces with their embedding vectors.

    Request:
        - Content-Type: multipart/form-data
        - Field: image_file (required) - Image file (JPEG, PNG, etc.)

    Returns:
        RepresentResponse with detected faces and embeddings

    Raises:
        RequestValidationError (400): If no image provided or invalid
        ImageDecodeError (400): If image cannot be decoded
        ModelNotReadyError (503): If model is not initialized

    Example response:
        {
            "embeddings": [
                {
                    "embedding": [0.1, 0.2, ...],
                    "bbox": [100, 100, 200, 200],
                    "keypoints": [[120, 130], ...],
                    "detection_score": 0.95,
                    "gender": 1,
                    "age": 30
                }
            ],
            "faces_detected": 1,
            "processing_time_ms": 150.5,
            "request_id": "abc12345"
        }
    """
    mm: ModelManager = current_app.config["model_manager"]
    settings: Settings = current_app.config["settings"]
    logger = current_app.config["logger"]

    request_id = getattr(g, "request_id", "unknown")
    start_time = time.perf_counter()

    # Validate file upload
    image_file = request.files.get("image_file")
    validate_image_file(image_file, settings.max_content_length)

    # Read and decode image
    file_bytes = image_file.read()
    image = decode_image(file_bytes)

    # Validate dimensions
    validate_image_dimensions(image, settings.max_image_dimension)

    # Run face detection
    logger.info(f"[{request_id}] Running face detection...")
    detection_start = time.perf_counter()
    faces = mm.get_faces(image)
    detection_time = (time.perf_counter() - detection_start) * 1000

    logger.info(
        f"[{request_id}] Detection completed in {detection_time:.1f}ms. "
        f"Faces found: {len(faces)}"
    )

    # Extract embeddings
    embeddings = [extract_face_data(face) for face in faces]

    # Build response
    processing_time = (time.perf_counter() - start_time) * 1000
    response = RepresentResponse(
        embeddings=embeddings,
        faces_detected=len(faces),
        processing_time_ms=round(processing_time, 2),
        request_id=request_id,
    )

    return jsonify(response.model_dump()), 200

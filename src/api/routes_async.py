"""
Async API route definitions for the InsightFace API.

This module defines a Quart Blueprint with all API endpoints for
face detection and embedding extraction, using async/await for
non-blocking request handling.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from quart import Blueprint, Response, current_app, g, jsonify, request

from src.exceptions import ModelNotReadyError
from src.models import HealthResponse, HealthStatus, RepresentResponse
from src.instrumentation import add_attribute, notice_error, record_metric, trace
from src.services import (
    ModelManager,
    decode_image,
    extract_face_data,
    validate_image_dimensions,
    validate_image_file,
)

if TYPE_CHECKING:
    from src.config import Settings
    from src.services import InferenceExecutor

# Create blueprint for API routes
api_blueprint_async = Blueprint("api", __name__)


@api_blueprint_async.route("/", methods=["GET"])
async def index() -> tuple[Response, int]:
    """
    API information endpoint.

    Returns basic information about the API including version
    and available endpoints.

    Returns:
        JSON response with API info and 200 status
    """
    return jsonify({
        "name": "InsightFace API",
        "version": "2.0.0",
        "endpoints": {
            "/represent": "POST - Extract face embeddings from image",
            "/health": "GET - Health check",
        },
    }), 200


@api_blueprint_async.route("/health", methods=["GET"])
async def health() -> tuple[Response, int]:
    """
    Health check endpoint with detailed status.

    Provides information about the service health including model
    status, uptime, and version information.

    Returns:
        - 200 with HealthResponse if healthy
        - 503 with HealthResponse if unhealthy/degraded
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


@api_blueprint_async.route("/up", methods=["GET"])
async def up() -> tuple[Response, int]:
    """
    Simple health check endpoint.

    Provides backwards compatibility with v1 API. Returns a simple
    status response without detailed health information.

    Returns:
        JSON response with status "ok" and 200 status
    """
    return jsonify({"status": "ok"}), 200


@api_blueprint_async.route("/represent", methods=["POST"])
async def represent() -> tuple[Response, int]:
    """
    Extract face embeddings from an uploaded image.

    Accepts a multipart form upload with an image file and returns
    detected faces with their embedding vectors.

    This async version runs inference in a thread pool to avoid
    blocking the event loop.

    Request:
        - Content-Type: multipart/form-data
        - Field: image_file (required) - Image file (JPEG, PNG, etc.)

    Returns:
        RepresentResponse with detected faces and embeddings

    Raises:
        RequestValidationError (400): If no image provided or invalid
        ImageDecodeError (400): If image cannot be decoded
        ModelNotReadyError (503): If model is not initialized
    """
    mm: ModelManager = current_app.config["model_manager"]
    executor: InferenceExecutor = current_app.config["inference_executor"]
    settings: Settings = current_app.config["settings"]
    logger = current_app.config["logger"]

    request_id = getattr(g, "request_id", "unknown")

    # Check model readiness early to avoid processing if model isn't ready
    if not mm.is_loaded:
        raise ModelNotReadyError(
            mm.initialization_error or "Model not initialized"
        )

    start_time = time.perf_counter()

    add_attribute("request_id", request_id)

    # --- Validate & read upload ---
    with trace("validate_request", "Custom/Validation"):
        files = await request.files
        image_file = files.get("image_file")
        validate_image_file(image_file, settings.max_content_length)
        file_bytes = image_file.read()  # FileStorage.read() is synchronous

    add_attribute("image_size_bytes", len(file_bytes))

    # --- Decode image bytes → numpy array ---
    with trace("decode_image", "Custom/ImageProcessing"):
        image = decode_image(file_bytes)

    add_attribute("image_width", image.shape[1])
    add_attribute("image_height", image.shape[0])

    with trace("validate_dimensions", "Custom/Validation"):
        validate_image_dimensions(image, settings.max_image_dimension)

    # --- Run face detection in thread pool (non-blocking) ---
    logger.info(f"[{request_id}] Running face detection...")
    detection_start = time.perf_counter()

    with trace("face_detection", "Custom/Inference"):
        faces = await executor.run(mm.get_faces, image)

    detection_time = (time.perf_counter() - detection_start) * 1000

    logger.info(
        f"[{request_id}] Detection completed in {detection_time:.1f}ms. "
        f"Faces found: {len(faces)}"
    )

    # --- Extract embeddings from detected faces ---
    with trace("extract_embeddings", "Custom/Extraction"):
        embeddings = [extract_face_data(face) for face in faces]

    # --- Transaction attributes (NRQL queryable) ---
    add_attribute("faces_detected", len(faces))
    add_attribute("detection_time_ms", round(detection_time, 2))

    processing_time = (time.perf_counter() - start_time) * 1000
    add_attribute("processing_time_ms", round(processing_time, 2))

    # --- Custom metrics (time-series, dashboards & alerting) ---
    record_metric("Inference/DurationMs", detection_time)
    record_metric("Inference/FacesDetected", len(faces))
    record_metric("Image/SizeBytes", len(file_bytes))
    record_metric("Request/ProcessingTimeMs", processing_time)

    response = RepresentResponse(
        embeddings=embeddings,
        faces_detected=len(faces),
        processing_time_ms=round(processing_time, 2),
        request_id=request_id,
    )

    return jsonify(response.model_dump()), 200

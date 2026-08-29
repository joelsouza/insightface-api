"""
Async API route definitions for the InsightFace API.

This module defines a Quart Blueprint with all API endpoints for
face detection and embedding extraction, using async/await for
non-blocking request handling.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

from quart import Blueprint, Response, current_app, g, jsonify, request

from src.exceptions import ModelNotReadyError, RequestValidationError
from src.models import HealthResponse, HealthStatus, RepresentResponse
from src.instrumentation import add_attribute, trace
from src.services import (
    ModelManager,
    fetch_image,
    process_image,
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


async def _read_input(settings: Settings) -> tuple[bytes, str]:
    """
    Read the request image, from either an upload or a URL.

    Exactly one of `image_file` (multipart) and `image_url` (JSON body or form
    field) must be present.

    Args:
        settings: Application settings

    Returns:
        (raw image bytes, input mode: "file" or "url")

    Raises:
        RequestValidationError: If neither or both inputs are present
        ImageDownloadError: If a URL download fails
    """
    image_url: Optional[str] = None
    image_file = None

    if request.is_json:
        payload = await request.get_json(silent=True)
        if not isinstance(payload, dict):
            raise RequestValidationError("Request body must be a JSON object")
        image_url = payload.get("image_url")
        if image_url is not None and not isinstance(image_url, str):
            raise RequestValidationError("image_url must be a string")
    else:
        form = await request.form
        image_url = form.get("image_url") or None
        files = await request.files
        image_file = files.get("image_file")

    if image_url and image_file is not None:
        raise RequestValidationError(
            "Provide either image_file or image_url, not both"
        )

    if not image_url and image_file is None:
        raise RequestValidationError(
            "No image provided. Send image_file or image_url."
        )

    if image_url:
        g.nr_event["input_mode"] = "url"
        client = current_app.config.get("http_client")
        if client is None:
            raise RequestValidationError("image_url input is not available")

        with trace("download_image", "Custom/ImageDownload"):
            download_start = time.perf_counter()
            try:
                data = await fetch_image(
                    image_url,
                    client,
                    current_app.config["download_semaphore"],
                    settings,
                    stats=getattr(g, "nr_event", None),
                )
            finally:
                g.nr_event["download_ms"] = (
                    time.perf_counter() - download_start
                ) * 1000
            return data, "url"

    assert image_file is not None  # narrowed by the checks above
    g.nr_event["input_mode"] = "file"
    validate_image_file(image_file, settings.max_content_length)
    return image_file.read(), "file"  # FileStorage.read() is synchronous


@api_blueprint_async.route("/represent", methods=["POST"])
async def represent() -> tuple[Response, int]:
    """
    Extract face embeddings from an image.

    Accepts either a multipart upload or a URL to download. All CPU-bound work
    (decode, validation, inference, extraction) runs as one job in the
    inference thread pool, so the event loop is never blocked.

    Request:
        - `multipart/form-data` with field `image_file`, or
        - `application/json` with field `image_url`

    Returns:
        RepresentResponse with detected faces and embeddings

    Raises:
        RequestValidationError (400): If no image provided or invalid
        ImageDecodeError (400): If image cannot be decoded
        ImageDownloadError (502): If the image URL cannot be fetched
        ModelNotReadyError (503): If model is not initialized
        ServiceOverloadedError (503): If no inference slot is free
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

    # --- Read the image, from an upload or a URL ---
    with trace("read_input", "Custom/Validation"):
        file_bytes, input_mode = await _read_input(settings)

    event = g.nr_event
    event["image_bytes"] = len(file_bytes)
    event["input_mode"] = input_mode

    add_attribute("image_size_bytes", len(file_bytes))
    add_attribute("input_mode", input_mode)

    # --- One thread-pool job: decode, validate, detect, extract ---
    logger.info(
        "inference_started",
        extra={"request_id": request_id, "input_mode": input_mode},
    )

    with trace("inference", "Custom/Inference"):
        result = await executor.run(
            process_image, file_bytes, mm, settings, stats=event
        )

    logger.info(
        "inference_completed",
        extra={
            "request_id": request_id,
            "input_mode": input_mode,
            "detect_ms": result.detect_ms,
            "decode_ms": result.decode_ms,
            "faces_detected": len(result.faces),
        },
    )

    # --- Transaction attributes (NRQL queryable) ---
    add_attribute("image_width", result.width)
    add_attribute("image_height", result.height)
    add_attribute("faces_detected", len(result.faces))
    add_attribute("detection_time_ms", round(result.detect_ms, 2))
    add_attribute("decode_time_ms", round(result.decode_ms, 2))

    event.update(
        {
            "image_width": result.width,
            "image_height": result.height,
            "downscaled": result.downscaled,
            "faces_detected": len(result.faces),
            "decode_ms": result.decode_ms,
            "detect_ms": result.detect_ms,
            "align_ms": result.align_ms,
            "embed_ms": result.embed_ms,
            "extract_ms": result.extract_ms,
            "in_flight": executor.in_flight,
        }
    )

    processing_time = (time.perf_counter() - start_time) * 1000
    add_attribute("processing_time_ms", round(processing_time, 2))

    response = RepresentResponse(
        embeddings=result.faces,
        faces_detected=len(result.faces),
        processing_time_ms=round(processing_time, 2),
        request_id=request_id,
    )

    # Serialize straight from Pydantic: `jsonify(model_dump())` re-validates
    # and re-walks 512 floats per face for no benefit.
    return (
        Response(response.model_dump_json(), content_type="application/json"),
        200,
    )

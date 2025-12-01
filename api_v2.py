"""
InsightFace API v2 - Face Detection and Embedding Extraction Service

A production-ready Flask API for extracting facial embeddings using InsightFace.
Designed with focus on performance, resilience, and maintainability.

Features:
    - Face detection and embedding extraction
    - Configurable via environment variables
    - Comprehensive error handling
    - Request validation with Pydantic
    - Detailed health checks
    - Structured logging

Example:
    Start the server:
        $ python api-v2.py

    Or with Gunicorn:
        $ gunicorn -w 4 -b 0.0.0.0:5001 api-v2:app

    Extract embeddings:
        $ curl -X POST -F "image_file=@photo.jpg" http://localhost:5001/represent
"""

from __future__ import annotations

import gc
import logging
import os
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generator, Optional

import cv2
import numpy as np
from flask import Flask, Response, g, jsonify, request
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings
from werkzeug.datastructures import FileStorage
from werkzeug.exceptions import HTTPException

# =============================================================================
# Configuration
# =============================================================================


class ExecutionProvider(str, Enum):
    """Supported ONNX execution providers."""

    CPU = "CPUExecutionProvider"
    CUDA = "CUDAExecutionProvider"
    TENSORRT = "TensorrtExecutionProvider"


class Settings(BaseSettings):
    """
    Application configuration loaded from environment variables.

    All settings can be overridden via environment variables with the same name.

    Attributes:
        port: Server port (default: 5001)
        max_content_length: Maximum upload size in bytes (default: 16MB)
        max_image_dimension: Maximum image width/height for processing (default: 640)
        detection_threshold: Minimum confidence for face detection (default: 0.5)
        model_name: InsightFace model to use (default: buffalo_l)
        model_root: Directory for model cache (default: ./insightface)
        execution_provider: ONNX execution provider (default: CPU)
        log_level: Logging level (default: INFO)
        request_timeout: Request processing timeout in seconds (default: 30)
    """

    port: int = Field(default=5001, ge=1, le=65535)
    max_content_length: int = Field(default=16 * 1024 * 1024, ge=1024)
    max_image_dimension: int = Field(default=640, ge=64, le=4096)
    detection_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    model_name: str = Field(default="buffalo_l")
    model_root: str = Field(default="./insightface")
    execution_provider: ExecutionProvider = Field(default=ExecutionProvider.CPU)
    log_level: str = Field(default="INFO")
    request_timeout: float = Field(default=30.0, ge=1.0)

    model_config = {
        "env_prefix": "",
        "case_sensitive": False,
    }


# =============================================================================
# Logging Setup
# =============================================================================


def setup_logging(level: str) -> logging.Logger:
    """
    Configure structured logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    return logging.getLogger("insightface-api")


# =============================================================================
# Custom Exceptions
# =============================================================================


class APIError(Exception):
    """Base exception for API errors."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code or self.__class__.__name__


class ImageDecodeError(APIError):
    """Raised when image decoding fails."""

    def __init__(self, message: str = "Failed to decode image") -> None:
        super().__init__(message, status_code=400)


class ImageValidationError(APIError):
    """Raised when image validation fails."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=400)


class ModelNotReadyError(APIError):
    """Raised when the ML model is not available."""

    def __init__(self, message: str = "Model not initialized") -> None:
        super().__init__(message, status_code=503)


class RequestValidationError(APIError):
    """Raised when request validation fails."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=400)


# =============================================================================
# Pydantic Models (Request/Response Schemas)
# =============================================================================


class FaceEmbedding(BaseModel):
    """
    Extracted face data including embedding and metadata.

    Attributes:
        embedding: 512-dimensional face embedding vector
        bbox: Bounding box coordinates [x1, y1, x2, y2]
        keypoints: Facial landmark coordinates (5 points)
        detection_score: Confidence score of face detection
        gender: Predicted gender (0=female, 1=male) if available
        age: Predicted age if available
    """

    embedding: list[float] = Field(description="512-dimensional face embedding")
    bbox: list[int] = Field(description="Bounding box [x1, y1, x2, y2]")
    keypoints: list[list[float]] = Field(description="Facial landmark coordinates")
    detection_score: float = Field(ge=0.0, le=1.0, description="Detection confidence")
    gender: Optional[int] = Field(default=None, description="Gender (0=F, 1=M)")
    age: Optional[int] = Field(default=None, ge=0, le=120, description="Predicted age")


class RepresentResponse(BaseModel):
    """
    Response from the /represent endpoint.

    Attributes:
        embeddings: List of detected faces with embeddings
        faces_detected: Number of faces found in the image
        processing_time_ms: Time taken to process the image
        request_id: Unique identifier for this request
    """

    embeddings: list[FaceEmbedding] = Field(default_factory=list)
    faces_detected: int = Field(ge=0)
    processing_time_ms: float = Field(ge=0)
    request_id: str


class ErrorResponse(BaseModel):
    """
    Standardized error response format.

    Attributes:
        error: Human-readable error message
        error_code: Machine-readable error code
        request_id: Unique identifier for debugging
    """

    error: str
    error_code: str
    request_id: Optional[str] = None


class HealthStatus(str, Enum):
    """Health check status values."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class HealthResponse(BaseModel):
    """
    Response from the /health endpoint.

    Attributes:
        status: Overall health status
        model_loaded: Whether the ML model is ready
        model_name: Name of the loaded model
        version: API version
        uptime_seconds: Seconds since server start
    """

    status: HealthStatus
    model_loaded: bool
    model_name: Optional[str] = None
    version: str = "2.0.0"
    uptime_seconds: float


class ImageInput(BaseModel):
    """Validated image input parameters."""

    file_size: int = Field(ge=1)
    content_type: Optional[str] = None

    @field_validator("content_type")
    @classmethod
    def validate_content_type(cls, v: Optional[str]) -> Optional[str]:
        """Validate that content type is an accepted image format."""
        if v is None:
            return v
        allowed = {"image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp"}
        if v.lower() not in allowed:
            raise ValueError(f"Unsupported content type: {v}")
        return v


# =============================================================================
# Model Manager
# =============================================================================


@dataclass
class ModelManager:
    """
    Manages the InsightFace model lifecycle.

    Handles model loading, inference, and resource cleanup with proper
    error handling and logging.

    Attributes:
        settings: Application settings
        logger: Logger instance
        model: InsightFace FaceAnalysis instance
        is_loaded: Whether the model is ready for inference
        load_time: Time when the model was loaded
        load_duration: How long model loading took
    """

    settings: Settings
    logger: logging.Logger
    model: Any = field(default=None, repr=False)
    is_loaded: bool = field(default=False)
    load_time: Optional[float] = field(default=None)
    load_duration: Optional[float] = field(default=None)
    _initialization_error: Optional[str] = field(default=None)

    def load(self) -> bool:
        """
        Load the InsightFace model.

        Returns:
            True if loading succeeded, False otherwise
        """
        # Defer import to allow mocking in tests
        import insightface

        self.logger.info(f"Loading InsightFace model '{self.settings.model_name}'...")
        start_time = time.perf_counter()

        try:
            self.model = insightface.app.FaceAnalysis(
                name=self.settings.model_name,
                root=self.settings.model_root,
                allowed_modules=["detection", "recognition"],
                providers=[self.settings.execution_provider.value],
            )
            self.model.prepare(
                ctx_id=0,
                det_size=(
                    self.settings.max_image_dimension,
                    self.settings.max_image_dimension,
                ),
                det_thresh=self.settings.detection_threshold,
            )

            self.load_duration = time.perf_counter() - start_time
            self.load_time = time.time()
            self.is_loaded = True
            self._initialization_error = None

            self.logger.info(
                f"Model loaded successfully in {self.load_duration:.2f}s"
            )
            return True

        except Exception as e:
            self.load_duration = time.perf_counter() - start_time
            self.is_loaded = False
            self._initialization_error = str(e)
            self.logger.error(f"Failed to load model: {e}")
            return False

    def unload(self) -> None:
        """Release model resources and trigger garbage collection."""
        if self.model is not None:
            self.logger.info("Unloading model and releasing resources...")
            del self.model
            self.model = None
            self.is_loaded = False
            gc.collect()

    def get_faces(self, image: np.ndarray) -> list[Any]:
        """
        Detect faces and extract embeddings from an image.

        Args:
            image: BGR image as numpy array

        Returns:
            List of detected face objects with embeddings

        Raises:
            ModelNotReadyError: If model is not loaded
        """
        if not self.is_loaded or self.model is None:
            raise ModelNotReadyError(
                self._initialization_error or "Model not initialized"
            )
        return self.model.get(image)

    @property
    def uptime(self) -> float:
        """Return seconds since model was loaded, or 0 if not loaded."""
        if self.load_time is None:
            return 0.0
        return time.time() - self.load_time


# =============================================================================
# Image Processing
# =============================================================================


def validate_image_file(file: FileStorage, max_size: int) -> ImageInput:
    """
    Validate an uploaded image file before processing.

    Args:
        file: Uploaded file from request
        max_size: Maximum allowed file size in bytes

    Returns:
        Validated ImageInput model

    Raises:
        RequestValidationError: If file is missing or invalid
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

    Args:
        file_bytes: Raw image bytes

    Returns:
        BGR image as numpy array

    Raises:
        ImageDecodeError: If decoding fails
    """
    nparr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise ImageDecodeError("Failed to decode image. File may be corrupted.")

    return image


def validate_image_dimensions(
    image: np.ndarray,
    max_dimension: int,
) -> None:
    """
    Validate that image dimensions are within acceptable limits.

    Args:
        image: Decoded image array
        max_dimension: Maximum allowed width/height

    Raises:
        ImageValidationError: If dimensions are invalid
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

    Args:
        face: InsightFace Face object

    Returns:
        FaceEmbedding with all available data
    """
    return FaceEmbedding(
        embedding=face.embedding.tolist(),
        bbox=face.bbox.astype(int).tolist(),
        keypoints=face.kps.tolist(),
        detection_score=float(face.det_score),
        gender=int(face.gender) if hasattr(face, "gender") and face.gender is not None else None,
        age=int(face.age) if hasattr(face, "age") and face.age is not None else None,
    )


# =============================================================================
# Request Context Manager
# =============================================================================


@contextmanager
def request_context(logger: logging.Logger) -> Generator[str, None, None]:
    """
    Context manager for request-scoped operations.

    Generates a unique request ID and logs request start/end.

    Args:
        logger: Logger instance

    Yields:
        Unique request ID
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.perf_counter()

    logger.info(f"[{request_id}] Request started")
    try:
        yield request_id
    finally:
        duration = (time.perf_counter() - start_time) * 1000
        logger.info(f"[{request_id}] Request completed in {duration:.1f}ms")


# =============================================================================
# Flask Application Factory
# =============================================================================


def create_app(settings: Optional[Settings] = None) -> Flask:
    """
    Create and configure the Flask application.

    Args:
        settings: Application settings (loads from env if not provided)

    Returns:
        Configured Flask application
    """
    if settings is None:
        settings = Settings()

    # Setup logging
    logger = setup_logging(settings.log_level)

    # Create Flask app
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = settings.max_content_length

    # Store settings and logger in app config for access in routes
    app.config["settings"] = settings
    app.config["logger"] = logger

    # Initialize model manager
    model_manager = ModelManager(settings=settings, logger=logger)
    model_manager.load()
    app.config["model_manager"] = model_manager

    # -------------------------------------------------------------------------
    # Request Hooks
    # -------------------------------------------------------------------------

    @app.before_request
    def before_request() -> None:
        """Generate request ID and store start time."""
        g.request_id = str(uuid.uuid4())[:8]
        g.start_time = time.perf_counter()

    @app.after_request
    def after_request(response: Response) -> Response:
        """Add request ID header to response."""
        if hasattr(g, "request_id"):
            response.headers["X-Request-ID"] = g.request_id
        return response

    # -------------------------------------------------------------------------
    # Error Handlers
    # -------------------------------------------------------------------------

    @app.errorhandler(APIError)
    def handle_api_error(error: APIError) -> tuple[Response, int]:
        """Handle custom API errors."""
        request_id = getattr(g, "request_id", None)
        logger.warning(f"[{request_id}] {error.error_code}: {error.message}")

        response = ErrorResponse(
            error=error.message,
            error_code=error.error_code,
            request_id=request_id,
        )
        return jsonify(response.model_dump()), error.status_code

    @app.errorhandler(HTTPException)
    def handle_http_exception(error: HTTPException) -> tuple[Response, int]:
        """Handle Werkzeug HTTP exceptions."""
        request_id = getattr(g, "request_id", None)
        logger.warning(f"[{request_id}] HTTP {error.code}: {error.description}")

        response = ErrorResponse(
            error=error.description or "An error occurred",
            error_code=f"HTTP_{error.code}",
            request_id=request_id,
        )
        return jsonify(response.model_dump()), error.code or 500

    @app.errorhandler(Exception)
    def handle_exception(error: Exception) -> tuple[Response, int]:
        """Handle unexpected exceptions."""
        request_id = getattr(g, "request_id", None)
        logger.exception(f"[{request_id}] Unexpected error: {error}")

        response = ErrorResponse(
            error="An internal error occurred",
            error_code="INTERNAL_ERROR",
            request_id=request_id,
        )
        return jsonify(response.model_dump()), 500

    # -------------------------------------------------------------------------
    # Routes
    # -------------------------------------------------------------------------

    @app.route("/", methods=["GET"])
    def index() -> tuple[Response, int]:
        """
        API information endpoint.

        Returns:
            JSON with API version and available endpoints
        """
        return jsonify({
            "name": "InsightFace API",
            "version": "2.0.0",
            "endpoints": {
                "/represent": "POST - Extract face embeddings from image",
                "/health": "GET - Health check",
            },
        }), 200

    @app.route("/health", methods=["GET"])
    def health() -> tuple[Response, int]:
        """
        Health check endpoint with detailed status.

        Returns:
            HealthResponse with model status and uptime
        """
        mm: ModelManager = app.config["model_manager"]

        if mm.is_loaded:
            status = HealthStatus.HEALTHY
        elif mm._initialization_error:
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

    @app.route("/up", methods=["GET"])
    def up() -> tuple[Response, int]:
        """
        Simple health check (backwards compatible with v1).

        Returns:
            Simple status response
        """
        return jsonify({"status": "ok"}), 200

    @app.route("/represent", methods=["POST"])
    def represent() -> tuple[Response, int]:
        """
        Extract face embeddings from an uploaded image.

        Expects a multipart form upload with field name 'image_file'.

        Returns:
            RepresentResponse with detected faces and embeddings

        Raises:
            RequestValidationError: If no image provided or invalid
            ImageDecodeError: If image cannot be decoded
            ModelNotReadyError: If model is not initialized
        """
        mm: ModelManager = app.config["model_manager"]
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

    return app


# =============================================================================
# Application Entry Point
# =============================================================================


def get_app() -> Flask:
    """
    Get or create the Flask application instance (lazy loaded).

    This function provides lazy initialization, allowing the module to be
    imported without triggering model loading. Useful for testing and
    for WSGI servers that need a callable.

    Usage with Gunicorn:
        gunicorn "api_v2:get_app()"

    Returns:
        Flask application instance
    """
    global _app_instance
    if _app_instance is None:
        _app_instance = create_app()
    return _app_instance


# Global app instance holder (lazy loaded via get_app())
_app_instance: Optional[Flask] = None


if __name__ == "__main__":
    settings = Settings()
    app = create_app(settings)
    app.run(
        host="0.0.0.0",
        port=settings.port,
        debug=False,
    )

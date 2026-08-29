"""
Quart application factory for the InsightFace API (async version).

This module provides the application factory pattern for creating
and configuring the Quart application with async support.

The key difference from the Flask version is that this uses a single
worker with a thread pool for inference, allowing the model to be
loaded once and shared across all requests.

Example:
    # Production with Uvicorn
    uvicorn "src.app_async:get_app" --factory --host 0.0.0.0 --port 5001
"""

from __future__ import annotations

import asyncio
import atexit
import io
import logging
import time
import uuid
from typing import IO, Optional

from quart import Quart, Response, g, jsonify, request
from quart.formparser import FormDataParser
from quart.wrappers import Request
from werkzeug.exceptions import HTTPException

from src.api.routes_async import api_blueprint_async
from src.config import Settings, setup_logging
from src.exceptions import APIError
from src.instrumentation import (
    add_attribute,
    asgi_wrap,
    ignore_transaction,
    notice_error,
    record_event,
    set_transaction_name,
)
from src.models import ErrorResponse
from src.services import InferenceExecutor, ModelManager


def _in_memory_stream(*_args: object, **_kwargs: object) -> IO[bytes]:
    """Return an in-memory buffer for an uploaded file part."""
    return io.BytesIO()


class InMemoryRequest(Request):
    """
    Request that keeps uploads in memory instead of spooling them to disk.

    Werkzeug's default stream factory switches to a temporary file above
    500 KB, so every ordinary photo upload becomes a disk write and a read
    back. `MAX_CONTENT_LENGTH` already bounds how much memory this can use.
    """

    def make_form_data_parser(self) -> FormDataParser:
        """Build a form parser that buffers file parts in memory."""
        return self.form_data_parser_class(
            max_content_length=self.max_content_length,
            max_form_memory_size=self.max_form_memory_size,
            max_form_parts=self.max_form_parts,
            cls=self.parameter_storage_class,
            stream_factory=_in_memory_stream,
        )


def create_async_app(settings: Optional[Settings] = None) -> Quart:
    """
    Create and configure the Quart application.

    This factory function creates a new Quart application instance with
    all routes, error handlers, and middleware configured.

    Args:
        settings: Application settings. If not provided, settings are
                 loaded from environment variables.

    Returns:
        Configured Quart application instance
    """
    if settings is None:
        settings = Settings()

    # Setup logging
    logger = setup_logging(settings.log_level)

    # Create Quart app
    app = Quart(__name__)
    app.request_class = InMemoryRequest
    app.config["MAX_CONTENT_LENGTH"] = settings.max_content_length

    # Store settings and logger in app config for access in routes
    app.config["settings"] = settings
    app.config["logger"] = logger

    # Initialize model manager (single instance for all requests)
    model_manager = ModelManager(settings=settings, logger=logger)
    model_manager.load()
    app.config["model_manager"] = model_manager

    # Initialize inference executor (thread pool for CPU-bound inference)
    inference_executor = InferenceExecutor(
        max_workers=settings.inference_pool_size,
        timeout=settings.inference_timeout,
        max_queue=settings.inference_max_queue,
    )
    app.config["inference_executor"] = inference_executor

    # Shared HTTP client for the image_url input
    app.config["http_client"] = None
    app.config["download_semaphore"] = asyncio.Semaphore(
        settings.download_max_concurrency
    )

    @app.before_serving
    async def _open_http_client() -> None:
        """Open the shared HTTP client used to download images."""
        import httpx

        app.config["http_client"] = httpx.AsyncClient(
            timeout=settings.download_timeout,
            follow_redirects=False,
            limits=httpx.Limits(
                max_connections=settings.download_max_concurrency,
                max_keepalive_connections=settings.download_max_concurrency,
            ),
        )

    @app.after_serving
    async def _close_http_client() -> None:
        """Close the shared HTTP client."""
        client = app.config.get("http_client")
        if client is not None:
            await client.aclose()
            app.config["http_client"] = None

    # Register blueprints
    app.register_blueprint(api_blueprint_async)

    # Register error handlers
    _register_error_handlers(app, logger)

    # Register request hooks
    _register_request_hooks(app)

    # Register shutdown handler to clean up resources
    def shutdown_handler() -> None:
        """Clean up resources on application shutdown."""
        logger.info("Shutting down async application...")
        inference_executor.shutdown(wait=True)
        model_manager.unload()
        logger.info("Shutdown complete")

    atexit.register(shutdown_handler)

    logger.info(
        f"Async application initialized (pool_size={settings.inference_pool_size}, "
        f"max_queue={settings.inference_max_queue}, "
        f"ort_intra_op_threads={settings.ort_intra_op_threads}, "
        f"timeout={settings.inference_timeout}s)"
    )

    return app


def _register_error_handlers(app: Quart, logger: logging.Logger) -> None:
    """
    Register error handlers with the Quart application.

    Args:
        app: Quart application instance
        logger: Logger for error messages
    """

    @app.errorhandler(APIError)
    async def handle_api_error(error: APIError) -> tuple[Response, int]:
        """Handle custom API errors."""
        request_id = getattr(g, "request_id", None)
        if error.status_code >= 500:
            notice_error()
        if hasattr(g, "nr_event"):
            g.nr_event["error_code"] = error.error_code
        add_attribute("request_id", request_id)
        add_attribute("error_code", error.error_code)
        logger.warning(
            "api_error",
            extra={
                "request_id": request_id,
                "error_code": error.error_code,
                "status_code": error.status_code,
            },
        )

        payload = ErrorResponse(
            error=error.message,
            error_code=error.error_code,
            request_id=request_id,
        )
        response = jsonify(payload.model_dump())

        # Tell batch clients when to come back instead of hammering the API.
        if error.status_code == 503:
            response.headers["Retry-After"] = "1"

        return response, error.status_code

    @app.errorhandler(HTTPException)
    async def handle_http_exception(error: HTTPException) -> tuple[dict, int]:
        """Handle Werkzeug HTTP exceptions."""
        request_id = getattr(g, "request_id", None)
        error_code = f"HTTP_{error.code}"
        if hasattr(g, "nr_event"):
            g.nr_event["error_code"] = error_code
        add_attribute("request_id", request_id)
        add_attribute("error_code", error_code)
        logger.warning(
            "http_error",
            extra={
                "request_id": request_id,
                "error_code": error_code,
                "status_code": error.code,
            },
        )

        response = ErrorResponse(
            error=error.description or "An error occurred",
            error_code=error_code,
            request_id=request_id,
        )
        return response.model_dump(), error.code or 500

    @app.errorhandler(Exception)
    async def handle_exception(error: Exception) -> tuple[dict, int]:
        """Handle unexpected exceptions."""
        request_id = getattr(g, "request_id", None)
        if hasattr(g, "nr_event"):
            g.nr_event["error_code"] = "INTERNAL_ERROR"
        add_attribute("request_id", request_id)
        add_attribute("error_code", "INTERNAL_ERROR")
        logger.exception(
            "unexpected_error",
            extra={"request_id": request_id, "error_code": "INTERNAL_ERROR"},
        )
        notice_error()

        response = ErrorResponse(
            error="An internal error occurred",
            error_code="INTERNAL_ERROR",
            request_id=request_id,
        )
        return response.model_dump(), 500


def _register_request_hooks(app: Quart) -> None:
    """
    Register before/after request hooks.

    Args:
        app: Quart application instance
    """

    @app.before_request
    async def before_request() -> None:
        """Generate request ID and store start time."""
        g.request_id = str(uuid.uuid4())[:8]
        g.start_time = time.perf_counter()
        g.nr_event = {
            "request_id": g.request_id,
            "input_mode": None,
            "status_code": None,
            "error_code": None,
            "image_bytes": None,
            "image_width": None,
            "image_height": None,
            "downscaled": None,
            "faces_detected": None,
            "download_ms": 0.0,
            "dns_ms": None,
            "semaphore_wait_ms": None,
            "transfer_ms": None,
            "queue_wait_ms": None,
            "decode_ms": None,
            "detect_ms": None,
            "align_ms": None,
            "embed_ms": None,
            "extract_ms": None,
            "total_ms": None,
            "pool_size": None,
            "max_queue": None,
            "in_flight": None,
        }

        if request.path == "/up":
            ignore_transaction()
        elif request.endpoint:
            set_transaction_name(request.endpoint, group="Quart")

    @app.after_request
    async def after_request(response: Response) -> Response:
        """Add request ID header and record the represent event."""
        if hasattr(g, "request_id"):
            response.headers["X-Request-ID"] = g.request_id

        if getattr(request, "endpoint", None) == "api.represent":
            event = getattr(g, "nr_event", {"request_id": g.request_id})
            event["status_code"] = response.status_code
            event["total_ms"] = (time.perf_counter() - g.start_time) * 1000

            executor = app.config.get("inference_executor")
            if executor is not None:
                if event.get("pool_size") is None:
                    event["pool_size"] = executor.max_workers
                if event.get("max_queue") is None:
                    event["max_queue"] = executor.max_queue
                if event.get("in_flight") is None:
                    event["in_flight"] = executor.in_flight

            for key, value in event.items():
                if value is not None:
                    add_attribute(key, value)
            record_event("FaceRepresent", event)
        return response


def get_app() -> Quart:
    """
    Get or create the Quart application instance (lazy loaded).

    This function provides lazy initialization, allowing the module to be
    imported without triggering model loading. Useful for testing and
    for ASGI servers that need a callable.

    Usage with Uvicorn:
        uvicorn "src.app_async:get_app" --factory

    Returns:
        Quart application instance
    """
    global _app_instance
    if _app_instance is None:
        _app_instance = asgi_wrap(create_async_app())
    return _app_instance


# Global app instance holder (lazy loaded via get_app())
_app_instance: Optional[Quart] = None


if __name__ == "__main__":
    import asyncio

    settings = Settings()
    app = create_async_app(settings)
    asyncio.run(app.run_task(host="0.0.0.0", port=settings.port))

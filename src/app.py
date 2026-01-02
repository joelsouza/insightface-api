"""
Flask application factory for the InsightFace API.

This module provides the application factory pattern for creating
and configuring the Flask application.

Example:
    # Development server
    python -m src.app

    # Production with Gunicorn
    gunicorn "src.app:get_app()"
"""

from __future__ import annotations

import atexit
import time
import uuid
from typing import Optional

from flask import Flask, Response, g

from src.api import api_blueprint, register_error_handlers
from src.config import Settings, setup_logging
from src.services import ModelManager


def create_app(settings: Optional[Settings] = None) -> Flask:
    """
    Create and configure the Flask application.

    This factory function creates a new Flask application instance with
    all routes, error handlers, and middleware configured.

    Args:
        settings: Application settings. If not provided, settings are
                 loaded from environment variables.

    Returns:
        Configured Flask application instance

    Example:
        >>> app = create_app()
        >>> app.run(debug=True)

        >>> # With custom settings
        >>> settings = Settings(port=8080, log_level="DEBUG")
        >>> app = create_app(settings)
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

    # Register shutdown handler to clean up resources
    def shutdown_handler() -> None:
        """Clean up resources on application shutdown."""
        logger.info("Shutting down application...")
        model_manager.unload()
        logger.info("Shutdown complete")

    atexit.register(shutdown_handler)

    # Register blueprints
    app.register_blueprint(api_blueprint)

    # Register error handlers
    register_error_handlers(app, logger)

    # Register request hooks
    _register_request_hooks(app)

    logger.info("Application initialized successfully")

    return app


def _register_request_hooks(app: Flask) -> None:
    """
    Register before/after request hooks.

    Sets up request ID generation and response header injection.

    Args:
        app: Flask application instance
    """

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


def get_app() -> Flask:
    """
    Get or create the Flask application instance (lazy loaded).

    This function provides lazy initialization, allowing the module to be
    imported without triggering model loading. Useful for testing and
    for WSGI servers that need a callable.

    Usage with Gunicorn:
        gunicorn "src.app:get_app()"

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

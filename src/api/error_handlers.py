"""
Error handlers for the Flask application.

This module provides centralized error handling for all exceptions,
converting them into consistent JSON responses.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from flask import Response, g, jsonify
from werkzeug.exceptions import HTTPException

from src.exceptions import APIError
from src.models import ErrorResponse

if TYPE_CHECKING:
    from flask import Flask


def register_error_handlers(app: Flask, logger: logging.Logger) -> None:
    """
    Register error handlers with the Flask application.

    Sets up handlers for:
    - Custom APIError exceptions (and subclasses)
    - Werkzeug HTTP exceptions
    - Unexpected exceptions (500 errors)

    All errors are converted to a consistent JSON format using ErrorResponse.

    Args:
        app: Flask application instance
        logger: Logger for error messages

    Example:
        >>> app = Flask(__name__)
        >>> logger = logging.getLogger("myapp")
        >>> register_error_handlers(app, logger)
    """

    @app.errorhandler(APIError)
    def handle_api_error(error: APIError) -> tuple[Response, int]:
        """
        Handle custom API errors.

        Converts APIError exceptions into JSON responses with appropriate
        status codes and error information.

        Args:
            error: The APIError exception

        Returns:
            Tuple of (JSON response, HTTP status code)
        """
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
        """
        Handle Werkzeug HTTP exceptions.

        Converts standard HTTP exceptions (404, 405, etc.) into
        consistent JSON responses.

        Args:
            error: The HTTPException

        Returns:
            Tuple of (JSON response, HTTP status code)
        """
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
        """
        Handle unexpected exceptions.

        Catches all unhandled exceptions and returns a generic error
        response. The full exception is logged for debugging.

        Args:
            error: The unhandled exception

        Returns:
            Tuple of (JSON response, 500 status code)
        """
        request_id = getattr(g, "request_id", None)
        logger.exception(f"[{request_id}] Unexpected error: {error}")

        response = ErrorResponse(
            error="An internal error occurred",
            error_code="INTERNAL_ERROR",
            request_id=request_id,
        )
        return jsonify(response.model_dump()), 500

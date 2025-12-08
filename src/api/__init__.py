"""API routes and error handlers for the InsightFace API."""

from src.api.error_handlers import register_error_handlers
from src.api.routes import api_blueprint

__all__ = ["api_blueprint", "register_error_handlers"]

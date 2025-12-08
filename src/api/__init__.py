"""API routes and error handlers for the InsightFace API."""

from src.api.error_handlers import register_error_handlers
from src.api.routes import api_blueprint
from src.api.routes_async import api_blueprint_async

__all__ = ["api_blueprint", "api_blueprint_async", "register_error_handlers"]

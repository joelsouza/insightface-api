"""
Pytest configuration and shared fixtures.

This module provides fixtures that are shared across all test modules,
including both Flask (sync) and Quart (async) test clients.
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import pytest_asyncio
from flask import Flask
from flask.testing import FlaskClient
from quart import Quart

# Mock insightface before importing src modules
sys.modules["insightface"] = MagicMock()
sys.modules["insightface.app"] = MagicMock()

from src.app import create_app
from src.app_async import create_async_app
from src.config import Settings
from src.services import InferenceExecutor, ModelManager


@pytest.fixture
def settings() -> Settings:
    """Create test settings with defaults."""
    return Settings(
        port=5001,
        max_content_length=16 * 1024 * 1024,
        max_image_dimension=640,
        detection_threshold=0.5,
        model_name="buffalo_l",
        log_level="WARNING",  # Reduce log noise in tests
    )


@pytest.fixture
def mock_logger() -> MagicMock:
    """Create a mock logger."""
    return MagicMock()


@pytest.fixture
def mock_face() -> MagicMock:
    """Create a mock InsightFace Face object."""
    face = MagicMock()
    face.embedding = np.random.rand(512).astype(np.float32)
    face.bbox = np.array([100, 100, 200, 200], dtype=np.float32)
    face.kps = np.array([
        [120.0, 130.0],
        [180.0, 130.0],
        [150.0, 160.0],
        [130.0, 190.0],
        [170.0, 190.0],
    ])
    face.det_score = 0.95
    face.gender = 1
    face.age = 30
    return face


@pytest.fixture
def mock_face_no_attributes() -> MagicMock:
    """Create a mock face without gender/age attributes."""
    face = MagicMock()
    face.embedding = np.random.rand(512).astype(np.float32)
    face.bbox = np.array([50, 50, 100, 100], dtype=np.float32)
    face.kps = np.array([
        [60.0, 65.0],
        [90.0, 65.0],
        [75.0, 80.0],
        [65.0, 95.0],
        [85.0, 95.0],
    ])
    face.det_score = 0.88
    # Remove gender and age attributes
    del face.gender
    del face.age
    return face


@pytest.fixture
def mock_model_manager(settings: Settings, mock_logger: MagicMock) -> ModelManager:
    """Create a model manager with mocked model."""
    manager = ModelManager(settings=settings, logger=mock_logger)
    manager.model = MagicMock()
    manager.is_loaded = True
    manager.load_time = 1000.0
    return manager


@pytest.fixture
def valid_image_array() -> np.ndarray:
    """Create a valid test image array (100x100 RGB)."""
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def small_image_array() -> np.ndarray:
    """Create a too-small image array (5x5)."""
    return np.random.randint(0, 255, (5, 5, 3), dtype=np.uint8)


@pytest.fixture
def app(settings: Settings) -> Flask:
    """Create test Flask application with mocked model."""
    from unittest.mock import patch

    with patch.object(ModelManager, "load", return_value=True):
        test_app = create_app(settings)
        test_app.config["model_manager"].is_loaded = True
        test_app.config["model_manager"].load_time = 1000.0
        yield test_app


@pytest.fixture
def client(app: Flask) -> FlaskClient:
    """Create test client."""
    return app.test_client()


# ============================================================================
# Async (Quart) Fixtures
# ============================================================================


@pytest_asyncio.fixture
async def async_app(settings: Settings) -> Quart:
    """Create test Quart application with mocked model."""
    from unittest.mock import patch

    with patch.object(ModelManager, "load", return_value=True):
        test_app = create_async_app(settings)
        test_app.config["model_manager"].is_loaded = True
        test_app.config["model_manager"].load_time = 1000.0
        yield test_app
        # Cleanup executor
        executor = test_app.config.get("inference_executor")
        if executor:
            executor.shutdown(wait=False)


@pytest_asyncio.fixture
async def async_client(async_app: Quart):
    """Create async test client."""
    async with async_app.test_client() as client:
        yield client


@pytest.fixture
def mock_inference_executor() -> InferenceExecutor:
    """Create a test inference executor with short timeout."""
    return InferenceExecutor(max_workers=2, timeout=5.0)

"""Tests for configuration module."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config import ExecutionProvider, Settings


class TestSettings:
    """Tests for Settings configuration."""

    def test_default_values(self) -> None:
        """Test default settings values."""
        settings = Settings()
        assert settings.port == 5001
        assert settings.max_image_dimension == 640
        assert settings.detection_threshold == 0.5
        assert settings.model_name == "buffalo_l"
        assert settings.execution_provider == ExecutionProvider.CPU

    def test_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test settings loaded from environment."""
        monkeypatch.setenv("PORT", "8080")
        monkeypatch.setenv("MAX_IMAGE_DIMENSION", "1024")
        monkeypatch.setenv("DETECTION_THRESHOLD", "0.7")
        settings = Settings()
        assert settings.port == 8080
        assert settings.max_image_dimension == 1024
        assert settings.detection_threshold == 0.7

    def test_invalid_port(self) -> None:
        """Test invalid port validation."""
        with pytest.raises(ValidationError):
            Settings(port=0)

        with pytest.raises(ValidationError):
            Settings(port=70000)

    def test_invalid_threshold(self) -> None:
        """Test invalid detection threshold validation."""
        with pytest.raises(ValidationError):
            Settings(detection_threshold=1.5)

        with pytest.raises(ValidationError):
            Settings(detection_threshold=-0.1)

    def test_invalid_dimension(self) -> None:
        """Test invalid dimension validation."""
        with pytest.raises(ValidationError):
            Settings(max_image_dimension=10)  # Below minimum

        with pytest.raises(ValidationError):
            Settings(max_image_dimension=10000)  # Above maximum


class TestExecutionProvider:
    """Tests for ExecutionProvider enum."""

    def test_cpu_provider(self) -> None:
        """Test CPU provider value."""
        assert ExecutionProvider.CPU.value == "CPUExecutionProvider"

    def test_cuda_provider(self) -> None:
        """Test CUDA provider value."""
        assert ExecutionProvider.CUDA.value == "CUDAExecutionProvider"

    def test_tensorrt_provider(self) -> None:
        """Test TensorRT provider value."""
        assert ExecutionProvider.TENSORRT.value == "TensorrtExecutionProvider"

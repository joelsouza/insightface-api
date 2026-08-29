"""
Application configuration and logging setup.

This module provides environment-based configuration using Pydantic Settings
and structured logging configuration for the application.
"""

from __future__ import annotations

import logging
import sys
from enum import Enum

from pydantic import Field
from pydantic_settings import BaseSettings


class ExecutionProvider(str, Enum):
    """
    Supported ONNX execution providers for model inference.

    Attributes:
        CPU: CPU-based inference (default, works everywhere)
        CUDA: NVIDIA GPU acceleration
        TENSORRT: NVIDIA TensorRT optimization
    """

    CPU = "CPUExecutionProvider"
    CUDA = "CUDAExecutionProvider"
    TENSORRT = "TensorrtExecutionProvider"


class Settings(BaseSettings):
    """
    Application configuration loaded from environment variables.

    All settings can be overridden via environment variables with the same name.
    For example, set `PORT=8080` to change the server port.

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
        inference_pool_size: Number of threads for inference executor (default: 4)
        inference_timeout: Timeout for inference operations in seconds (default: 30)
        inference_max_queue: Extra requests allowed to wait for a free thread
            before the API rejects with 503 (default: 16)
        det_model_file: Detection model file name inside the model directory
        rec_model_file: Recognition model file name inside the model directory
        ort_intra_op_threads: ONNX Runtime threads per session (default: 1)
        embedding_decimals: Decimal places kept in returned embeddings (default: 6)
        image_url_allowed_hosts: Comma-separated host patterns allowed for the
            `image_url` input. Empty disables URL input (default: "")
        download_timeout: Timeout for image downloads in seconds (default: 10)
        download_max_concurrency: Maximum parallel image downloads (default: 16)

    Example:
        >>> settings = Settings()
        >>> settings.port
        5001
        >>> settings = Settings(port=8080, log_level="DEBUG")
        >>> settings.port
        8080
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
    inference_pool_size: int = Field(default=4, ge=1, le=32)
    inference_timeout: float = Field(default=30.0, ge=1.0, le=300.0)
    inference_max_queue: int = Field(default=16, ge=0)

    # Model files loaded from `<model_root>/models/<model_name>/`.
    det_model_file: str = Field(default="det_10g.onnx")
    rec_model_file: str = Field(default="w600k_r50.onnx")
    ort_intra_op_threads: int = Field(default=1, ge=1, le=32)

    embedding_decimals: int = Field(default=6, ge=1, le=17)

    # URL input. Empty allowlist keeps the `image_url` input disabled.
    image_url_allowed_hosts: str = Field(default="")
    download_timeout: float = Field(default=10.0, ge=0.1, le=300.0)
    download_max_concurrency: int = Field(default=16, ge=1, le=256)

    model_config = {
        "env_prefix": "",
        "case_sensitive": False,
    }

    @property
    def allowed_image_url_hosts(self) -> list[str]:
        """
        Return the `image_url` host allowlist as a list of fnmatch patterns.

        An empty list means URL input is disabled.

        Example:
            >>> Settings(image_url_allowed_hosts="*.r2.cloudflarestorage.com").allowed_image_url_hosts
            ['*.r2.cloudflarestorage.com']
        """
        return [h.strip().lower() for h in self.image_url_allowed_hosts.split(",") if h.strip()]


def setup_logging(level: str) -> logging.Logger:
    """
    Configure structured logging for the application.

    Sets up a consistent logging format with timestamps, log levels,
    and logger names. Output is directed to stdout for container compatibility.

    Args:
        level: Logging level string (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logger instance for the application

    Example:
        >>> logger = setup_logging("INFO")
        >>> logger.info("Application started")
        2024-01-15 10:30:00 | INFO     | insightface-api | Application started
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    return logging.getLogger("insightface-api")

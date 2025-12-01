"""
InsightFace model lifecycle management.

This module provides the ModelManager class which handles loading, unloading,
and inference with the InsightFace face detection model.
"""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from src.config import Settings
from src.exceptions import ModelNotReadyError

if TYPE_CHECKING:
    pass


@dataclass
class ModelManager:
    """
    Manages the InsightFace model lifecycle.

    Handles model loading, inference, and resource cleanup with proper
    error handling and logging. The model is loaded lazily and can be
    reloaded or unloaded as needed.

    Attributes:
        settings: Application settings for model configuration
        logger: Logger instance for status messages
        model: InsightFace FaceAnalysis instance (None if not loaded)
        is_loaded: Whether the model is ready for inference
        load_time: Unix timestamp when the model was loaded
        load_duration: How long model loading took in seconds

    Example:
        >>> from src.config import Settings, setup_logging
        >>> settings = Settings()
        >>> logger = setup_logging(settings.log_level)
        >>> manager = ModelManager(settings=settings, logger=logger)
        >>> manager.load()
        True
        >>> faces = manager.get_faces(image_array)
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

        Initializes the face analysis model with the configured settings.
        The import is deferred to allow mocking in tests.

        Returns:
            True if loading succeeded, False otherwise

        Note:
            If loading fails, the error message is stored in _initialization_error
            and will be included in subsequent ModelNotReadyError exceptions.
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
        """
        Release model resources and trigger garbage collection.

        This method should be called when shutting down the application
        or when the model needs to be reloaded.
        """
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
            image: BGR image as numpy array (OpenCV format)

        Returns:
            List of detected face objects, each containing:
                - embedding: 512-dimensional face embedding
                - bbox: Bounding box coordinates
                - kps: Facial keypoints
                - det_score: Detection confidence
                - gender: Predicted gender (if available)
                - age: Predicted age (if available)

        Raises:
            ModelNotReadyError: If model is not loaded or initialization failed

        Example:
            >>> import cv2
            >>> image = cv2.imread("photo.jpg")
            >>> faces = manager.get_faces(image)
            >>> print(f"Found {len(faces)} faces")
        """
        if not self.is_loaded or self.model is None:
            raise ModelNotReadyError(
                self._initialization_error or "Model not initialized"
            )
        return self.model.get(image)

    @property
    def uptime(self) -> float:
        """
        Return seconds since model was loaded.

        Returns:
            Seconds since load_time, or 0.0 if not loaded
        """
        if self.load_time is None:
            return 0.0
        return time.time() - self.load_time

    @property
    def initialization_error(self) -> Optional[str]:
        """
        Return the initialization error message if loading failed.

        Returns:
            Error message string, or None if no error occurred
        """
        return self._initialization_error

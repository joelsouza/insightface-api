"""
InsightFace model lifecycle management.

This module provides the ModelManager class which handles loading, unloading,
and inference with the InsightFace face detection model.

Instead of `insightface.app.FaceAnalysis`, the manager builds the two models it
actually needs (detection and recognition) directly. This gives two things
`FaceAnalysis` cannot:

- Control over the ONNX Runtime session options. `FaceAnalysis` forwards only
  `providers`, so every session defaults to one intra-op thread per CPU core.
  With a thread pool on top, that oversubscribes the CPU badly.
- Only two ONNX sessions instead of five. `FaceAnalysis` creates a session for
  every `.onnx` file in the model directory and then throws three away.
"""

from __future__ import annotations

import gc
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np

from src.config import Settings
from src.exceptions import ModelNotReadyError
from src.instrumentation import background_task

# ArcFace models always take 112x112 aligned crops.
RECOGNITION_IMAGE_SIZE = 112


def _make_session(model_file: str, settings: Settings) -> Any:
    """
    Build an ONNX Runtime session tuned for a thread pool of workers.

    Each request already runs on its own pool thread, so a session must not
    spawn threads of its own. Spinning is disabled too: idle intra-op threads
    busy-wait by default and steal CPU from the other sessions. Logging is
    limited to errors, because ONNX Runtime warns on every batched
    recognition call and the write costs real throughput.

    Args:
        model_file: Path to the `.onnx` file
        settings: Application settings (thread count, execution provider)

    Returns:
        Configured `onnxruntime.InferenceSession`
    """
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = settings.ort_intra_op_threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")

    # Errors only. The recognition model declares a fixed output shape of
    # (1, 512), so every batched call logs a shape warning to stderr. Writing
    # that line per request costs about a third of the throughput.
    options.log_severity_level = 3

    return ort.InferenceSession(
        model_file,
        sess_options=options,
        providers=[settings.execution_provider.value],
    )


class _FaceEngine:
    """
    Minimal replacement for `insightface.app.FaceAnalysis`.

    Runs detection, then a single batched recognition call for all faces in
    the image. `w600k_r50.onnx` has a dynamic batch dimension, so N faces cost
    one ONNX call instead of N.

    Attributes:
        det: Detection model (`RetinaFace`)
        rec: Recognition model (`ArcFaceONNX`)
        det_size: Detection input size in pixels (square)
    """

    def __init__(
        self,
        det: Any,
        rec: Any,
        face_cls: Any,
        face_align: Any,
        det_size: int,
    ) -> None:
        self.det = det
        self.rec = rec
        self._face_cls = face_cls
        self._face_align = face_align
        self.det_size = det_size

    def get(self, img: np.ndarray, max_num: int = 0) -> list[Any]:
        """
        Detect faces and attach an embedding to each one.

        Args:
            img: BGR image as numpy array
            max_num: Maximum faces to return (0 means no limit)

        Returns:
            List of `insightface.app.common.Face` objects
        """
        bboxes, kpss = self.det.detect(img, max_num=max_num, metric="default")

        if bboxes.shape[0] == 0:
            return []

        faces = []
        crops = []
        for i in range(bboxes.shape[0]):
            kps = kpss[i] if kpss is not None else None
            faces.append(
                self._face_cls(
                    bbox=bboxes[i, 0:4],
                    kps=kps,
                    det_score=bboxes[i, 4],
                )
            )
            if kps is not None:
                crops.append(
                    self._face_align.norm_crop(
                        img, landmark=kps, image_size=RECOGNITION_IMAGE_SIZE
                    )
                )

        if crops:
            # One ONNX call for every face in the image.
            feats = self.rec.get_feat(crops)
            for face, feat in zip(faces, feats):
                face.embedding = feat.flatten()

        return faces

    def warmup(self) -> None:
        """
        Run one dummy inference per session.

        The first real call otherwise pays for arena allocation and kernel
        selection, which adds 0.5-2s to the first request.
        """
        blank_det = np.zeros((self.det_size, self.det_size, 3), dtype=np.uint8)
        self.det.detect(blank_det, max_num=0, metric="default")

        blank_rec = np.zeros(
            (RECOGNITION_IMAGE_SIZE, RECOGNITION_IMAGE_SIZE, 3), dtype=np.uint8
        )
        self.rec.get_feat([blank_rec])


@dataclass
class ModelManager:
    """
    Manages the InsightFace model lifecycle.

    Handles model loading, inference, and resource cleanup with proper
    error handling and logging. The model is loaded at application startup
    and can be reloaded or unloaded as needed.

    Attributes:
        settings: Application settings for model configuration
        logger: Logger instance for status messages
        model: Face engine instance (None if not loaded)
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
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @background_task(name="ModelLoad", group="Startup")
    def load(self) -> bool:
        """
        Load the detection and recognition models.

        Builds one ONNX session per model with explicit thread settings, then
        runs a warm-up inference. The import is deferred to allow mocking in
        tests. Thread-safe via lock.

        Returns:
            True if loading succeeded, False otherwise

        Note:
            If loading fails, the error message is stored in _initialization_error
            and will be included in subsequent ModelNotReadyError exceptions.
        """
        with self._lock:
            # Check if already loaded (another thread may have loaded it)
            if self.is_loaded:
                return True

            # Defer import to allow mocking in tests
            import insightface

            # OpenCV spawns one thread per core by default, which competes
            # with the inference pool for the same CPUs.
            cv2.setNumThreads(1)

            self.logger.info(
                f"Loading InsightFace model '{self.settings.model_name}'..."
            )
            start_time = time.perf_counter()

            try:
                # Falls back to downloading the model bundle when the files
                # are missing (local development; the image bakes them in).
                model_dir = insightface.utils.ensure_available(
                    "models",
                    self.settings.model_name,
                    root=self.settings.model_root,
                )
                det_path = os.path.join(model_dir, self.settings.det_model_file)
                rec_path = os.path.join(model_dir, self.settings.rec_model_file)

                det_size = self.settings.max_image_dimension

                # insightface ships no type information for these.
                zoo = insightface.model_zoo  # type: ignore[attr-defined]
                common = insightface.app.common  # type: ignore[attr-defined]

                det = zoo.RetinaFace(
                    model_file=det_path,
                    session=_make_session(det_path, self.settings),
                )
                det.prepare(
                    0,
                    input_size=(det_size, det_size),
                    det_thresh=self.settings.detection_threshold,
                )

                rec = zoo.ArcFaceONNX(
                    model_file=rec_path,
                    session=_make_session(rec_path, self.settings),
                )
                rec.prepare(0)

                face_align = getattr(insightface.utils, "face_align", None)
                if face_align is None:  # pragma: no cover - defensive
                    from insightface.utils import face_align  # type: ignore[no-redef]

                self.model = _FaceEngine(
                    det=det,
                    rec=rec,
                    face_cls=common.Face,
                    face_align=face_align,
                    det_size=det_size,
                )
                self.model.warmup()

                self.load_duration = time.perf_counter() - start_time
                self.load_time = time.time()
                self.is_loaded = True
                self._initialization_error = None

                self.logger.info(
                    f"Model loaded successfully in {self.load_duration:.2f}s "
                    f"(det={self.settings.det_model_file}, "
                    f"rec={self.settings.rec_model_file}, "
                    f"intra_op_threads={self.settings.ort_intra_op_threads})"
                )
                return True

            except Exception as e:
                self.load_duration = time.perf_counter() - start_time
                self.is_loaded = False
                self._initialization_error = str(e)
                self.logger.exception(f"Failed to load model: {e}")
                return False

    def unload(self) -> None:
        """
        Release model resources and trigger garbage collection.

        This method should be called when shutting down the application
        or when the model needs to be reloaded. Thread-safe via lock.
        """
        with self._lock:
            if self.model is not None:
                self.logger.info("Unloading model and releasing resources...")
                del self.model
                self.model = None
                self.is_loaded = False
                self.load_time = None
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

"""
Performance instrumentation module for InsightFace API.
Provides custom metrics and tracing for NewRelic APM.
"""

import time
import functools
import psutil
import os
from contextlib import contextmanager
from typing import Optional, Dict, Any, Callable

try:
    import newrelic.agent as nr_agent
    NEWRELIC_AVAILABLE = True
except ImportError:
    NEWRELIC_AVAILABLE = False
    nr_agent = None


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def record_custom_metric(name: str, value: float) -> None:
    """Record a custom metric to NewRelic."""
    if NEWRELIC_AVAILABLE and nr_agent:
        nr_agent.record_custom_metric(f"Custom/{name}", value)


def record_custom_event(event_type: str, params: Dict[str, Any]) -> None:
    """Record a custom event to NewRelic."""
    if NEWRELIC_AVAILABLE and nr_agent:
        nr_agent.record_custom_event(event_type, params)


def add_custom_attribute(key: str, value: Any) -> None:
    """Add a custom attribute to the current transaction."""
    if NEWRELIC_AVAILABLE and nr_agent:
        nr_agent.add_custom_attribute(key, value)


@contextmanager
def trace_segment(name: str, group: str = "Custom", record_memory: bool = True):
    """
    Context manager for tracing a code segment with timing and memory metrics.

    Usage:
        with trace_segment("decode_image", "ImageProcessing"):
            # code to trace
    """
    start_time = time.perf_counter()
    start_memory = get_memory_usage_mb() if record_memory else 0

    try:
        if NEWRELIC_AVAILABLE and nr_agent:
            with nr_agent.FunctionTrace(name=name, group=group):
                yield
        else:
            yield
    finally:
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        record_custom_metric(f"{group}/{name}/duration_ms", elapsed_ms)

        if record_memory:
            end_memory = get_memory_usage_mb()
            memory_delta = end_memory - start_memory
            record_custom_metric(f"{group}/{name}/memory_delta_mb", memory_delta)
            record_custom_metric(f"{group}/{name}/memory_peak_mb", end_memory)


def trace_function(name: Optional[str] = None, group: str = "Custom", record_memory: bool = True):
    """
    Decorator for tracing function execution with timing and memory metrics.

    Usage:
        @trace_function("process_face", "FaceDetection")
        def process_face(image):
            ...
    """
    def decorator(func: Callable) -> Callable:
        trace_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with trace_segment(trace_name, group, record_memory):
                return func(*args, **kwargs)

        return wrapper
    return decorator


class PipelineTracer:
    """
    Traces an entire processing pipeline with step-by-step metrics.

    Usage:
        tracer = PipelineTracer("face_analysis")

        with tracer.step("decode"):
            # decode image

        with tracer.step("detect"):
            # detect faces

        tracer.finish()
    """

    def __init__(self, pipeline_name: str):
        self.pipeline_name = pipeline_name
        self.start_time = time.perf_counter()
        self.start_memory = get_memory_usage_mb()
        self.steps: list = []
        self.current_step: Optional[str] = None
        self.step_start_time: float = 0
        self.step_start_memory: float = 0

        add_custom_attribute("pipeline_name", pipeline_name)

    @contextmanager
    def step(self, step_name: str):
        """Trace a single step in the pipeline."""
        step_start = time.perf_counter()
        step_memory_start = get_memory_usage_mb()

        try:
            if NEWRELIC_AVAILABLE and nr_agent:
                with nr_agent.FunctionTrace(name=step_name, group=f"Pipeline/{self.pipeline_name}"):
                    yield
            else:
                yield
        finally:
            elapsed_ms = (time.perf_counter() - step_start) * 1000
            memory_end = get_memory_usage_mb()
            memory_delta = memory_end - step_memory_start

            step_data = {
                "step": step_name,
                "duration_ms": elapsed_ms,
                "memory_delta_mb": memory_delta,
                "memory_after_mb": memory_end
            }
            self.steps.append(step_data)

            record_custom_metric(f"Pipeline/{self.pipeline_name}/{step_name}/duration_ms", elapsed_ms)
            record_custom_metric(f"Pipeline/{self.pipeline_name}/{step_name}/memory_delta_mb", memory_delta)

    def finish(self) -> Dict[str, Any]:
        """Finish tracing and record summary metrics."""
        total_duration_ms = (time.perf_counter() - self.start_time) * 1000
        final_memory = get_memory_usage_mb()
        total_memory_delta = final_memory - self.start_memory

        summary = {
            "pipeline": self.pipeline_name,
            "total_duration_ms": total_duration_ms,
            "total_memory_delta_mb": total_memory_delta,
            "final_memory_mb": final_memory,
            "step_count": len(self.steps),
            "steps": self.steps
        }

        record_custom_metric(f"Pipeline/{self.pipeline_name}/total_duration_ms", total_duration_ms)
        record_custom_metric(f"Pipeline/{self.pipeline_name}/total_memory_delta_mb", total_memory_delta)
        record_custom_metric(f"Pipeline/{self.pipeline_name}/step_count", len(self.steps))

        record_custom_event("PipelineExecution", {
            "pipeline_name": self.pipeline_name,
            "total_duration_ms": total_duration_ms,
            "total_memory_delta_mb": total_memory_delta,
            "step_count": len(self.steps)
        })

        for step_data in self.steps:
            add_custom_attribute(f"step_{step_data['step']}_duration_ms", step_data["duration_ms"])

        return summary


class RequestMetrics:
    """
    Captures and records metrics for an API request.
    """

    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.start_time = time.perf_counter()
        self.start_memory = get_memory_usage_mb()
        self.metrics: Dict[str, Any] = {}

        add_custom_attribute("endpoint", endpoint)
        add_custom_attribute("request_start_memory_mb", self.start_memory)

    def set(self, key: str, value: Any) -> None:
        """Set a metric value."""
        self.metrics[key] = value
        add_custom_attribute(key, value)

    def increment(self, key: str, amount: int = 1) -> None:
        """Increment a counter metric."""
        self.metrics[key] = self.metrics.get(key, 0) + amount
        add_custom_attribute(key, self.metrics[key])

    def finish(self) -> Dict[str, Any]:
        """Finish capturing metrics and record to NewRelic."""
        elapsed_ms = (time.perf_counter() - self.start_time) * 1000
        final_memory = get_memory_usage_mb()
        memory_delta = final_memory - self.start_memory

        self.metrics.update({
            "total_duration_ms": elapsed_ms,
            "memory_delta_mb": memory_delta,
            "final_memory_mb": final_memory
        })

        record_custom_metric(f"Request/{self.endpoint}/duration_ms", elapsed_ms)
        record_custom_metric(f"Request/{self.endpoint}/memory_delta_mb", memory_delta)

        record_custom_event("APIRequest", {
            "endpoint": self.endpoint,
            **self.metrics
        })

        return self.metrics


def record_face_detection_metrics(
    image_width: int,
    image_height: int,
    faces_detected: int,
    detection_time_ms: float
) -> None:
    """Record specific metrics for face detection operations."""
    pixels = image_width * image_height
    pixels_per_ms = pixels / detection_time_ms if detection_time_ms > 0 else 0

    record_custom_metric("FaceDetection/image_pixels", pixels)
    record_custom_metric("FaceDetection/faces_detected", faces_detected)
    record_custom_metric("FaceDetection/detection_time_ms", detection_time_ms)
    record_custom_metric("FaceDetection/pixels_per_ms", pixels_per_ms)

    add_custom_attribute("image_width", image_width)
    add_custom_attribute("image_height", image_height)
    add_custom_attribute("faces_detected", faces_detected)

    record_custom_event("FaceDetection", {
        "image_width": image_width,
        "image_height": image_height,
        "image_pixels": pixels,
        "faces_detected": faces_detected,
        "detection_time_ms": detection_time_ms,
        "pixels_per_ms": pixels_per_ms
    })


def record_embedding_metrics(
    face_count: int,
    embedding_dim: int,
    embedding_time_ms: float
) -> None:
    """Record specific metrics for face embedding operations."""
    faces_per_second = (face_count / embedding_time_ms * 1000) if embedding_time_ms > 0 else 0

    record_custom_metric("FaceEmbedding/face_count", face_count)
    record_custom_metric("FaceEmbedding/embedding_dim", embedding_dim)
    record_custom_metric("FaceEmbedding/embedding_time_ms", embedding_time_ms)
    record_custom_metric("FaceEmbedding/faces_per_second", faces_per_second)

    record_custom_event("FaceEmbedding", {
        "face_count": face_count,
        "embedding_dim": embedding_dim,
        "embedding_time_ms": embedding_time_ms,
        "faces_per_second": faces_per_second
    })


def record_comparison_metrics(
    comparison_count: int,
    comparison_time_ms: float,
    match_found: bool,
    similarity_score: Optional[float] = None
) -> None:
    """Record specific metrics for face comparison operations."""
    comparisons_per_second = (comparison_count / comparison_time_ms * 1000) if comparison_time_ms > 0 else 0

    record_custom_metric("FaceComparison/comparison_count", comparison_count)
    record_custom_metric("FaceComparison/comparison_time_ms", comparison_time_ms)
    record_custom_metric("FaceComparison/comparisons_per_second", comparisons_per_second)

    if similarity_score is not None:
        record_custom_metric("FaceComparison/similarity_score", similarity_score)

    record_custom_event("FaceComparison", {
        "comparison_count": comparison_count,
        "comparison_time_ms": comparison_time_ms,
        "comparisons_per_second": comparisons_per_second,
        "match_found": match_found,
        "similarity_score": similarity_score
    })


def initialize_newrelic(config_file: str = "newrelic.ini", environment: str = "production") -> bool:
    """
    Initialize NewRelic agent.

    Should be called at application startup before any other code runs.
    Returns True if initialization was successful.
    """
    if not NEWRELIC_AVAILABLE:
        print("NewRelic package not available. Instrumentation disabled.")
        return False

    try:
        nr_agent.initialize(config_file, environment)
        print(f"NewRelic initialized with config: {config_file}, environment: {environment}")
        return True
    except Exception as e:
        print(f"Failed to initialize NewRelic: {e}")
        return False

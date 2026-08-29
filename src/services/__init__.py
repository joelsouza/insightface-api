"""Service layer for the InsightFace API."""

from src.services.image import (
    decode_image,
    extract_face_data,
    validate_image_dimensions,
    validate_image_file,
    verify_magic_bytes,
)
from src.services.image_fetch import (
    fetch_image,
    is_public_address,
    resolve_public_addresses,
    validate_image_url,
)
from src.services.inference_executor import InferenceExecutor
from src.services.model_manager import ModelManager
from src.services.pipeline import PipelineResult, process_image

__all__ = [
    "InferenceExecutor",
    "ModelManager",
    "PipelineResult",
    "decode_image",
    "extract_face_data",
    "fetch_image",
    "is_public_address",
    "process_image",
    "resolve_public_addresses",
    "validate_image_dimensions",
    "validate_image_file",
    "validate_image_url",
    "verify_magic_bytes",
]

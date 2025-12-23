# AGENTS.md

This file provides guidance to Coding Agents when working with code in this repository.

## Commands

```bash
# Development (Flask sync server)
python -m src.app

# Production (Uvicorn async server)
./bin/start

# Tests
pytest tests/
pytest tests/test_api.py -v              # Single test file
pytest tests/test_api.py::test_name -v   # Single test

# Docker
docker build -t insightface-api .
docker run -p 5001:5001 insightface-api
```

## Architecture

This is a REST API for face detection/embedding extraction using InsightFace.

### Dual Implementation
- **Flask sync** (`src/app.py`, `src/api/routes.py`) - Development/simple deployments
- **Quart async** (`src/app_async.py`, `src/api/routes_async.py`) - **Production default**

Both use the factory pattern via `get_app()` callable for Uvicorn compatibility.

### Key Design Decisions

**Single-process model loading**: The InsightFace model (~500MB buffalo_l) loads once and is held in memory. This is intentional—model loading is expensive and the weights are shared across requests.

**ThreadPoolExecutor for inference**: The async version uses a 4-worker thread pool (`src/services/inference_executor.py`) to run CPU-bound inference without blocking the event loop. ONNX runtime releases the GIL during compute, enabling true parallelism.

**Lazy initialization**: Model loads on first request, not at startup. This enables faster container starts and allows health checks before model readiness.

### Request Flow
1. Image upload (multipart form-data) → `src/api/routes_async.py`
2. Validation & decode → `src/services/image.py`
3. Inference via thread pool → `src/services/model_manager.py`
4. Return 512-dim embeddings + metadata (bbox, landmarks, age, gender)

## Configuration

All config via Pydantic BaseSettings in `src/config/settings.py`:

```bash
PORT=5001
MAX_IMAGE_DIMENSION=640          # det_size for model
DETECTION_THRESHOLD=0.5          # Face confidence 0-1
EXECUTION_PROVIDER=CPUExecutionProvider  # or CUDAExecutionProvider, TensorrtExecutionProvider
LOG_LEVEL=INFO
```

## Testing

Tests mock InsightFace at module level in `tests/conftest.py` to avoid loading the actual model (~500MB). The mock is applied before any imports from src modules.

Key fixtures:
- `mock_face` - Returns mock face objects with numpy arrays
- `app` / `client` - Flask test client with mocked model

When writing new tests, ensure InsightFace mocking happens before importing app modules.

## Exception Hierarchy

Custom exceptions in `src/exceptions/errors.py`:
- `ImageDecodeError` (400) - cv2.imdecode failure
- `ImageValidationError` (400) - Dimension/size issues
- `RequestValidationError` (400) - Missing/invalid request
- `ModelNotReadyError` (503) - Model not initialized

All inherit from `APIError` with `status_code` and `error_code` attributes.

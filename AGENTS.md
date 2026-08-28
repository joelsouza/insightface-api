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

# Load test (server must already be running)
bin/bench.py --image photo.jpg --concurrency 40 --requests 200

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

**Single-process model loading**: The two model files (~280MB of `buffalo_l`) load once at startup and are held in memory. This is intentional—model loading is expensive and the weights are shared across requests. `create_async_app` calls `ModelManager.load()` directly, so the process is not ready to serve until the model is in memory.

**No `FaceAnalysis`**: `src/services/model_manager.py` builds `RetinaFace` and `ArcFaceONNX` itself. `FaceAnalysis` gives no way to pass `SessionOptions`, and it creates an ONNX session for every `.onnx` file in the model directory before throwing three of them away.

**One ONNX thread per session**: Each session runs with `intra_op_num_threads=1`, spinning off, and errors-only logging. The thread pool is the only source of parallelism, so `INFERENCE_POOL_SIZE` alone decides how many CPUs inference uses. Use `bin/bench.py` to confirm the split on a given host before changing `ORT_INTRA_OP_THREADS`.

**Batched recognition**: `w600k_r50.onnx` has a dynamic batch dimension. All faces in an image are cropped and embedded in one ONNX call rather than one call per face.

**One thread-pool job per request**: `src/services/pipeline.py` runs decode, validation, inference, and extraction together in a worker thread. Decoding a 12 MP JPEG takes 80-150ms and must never run on the event loop.

**Bounded in-flight requests**: `InferenceExecutor` admits `INFERENCE_POOL_SIZE + INFERENCE_MAX_QUEUE` requests and rejects the rest with 503 and `Retry-After`. `asyncio.wait_for` cannot cancel a running thread, so unbounded queueing only makes every request time out.

### Request Flow
1. Input → `src/api/routes_async.py`: multipart `image_file`, or `image_url` (JSON body or form field). Exactly one is required.
2. A URL input is downloaded on the event loop → `src/services/image_fetch.py` (HTTPS only, host allowlist, size-bounded)
3. One thread-pool job → `src/services/pipeline.py`: decode, validate, detect + embed, extract
4. Return 512-dim embeddings + metadata (bbox, landmarks, age, gender)

The Flask sync app still uses the older step-by-step path (`decode_image`, then `get_faces`) in `src/services/image.py`. That path is unchanged.

## Configuration

All config via Pydantic BaseSettings in `src/config/settings.py`:

```bash
PORT=5001
MAX_IMAGE_DIMENSION=640          # det_size for model
DETECTION_THRESHOLD=0.5          # Face confidence 0-1
EXECUTION_PROVIDER=CPUExecutionProvider  # or CUDAExecutionProvider, TensorrtExecutionProvider
LOG_LEVEL=INFO

# Throughput
INFERENCE_POOL_SIZE=4            # Inference threads; set to the CPU count
ORT_INTRA_OP_THREADS=1           # ONNX threads per session
INFERENCE_MAX_QUEUE=16           # Extra requests allowed to wait before 503
INFERENCE_TIMEOUT=30             # Seconds before a job gives up

# Models (file names inside <MODEL_ROOT>/models/<MODEL_NAME>/)
DET_MODEL_FILE=det_10g.onnx
REC_MODEL_FILE=w600k_r50.onnx

# Response size
EMBEDDING_DECIMALS=6             # float32 prints 17 digits; 6 halves the payload

# URL input (disabled while the allowlist is empty)
IMAGE_URL_ALLOWED_HOSTS=         # Comma-separated fnmatch patterns
DOWNLOAD_TIMEOUT=10
DOWNLOAD_MAX_CONCURRENCY=16
```

`IMAGE_URL_ALLOWED_HOSTS` is a security boundary, not a convenience setting. `src/services/image_fetch.py` is the trust boundary against server-side request forgery, and it applies three checks that all have to stay:

1. HTTPS only, host must match the allowlist, IP-literal hosts refused.
2. **Every** address the host resolves to must be public. An allowlisted name can point at loopback, a private range, or a cloud metadata service, so checking the host text alone is not enough.
3. The connection is **pinned** to the address that was checked, with the original host in the `Host` header and in the TLS server name. Resolving and then connecting by name would leave a window for a second DNS answer to swap in a private address.

Redirects are not followed, because a redirect target would skip all three.

## Testing

Tests mock InsightFace **and onnxruntime** at module level in `tests/conftest.py` to avoid loading the actual model. Both mocks are applied before any imports from src modules.

Key fixtures:
- `mock_face` - Returns mock face objects with numpy arrays
- `app` / `client` - Flask test client with mocked model
- `async_app` / `async_client` - Quart test client with mocked model
- `jpeg_bytes` / `large_jpeg_bytes` / `png_bytes` - Real encoded images

When writing new tests, ensure the mocking happens before importing app modules.

Two notes on the async tests:
- `app.test_client()` does **not** run the `before_serving` hooks. A test that needs the shared HTTP client (the `image_url` path) must use `async with app.test_app()`.
- Patch `src.services.model_manager._make_session` to control model loading, not `insightface.app.FaceAnalysis`.

## Exception Hierarchy

Custom exceptions in `src/exceptions/errors.py`:
- `ImageDecodeError` (400) - cv2.imdecode failure
- `ImageValidationError` (400) - Dimension/size issues
- `RequestValidationError` (400) - Missing/invalid request
- `ModelNotReadyError` (503) - Model not initialized
- `ServiceOverloadedError` (503) - No free inference slot; response carries `Retry-After`
- `ImageDownloadError` (502) - `image_url` could not be fetched
- `InferenceTimeoutError` (504) - Inference exceeded `INFERENCE_TIMEOUT`

All inherit from `APIError` with `status_code` and `error_code` attributes.

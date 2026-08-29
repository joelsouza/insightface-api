# InsightFace API

A REST API for face detection and facial embedding extraction. Send an image, get back detected faces with 512-dimensional embeddings for face recognition and verification tasks.

## What It Does

- Detects faces in images and returns bounding boxes
- Extracts facial embeddings for face recognition and matching
- Provides facial landmarks (eyes, nose, mouth corners)
- Returns confidence scores for detected faces
- Predicts age and gender attributes

## Getting Started

### Requirements

- Python 3.12+
- Docker (optional)

### Installation

```bash
pip install -r requirements.txt
```

### Run Locally

```bash
# Development server
python -m src.app

# Production with Uvicorn
./bin/start
```

The API will be available at `http://localhost:5001`

## Configuration

Set environment variables to customize behavior:

```bash
PORT=5001                              # Server port
LOG_LEVEL=INFO                         # Logging verbosity
MAX_IMAGE_DIMENSION=640                # Image resolution for detection
DETECTION_THRESHOLD=0.5                # Face confidence threshold (0-1)
EXECUTION_PROVIDER=CPUExecutionProvider  # CPUExecutionProvider, CUDAExecutionProvider, TensorrtExecutionProvider
NEW_RELIC_LICENSE_KEY=your_key         # Optional: Enable New Relic APM
```

### Observability

The production Quart application is wrapped by the New Relic agent. It names
transactions by route and ignores the `/up` Docker probe. Each `/represent`
request emits one `FaceRepresent` event with request, image, pipeline, queue,
and response data. New Relic forwards application logs with `request_id` and
`input_mode` fields.

The dashboard definition is in `newrelic/dashboard.json`. Deployment markers
are sent by `.github/workflows/deploy-marker.yml` when `NEW_RELIC_API_KEY` is
set as a repository secret. The marker time is the GitHub push time. Coolify
deploys the application outside GitHub Actions.

### Throughput

```bash
INFERENCE_POOL_SIZE=4                  # Inference threads; set this to the CPU count
ORT_INTRA_OP_THREADS=1                 # ONNX threads per session
INFERENCE_MAX_QUEUE=16                 # Extra requests allowed to wait before 503
INFERENCE_TIMEOUT=30                   # Seconds before a request gives up
EMBEDDING_DECIMALS=6                   # Decimals kept in the returned embedding
```

Each ONNX session runs single-threaded, so `INFERENCE_POOL_SIZE` is the only
knob that decides how many CPUs inference uses. Measure before changing the
split:

```bash
bin/bench.py --image photo.jpg --concurrency 40 --requests 200
```

### Downloading images by URL

URL input is off until you list the hosts you trust:

```bash
IMAGE_URL_ALLOWED_HOSTS="*.r2.cloudflarestorage.com,*.s3.amazonaws.com"
DOWNLOAD_TIMEOUT=10                    # Seconds
DOWNLOAD_MAX_CONCURRENCY=16            # Parallel downloads
```

Only HTTPS URLs are fetched and the host must match a pattern. Beyond that,
every address the host resolves to must be a public address, and the
connection is pinned to the address that was checked. An allowlisted name
therefore cannot reach loopback, a private range, or a cloud metadata service,
and it cannot be swapped for one by a second DNS answer. Redirects are not
followed, and the body is dropped once it passes the upload limit.

A pattern of `*` is still a bad idea, but it no longer exposes internal
services.

## API Endpoints

### `GET /`
Returns API info and available endpoints.

### `GET /health`
Detailed health check with model status and uptime.

```bash
curl http://localhost:5001/health
```

### `POST /represent`
Detect faces and extract embeddings. Send either an image file or a URL.

```bash
# Upload a file
curl -X POST -F "image_file=@image.jpg" http://localhost:5001/represent

# Or let the API download it (requires IMAGE_URL_ALLOWED_HOSTS)
curl -X POST -H "content-type: application/json" \
  -d '{"image_url": "https://bucket.r2.cloudflarestorage.com/image.jpg"}' \
  http://localhost:5001/represent
```

**Response:**
```json
{
  "embeddings": [
    {
      "embedding": [0.123456, -0.456789, "..."],
      "bbox": [100, 150, 200, 300],
      "keypoints": [[110, 160], [190, 170], "..."],
      "det_score": 0.99,
      "age": 28,
      "gender": 1
    }
  ],
  "faces_detected": 1,
  "request_id": "abc12345",
  "processing_time_ms": 45.2
}
```

Bounding boxes and keypoints are always in original image pixels.

**Error responses** carry `error`, `error_code`, and `request_id`:

The stable error codes are `REQUEST_INVALID`, `IMAGE_DECODE_FAILED`,
`IMAGE_VALIDATION_FAILED`, `MODEL_NOT_READY`, `IMAGE_DOWNLOAD_FAILED`,
`OVERLOADED`, and `INFERENCE_TIMEOUT`.

| Status | Meaning |
|--------|---------|
| 400 | Missing, invalid, or disallowed input |
| 502 | The `image_url` could not be downloaded |
| 503 | Model not ready, or the server is at capacity (`Retry-After` header) |
| 504 | Inference exceeded `INFERENCE_TIMEOUT` |

A batch client should treat 503 as "retry after the number of seconds in
`Retry-After`", not as a failure.

## Docker

Build and run with Docker:

```bash
docker build -t insightface-api .
docker run -p 5001:5001 insightface-api
```

## Testing

```bash
pytest tests/
```

## Performance Notes

- Models load at startup, not on the first request. The container image bakes
  them in, so a cold start does not download anything.
- Large JPEGs (longest side at 4x `MAX_IMAGE_DIMENSION` or more) are decoded at
  half resolution. Detection resizes them anyway.
- All faces in an image are embedded in a single batched ONNX call.
- Under a burst the server admits `INFERENCE_POOL_SIZE + INFERENCE_MAX_QUEUE`
  requests and rejects the rest immediately with 503, instead of letting every
  request queue until it times out.
- GPU acceleration available via CUDA or TensorRT providers.

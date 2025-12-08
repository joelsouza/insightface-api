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

# Production with Gunicorn
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

## API Endpoints

### `GET /`
Returns API info and available endpoints.

### `GET /health`
Detailed health check with model status and uptime.

```bash
curl http://localhost:5001/health
```

### `POST /represent`
Detect faces and extract embeddings from an image file.

```bash
curl -X POST -F "file=@image.jpg" http://localhost:5001/represent
```

**Response:**
```json
{
  "results": [
    {
      "embedding": [0.123, -0.456, ...],
      "bbox": [100, 150, 200, 300],
      "det_score": 0.99,
      "landmarks": [[110, 160], [190, 170], ...],
      "age": 28,
      "gender": "M"
    }
  ],
  "request_id": "uuid-string",
  "processing_time_ms": 45
}
```

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

- First request takes longer as the model loads (~2-3 seconds)
- GPU acceleration available via CUDA or TensorRT providers
- Typical inference time: 50-100ms per image on CPU

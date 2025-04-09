# InsightFace API

A simple API built with Flask for facial embedding extraction using InsightFace.

## Description

This API uses the InsightFace library to detect faces in images and extract facial embedding vectors, which can be used for facial recognition, identity verification, and other related use cases.

## Features

- Facial detection in images
- Facial embedding extraction
- Additional information such as bounding box, facial keypoints, and detection scores
- Support for base64 encoded images

## Requirements

- Python 3.x
- Flask
- InsightFace
- OpenCV
- Other requirements listed in `requirements.txt`

## Installation

### Using Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-username/insightface-api.git
cd insightface-api

# Build the Docker image
docker build -t insightface-api .

# Run the container
docker run -p 5001:5001 insightface-api
```

### Local Installation

```bash
# Clone the repository
git clone https://github.com/your-username/insightface-api.git
cd insightface-api

# Install dependencies
pip install -r requirements.txt

# Run the API
python api.py
```

## API Usage

### Endpoint: `/represent`

This endpoint receives a base64 encoded image and returns facial embeddings and other information for each detected face.

**Method:** POST

**Parameters:**
- `image_data`: Base64 encoded image

**Request Example:**

```python
import requests
import base64

# Read the image and encode in base64
with open("image.jpg", "rb") as img_file:
    img_data = base64.b64encode(img_file.read()).decode('utf-8')

# Send request to the API
response = requests.post(
    "http://localhost:5001/represent",
    data={"image_data": img_data}
)

# Process the response
result = response.json()
print(f"Faces detected: {len(result['embeddings'])}")
```

**Response Example:**

```json
{
  "embeddings": [
    {
      "embedding": [0.023, -0.045, 0.012, ...],
      "bbox": [142.5, 187.2, 375.8, 420.3],
      "kps": [[200.1, 250.3], [240.4, 251.2], ...],
      "det_score": 0.998
    },
    {
      "embedding": [-0.011, 0.026, 0.157, ...],
      "bbox": [512.6, 189.5, 745.1, 422.7],
      "kps": [[580.3, 253.1], [620.8, 254.6], ...],
      "det_score": 0.987
    }
  ]
}
```

## Configuration

By default, the API uses only CPU for inference. If you have a GPU available, modify the `providers` variable in `api.py`:

```python
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

## Notes

- The InsightFace model used is 'buffalo_l', which includes detection and recognition modules.
- The default port is 5001, but it can be configured through the PORT environment variable.

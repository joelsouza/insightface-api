#!/bin/bash
#
# Startup script for InsightFace API v2
#
# This script starts the production server using Gunicorn with optimized settings
# for face detection workloads (mixed CPU-bound and I/O-bound operations).
#
# Environment Variables:
#   PORT                  - Server port (default: 5001)
#   WORKERS               - Number of Gunicorn workers (default: 4)
#   THREADS               - Threads per worker (default: 1)
#   TIMEOUT               - Request timeout in seconds (default: 60)
#   MAX_REQUESTS          - Requests before worker restart (default: 100)
#   LOG_LEVEL             - Logging level (default: info)
#   MAX_IMAGE_DIMENSION   - Max image size for detection (default: 640)
#   DETECTION_THRESHOLD   - Face detection threshold (default: 0.5)
#

set -e

# Server configuration
PORT="${PORT:-5001}"
WORKERS="${WORKERS:-4}"
THREADS="${THREADS:-1}"
TIMEOUT="${TIMEOUT:-60}"
MAX_REQUESTS="${MAX_REQUESTS:-100}"
MAX_REQUESTS_JITTER="${MAX_REQUESTS_JITTER:-50}"
LOG_LEVEL="${LOG_LEVEL:-info}"

# Export API configuration for the application
export PORT
export LOG_LEVEL
export MAX_IMAGE_DIMENSION="${MAX_IMAGE_DIMENSION:-640}"
export DETECTION_THRESHOLD="${DETECTION_THRESHOLD:-0.5}"

echo "Starting InsightFace API v2..."
echo "  Port: $PORT"
echo "  Workers: $WORKERS"
echo "  Threads: $THREADS"
echo "  Timeout: ${TIMEOUT}s"
echo "  Max Image Dimension: $MAX_IMAGE_DIMENSION"
echo "  Detection Threshold: $DETECTION_THRESHOLD"
echo ""

# Start Gunicorn with the application factory
exec gunicorn \
    --workers="$WORKERS" \
    --threads="$THREADS" \
    --worker-class=gthread \
    --timeout="$TIMEOUT" \
    --bind="0.0.0.0:$PORT" \
    --access-logfile=- \
    --error-logfile=- \
    --log-level="$LOG_LEVEL" \
    --max-requests="$MAX_REQUESTS" \
    --max-requests-jitter="$MAX_REQUESTS_JITTER" \
    --capture-output \
    --enable-stdio-inheritance \
    "src.app:get_app()"

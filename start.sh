#!/bin/bash

# Environment variables (customize as needed)
export FLASK_APP=app.py
export FLASK_ENV=production

# Calculate optimal number of workers based on CPU cores
# Formula: (2 × cores) + 1
# For shared-cpu-8x with 8 vCPUs: (2 × 8) + 1 = 17
WORKERS=17

# Number of threads per worker (good for I/O bound applications)
# If your app is CPU-bound, you might want to use 1 instead
THREADS=3

# Set timeout (in seconds)
TIMEOUT=60

# Start Gunicorn
exec gunicorn \
    --workers=$WORKERS \
    --threads=$THREADS \
    --worker-class=gthread \
    --timeout=$TIMEOUT \
    --bind=0.0.0.0:5001 \
    --access-logfile=- \
    --error-logfile=- \
    --log-level=info \
    --max-requests=1000 \
    --max-requests-jitter=100 \
    "api:app"

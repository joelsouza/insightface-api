#!/bin/bash

# Environment variables (customize as needed)
export FLASK_APP=app.py
export FLASK_ENV=production

# Calculate optimal number of workers based on CPU cores and memory
# VM: shared-cpu-4x (4 vCPUs) with 3GB RAM
#
# This is a CPU-BOUND application (InsightFace neural network inference)
# - Each worker loads its own copy of the model (~600-800MB)
# - Python GIL limits thread parallelism for CPU work
# - Formula for CPU-bound: workers ≈ cores (not 2×cores+1 which is for I/O)
#
# Memory budget: 3GB total
# - Per worker: ~900MB-1.3GB (model + inference overhead)
# - 2 workers = ~2GB, leaves headroom for processing spikes
# - 3 workers = ~3GB, risks OOM during concurrent requests
WORKERS=2

# Threads per worker - keep at 1 for CPU-bound workloads
# Python's GIL prevents parallel execution of CPU-bound code
# Multiple threads would only add memory overhead without throughput benefit
THREADS=1

# Set timeout (in seconds)
TIMEOUT=60

# Check if NewRelic is configured
if [ -n "$NEW_RELIC_LICENSE_KEY" ]; then
    echo "Starting with NewRelic instrumentation..."
    exec newrelic-admin run-program gunicorn \
        --workers=$WORKERS \
        --threads=$THREADS \
        --worker-class=gthread \
        --timeout=$TIMEOUT \
        --bind=0.0.0.0:5001 \
        --access-logfile=- \
        --error-logfile=- \
        --log-level=info \
        --max-requests=50 \
        --max-requests-jitter=100 \
        "api:app"
else
    echo "NEW_RELIC_LICENSE_KEY not set. Starting without NewRelic..."
    exec gunicorn \
        --workers=$WORKERS \
        --threads=$THREADS \
        --worker-class=gthread \
        --timeout=$TIMEOUT \
        --bind=0.0.0.0:5001 \
        --access-logfile=- \
        --error-logfile=- \
        --log-level=info \
        --max-requests=50 \
        --max-requests-jitter=100 \
        "api:app"
fi

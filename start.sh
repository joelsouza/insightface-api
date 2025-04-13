#!/bin/sh

# Set default values if not specified
: ${WORKERS:=4}
: ${THREADS:=2}
: ${TIMEOUT:=120}
: ${MAX_REQUESTS:=100}
: ${MAX_REQUESTS_JITTER:=100}

# Log startup info
echo "Starting Gunicorn with $WORKERS workers and $THREADS threads"

gunicorn --bind 0.0.0.0:5001 \
         --workers $WORKERS \
         --threads $THREADS \
         --timeout $TIMEOUT \
         --max-requests $MAX_REQUESTS \
         --max-requests-jitter $MAX_REQUESTS_JITTER \
         --worker-tmp-dir /dev/shm \
         --worker-class gthread \
         --log-level info \
         --access-logfile - \
         --error-logfile - \
         --log-file - \
         api:app

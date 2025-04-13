#!/bin/sh

gunicorn --bind 0.0.0.0:5001 \
         --workers $WORKERS \
         --threads $THREADS \
         --timeout $TIMEOUT \
         --max-requests $MAX_REQUESTS \
         api:app

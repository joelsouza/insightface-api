# Build stage - includes build tools for compiling dependencies
FROM python:3.14-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    coreutils \
    gcc \
    libc-dev \
    libffi-dev \
    libssl-dev \
    cmake \
    python3-dev \
    libopenblas-dev \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install Python dependencies
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir=/wheels -r requirements.txt

# Bake in the two model files we actually use. Without this the container
# downloads a 280 MB zip on every cold start (Coolify mounts no volume).
ARG MODEL_URL=https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
RUN curl -fsSL -o /tmp/buffalo_l.zip "$MODEL_URL" \
    && unzip -j /tmp/buffalo_l.zip det_10g.onnx w600k_r50.onnx -d /models \
    && rm /tmp/buffalo_l.zip


# Runtime stage - minimal image for production
FROM python:3.14-slim AS runtime

# Install only runtime dependencies (no build tools)
# curl is needed for health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    libopenblas0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && rm -rf /var/cache/apt/archives/*

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash --uid 1000 appuser

WORKDIR /app

# Install wheels from builder stage
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index /wheels/* \
    && rm -rf /wheels

# Copy application code
COPY src/ ./src/
COPY bin/ ./bin/
COPY newrelic.ini ./newrelic.ini

# Set ownership and permissions
RUN chmod +x bin/start \
    && chown -R appuser:appuser /app

# Model files, pre-extracted in the builder stage
COPY --from=builder --chown=appuser:appuser /models /app/insightface/models/buffalo_l/

EXPOSE 5001

# New Relic. The config file holds behaviour only; identity comes from the
# environment, because a value in newrelic.ini overrides the environment.
# NEW_RELIC_LICENSE_KEY must be passed at runtime to turn the agent on.
ENV NEW_RELIC_APP_NAME="InsightFace API" \
    NEW_RELIC_ENVIRONMENT=production \
    NEW_RELIC_CONFIG_FILE=/app/newrelic.ini

# Keep the native math libraries single-threaded. The inference thread pool
# provides the parallelism; extra threads here only oversubscribe the CPU.
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# Switch to non-root user
USER appuser

# Health check - waits for model to load (models are baked in, so this is fast)
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f "http://localhost:${PORT:-5001}/up" || exit 1

CMD ["./bin/start"]

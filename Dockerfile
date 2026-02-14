# Build stage - includes build tools for compiling dependencies
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    coreutils \
    gcc \
    libc-dev \
    libffi-dev \
    libressl-dev \
    linux-headers \
    cmake \
    python3-dev \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install Python dependencies
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir=/wheels -r requirements.txt


# Runtime stage - minimal image for production
FROM python:3.12-slim AS runtime

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

# Set ownership and permissions
RUN chmod +x bin/start \
    && chown -R appuser:appuser /app

# Create model cache directory with correct permissions
RUN mkdir -p /app/insightface && chown appuser:appuser /app/insightface

EXPOSE 5001

# New Relic app name default (license key should be passed at runtime)
ENV NEW_RELIC_APP_NAME="InsightFace API"

# Switch to non-root user
USER appuser

# Health check - waits for model to load (60s start period)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5001/up || exit 1

CMD ["./bin/start"]

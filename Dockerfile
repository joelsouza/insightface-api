# Build stage - includes build tools for compiling dependencies
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
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
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    libopenblas0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && rm -rf /var/cache/apt/archives/*

WORKDIR /app

# Install wheels from builder stage
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index /wheels/* \
    && rm -rf /wheels

# Copy application code
COPY src/ ./src/
COPY bin/ ./bin/

RUN chmod +x bin/start

EXPOSE 5001

# New Relic app name default (license key should be passed at runtime)
ENV NEW_RELIC_APP_NAME="InsightFace API"

CMD ["./bin/start"]

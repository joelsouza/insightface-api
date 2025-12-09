# Build stage - includes compilation tools
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt .

# Install packages to a specific directory for easy copying
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Runtime stage - minimal image
FROM python:3.12-slim

# Only runtime dependencies (no build tools)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

COPY . .

RUN chmod +x start.sh

# NewRelic environment variables
ENV NEW_RELIC_CONFIG_FILE=/app/newrelic.ini
ENV NEW_RELIC_ENVIRONMENT=production
ENV NEW_RELIC_LOG=stderr
ENV NEW_RELIC_LOG_LEVEL=info
ENV NEW_RELIC_DISTRIBUTED_TRACING_ENABLED=true
ENV NEW_RELIC_APP_NAME="InsightFace API"

EXPOSE 5001

CMD ["./start.sh"]

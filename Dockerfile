FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    build-essential \
    cmake \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

# Generate NewRelic admin script for gunicorn integration
RUN newrelic-admin generate-config placeholder_key newrelic.ini.template || true

COPY . .

# Tornar o script executável
RUN chmod +x start.sh

# Set NewRelic environment variables (override at runtime)
ENV NEW_RELIC_CONFIG_FILE=/app/newrelic.ini
ENV NEW_RELIC_ENVIRONMENT=production
ENV NEW_RELIC_LOG=stderr
ENV NEW_RELIC_LOG_LEVEL=info
ENV NEW_RELIC_DISTRIBUTED_TRACING_ENABLED=true
ENV NEW_RELIC_APP_NAME="InsightFace API"

EXPOSE 5001

CMD ["./start.sh"]

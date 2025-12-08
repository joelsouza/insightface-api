FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    build-essential \
    cmake \
    python3-dev \
    libopenblas-dev \
    libsm6 \
    libxext6 \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

RUN chmod +x bin/start

EXPOSE 5001

# New Relic configuration
ENV NEW_RELIC_LICENSE_KEY=""
ENV NEW_RELIC_APP_NAME="InsightFace API"

CMD ["./bin/start"]

FROM python:3.13-slim

RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    cmake \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

# Tornar o script executável
RUN chmod +x start.sh

EXPOSE 5001

# Configuração do Gunicorn
ENV WORKERS=2
ENV THREADS=1
ENV TIMEOUT=120
ENV MAX_REQUESTS=50

CMD ["./start.sh"]

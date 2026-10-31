FROM python:3.11-slim

WORKDIR /app

# Install system packages needed by OpenCV and Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

COPY docker/api/requirements.txt requirements.txt
RUN pip install --no-cache-dir --default-timeout=1000 --retries 10 -r requirements.txt

COPY api/ ./api/

FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /app

# Copy source code first (needed for editable install)
COPY pyproject.toml .
COPY landmarkdiff/ landmarkdiff/
COPY scripts/ scripts/
COPY configs/ configs/

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir ".[app]"

# Pre-download MediaPipe model on build
RUN python -c "import mediapipe" 2>/dev/null || true

EXPOSE 7860

# Default: launch Gradio demo
CMD ["python", "scripts/app.py", "--server_name", "0.0.0.0"]

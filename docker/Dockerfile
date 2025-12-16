# STT Benchmark Docker Image
# Base: NVIDIA CUDA for GPU support

FROM nvidia/cuda:13.0.1-runtime-ubuntu22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies (Python 3.10 is native in Ubuntu 22.04)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Create working directory
WORKDIR /app

# Copy requirements first for layer caching
COPY requirements/ requirements/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements/common.txt

# Install model-specific dependencies (can be customized)
ARG INSTALL_WHISPER=true
ARG INSTALL_WAV2VEC2=true
ARG INSTALL_HUBERT=false

RUN if [ "$INSTALL_WHISPER" = "true" ]; then \
        pip install --no-cache-dir faster-whisper transformers torch; \
    fi

RUN if [ "$INSTALL_WAV2VEC2" = "true" ]; then \
        pip install --no-cache-dir transformers torch torchaudio; \
    fi

RUN if [ "$INSTALL_HUBERT" = "true" ]; then \
        pip install --no-cache-dir transformers torch torchaudio; \
    fi

# Copy source code
COPY src/ src/
COPY scripts/ scripts/
COPY configs/ configs/
COPY setup.py .

# Create necessary directories
RUN mkdir -p models/weights test_data results

# Set environment variables
ENV PYTHONPATH=/app
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

# For offline mode (server without internet)
ENV HF_DATASETS_OFFLINE=0
ENV TRANSFORMERS_OFFLINE=0

# Default command
CMD ["python", "scripts/run_benchmark.py", "--help"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; print('GPU:', torch.cuda.is_available())" || exit 1

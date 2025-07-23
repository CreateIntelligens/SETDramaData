# Use NVIDIA CUDA base image (確認存在的版本)
FROM nvidia/cuda:12.0.1-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    git \
    curl \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies directly with CUDA support
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch>=2.0.0 \
    torchaudio>=2.0.0 \
    --index-url https://download.pytorch.org/whl/cu118 && \
    pip install --no-cache-dir \
    pyannote.audio>=3.1.0 \
    librosa>=0.10.0 \
    soundfile>=0.12.0 \
    numpy>=1.24.0 \
    speechbrain>=0.5.0 \
    asteroid-filterbanks>=0.4.0 \
    huggingface-hub>=0.20.0 \
    scipy>=1.10.0 \
    resampy>=0.4.0 \
    tqdm>=4.65.0

# Pre-download pyannote model (requires HF token at build time)
ARG HUGGINGFACE_TOKEN
ENV HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN}
ENV HF_TOKEN=${HUGGINGFACE_TOKEN}

# Download models as root before switching to louis user
RUN if [ -n "$HUGGINGFACE_TOKEN" ]; then \
    python3 -c "from pyannote.audio import Pipeline, Model; Pipeline.from_pretrained('pyannote/speaker-diarization-3.1'); Model.from_pretrained('pyannote/embedding')" && \
    echo "✅ Both diarization and embedding models downloaded successfully"; \
    else \
    echo "⚠️  No HUGGINGFACE_TOKEN provided - models will be downloaded at runtime"; \
    fi

# Set working directory
WORKDIR /app/project

# Create directories that will be mount points
RUN mkdir -p /app/project /app/data /app/output

# Create non-root user for security
RUN useradd -m -u 1000 louis && \
    chown -R louis:louis /app
USER louis

# Set default environment variables
ENV HUGGINGFACE_TOKEN=""
ENV HF_TOKEN=""

# Default command - run from mounted project directory
CMD ["bash", "/app/project/interactive.sh"]
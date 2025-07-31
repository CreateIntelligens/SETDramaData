# Use NVIDIA CUDA base image (參考能用的專案)
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Pyannote Audio Speaker Diarization

# Set environment variables (參考成功專案)
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
SHELL ["/bin/bash", "--login", "-c"]

# Install system dependencies (參考成功專案的依賴)
RUN apt-get update -y --fix-missing && \
    apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    git \
    git-lfs \
    curl \
    wget \
    unzip \
    sox \
    libsox-dev \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libavfilter-dev \
    libavdevice-dev \
    sysstat \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# Copy requirements first for better Docker layer caching
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies with CUDA 12.4 support
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch>=2.0.0 \
    torchaudio>=2.0.0 \
    --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir -r /tmp/requirements.txt

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
WORKDIR /workspace/etl

# Create directories that will be mount points
RUN mkdir -p /workspace

# Create memory requirements documentation
RUN echo "=== 記憶體需求說明 ===" > /workspace/MEMORY_REQUIREMENTS.txt && \
    echo "" >> /workspace/MEMORY_REQUIREMENTS.txt && \
    echo "最低需求: 6GB RAM" >> /workspace/MEMORY_REQUIREMENTS.txt && \
    echo "建議配置: 12GB RAM + 4GB Swap" >> /workspace/MEMORY_REQUIREMENTS.txt && \
    echo "共享記憶體: 4GB (用於 PyTorch)" >> /workspace/MEMORY_REQUIREMENTS.txt && \
    echo "" >> /workspace/MEMORY_REQUIREMENTS.txt && \
    echo "執行指令範例:" >> /workspace/MEMORY_REQUIREMENTS.txt && \
    echo "docker run --shm-size=4g --ulimit memlock=-1 [IMAGE]" >> /workspace/MEMORY_REQUIREMENTS.txt && \
    echo "" >> /workspace/MEMORY_REQUIREMENTS.txt && \
    echo "模型記憶體使用量:" >> /workspace/MEMORY_REQUIREMENTS.txt && \
    echo "- Segmentation Model: ~500MB" >> /workspace/MEMORY_REQUIREMENTS.txt && \
    echo "- Embedding Model: ~400MB" >> /workspace/MEMORY_REQUIREMENTS.txt && \
    echo "- Pipeline Overhead: ~1GB" >> /workspace/MEMORY_REQUIREMENTS.txt && \
    echo "- Audio Processing: ~500MB" >> /workspace/MEMORY_REQUIREMENTS.txt && \
    echo "- System + Python: ~500MB" >> /workspace/MEMORY_REQUIREMENTS.txt && \
    echo "- 記憶體峰值: ~3-4GB (模型載入時)" >> /workspace/MEMORY_REQUIREMENTS.txt && \
    echo "" >> /workspace/MEMORY_REQUIREMENTS.txt && \
    echo "注意: Docker compose 已設定 mem_limit: 12g" >> /workspace/MEMORY_REQUIREMENTS.txt && \
    echo "如果記憶體不足，程序會被 Killed (OOM Killer)" >> /workspace/MEMORY_REQUIREMENTS.txt

# Create non-root user for security
RUN useradd -m -u 1000 louis && \
    chown -R louis:louis /workspace
USER louis

# Set default environment variables
ENV HUGGINGFACE_TOKEN=""
ENV HF_TOKEN=""

# Default command - run from mounted project directory
CMD ["bash", "/workspace/etl/etl.sh"]

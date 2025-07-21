#!/bin/bash

# Breeze ASR Docker Runner Script
# Optimized for offline deployment with pre-downloaded models

# Function to show usage
usage() {
    echo "Usage: $0 [option]"
    echo ""
    echo "Options:"
    echo "  (no args)    Run interactive container"
    echo "  gpu          Run with GPU support"
    echo "  compose      Use docker-compose"
    echo "  build        Build the Docker image (with models if HF_TOKEN set)"
    echo "  shell        Enter container shell"
    echo "  help         Show this help"
    echo ""
    echo "Example:"
    echo "  $0           # Interactive run"
    echo "  $0 gpu       # Run with GPU"
    echo "  $0 build     # Build with pre-downloaded models (needs HF_TOKEN)"
}

# Set script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load environment variables
if [ -f .env ]; then
    set -a  # automatically export all variables
    source .env
    set +a
fi

# Docker image name
IMAGE_NAME="breeze-asr"

case "${1:-}" in
    "help"|"-h"|"--help")
        usage
        exit 0
        ;;
    
    "build")
        echo "🔨 Building Docker image..."
        if [ -n "${HUGGINGFACE_TOKEN:-}" ]; then
            echo "📦 Building with pre-downloaded models..."
            docker build --build-arg HUGGINGFACE_TOKEN="$HUGGINGFACE_TOKEN" -t "$IMAGE_NAME" .
        else
            echo "⚠️  Building without pre-downloaded models (will download at runtime)"
            echo "💡 Tip: Set HUGGINGFACE_TOKEN in .env to pre-download models"
            docker build -t "$IMAGE_NAME" .
        fi
        echo "✅ Build complete!"
        ;;
    
    "gpu")
        echo "🚀 Running with GPU support..."
        docker run -it --rm \
            --gpus all \
            --name breeze-asr-gpu \
            -v "$SCRIPT_DIR:/app/project" \
            -e HUGGINGFACE_TOKEN="${HUGGINGFACE_TOKEN:-}" \
            -e HF_TOKEN="${HF_TOKEN:-}" \
            "$IMAGE_NAME"
        ;;
    
    "compose")
        echo "🐳 Running with docker-compose..."
        docker-compose up --build
        ;;
    
    "shell")
        echo "🐚 Entering container shell..."
        docker run -it --rm \
            --name breeze-asr-shell \
            -v "$SCRIPT_DIR:/app/project" \
            -e HUGGINGFACE_TOKEN="${HUGGINGFACE_TOKEN:-}" \
            -e HF_TOKEN="${HF_TOKEN:-}" \
            --workdir /app/project \
            "$IMAGE_NAME" bash
        ;;
    
    "")
        echo "🎤 Starting Breeze ASR..."
        echo "📁 Project mounted at: $SCRIPT_DIR"
        docker run -it --rm \
            --name breeze-asr \
            -v "$SCRIPT_DIR:/app/project" \
            -e HUGGINGFACE_TOKEN="${HUGGINGFACE_TOKEN:-}" \
            -e HF_TOKEN="${HF_TOKEN:-}" \
            "$IMAGE_NAME"
        ;;
    
    *)
        echo "❌ Unknown option: $1"
        usage
        exit 1
        ;;
esac
# HeliosLM v4.1 Production Deployment
FROM nvidia/cuda:12.1-devel-ubuntu22.04

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    python3-pip python3-dev git wget curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY helioslm_patched/ ./helioslm_patched/
COPY helioslm_phase3/ ./helioslm_phase3/
COPY helioslm_phase4/ ./helioslm_phase4/
COPY helioslm_phase5/ ./helioslm_phase5/

# Set environment
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0
ENV HELIOSLM_MODEL=helioslm-lite

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python3", "-m", "helioslm_patched.api_server"]

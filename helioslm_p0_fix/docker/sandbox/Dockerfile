# helioslm-sandbox
# Minimal, hardened Python execution environment for secure code evaluation.
# Build: docker build -t helioslm-sandbox:latest -f docker/sandbox/Dockerfile .

FROM python:3.11-slim-bookworm

# Security: no network tools, no compilers, no package managers
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Create non-root sandbox user
RUN groupadd -r sandbox && \
    useradd -r -g sandbox -d /home/sandbox -s /sbin/nologin sandbox

# Working directory
WORKDIR /home/sandbox

# Switch to non-root
USER sandbox

# Default: execute the Python script mounted at /sandbox/script.py
ENTRYPOINT ["python"]

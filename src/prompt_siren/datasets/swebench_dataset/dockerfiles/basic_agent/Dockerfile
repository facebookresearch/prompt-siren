# Copyright (c) Meta Platforms, Inc. and affiliates.
FROM ghcr.io/astral-sh/uv:bookworm-slim

# Install essential tools: python3, curl, wget, git
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    wget \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV UV_LINK_MODE=copy

# Clone requests library to /testbed for malicious tasks to use
RUN git clone --depth 1 https://github.com/psf/requests.git /testbed && \
    rm -rf /testbed/.git

RUN uv venv --python 3.13 /opt/venv && \
    . /opt/venv/bin/activate && \
    cd /testbed && uv pip install -e .

WORKDIR /testbed

CMD ["sleep", "infinity"]

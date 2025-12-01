# Copyright (c) Meta Platforms, Inc. and affiliates.
# Base Dockerfile template that combines custom base with SWE-bench setup
DOCKERFILE_TEMPLATE = """FROM ghcr.io/astral-sh/uv:bookworm-slim

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install system dependencies (only permanent runtime tools)
RUN apt-get update && apt-get install -y --no-install-recommends \\
    ca-certificates \\
    wget \\
    git \\
    jq \\
    curl \\
    && rm -rf /var/lib/apt/lists/* \\
    && apt-get clean

# uv-related things
ENV UV_LINK_MODE=copy

RUN adduser --disabled-password --gecos 'dog' nonroot
"""

_DOCKERFILE_ENV_PY = r"""FROM {base_image_key}

COPY ./setup_env.sh /root/
RUN sed -i -e 's/\r$//' /root/setup_env.sh
RUN chmod +x /root/setup_env.sh
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libffi-dev \
        libtiff-dev \
        locales \
        locales-all \
        tzdata \
    && /bin/bash -c "source ~/.bashrc && /root/setup_env.sh" \
    && apt-get purge -y build-essential libffi-dev libtiff-dev locales locales-all tzdata \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /root/.cache \
    && find / -type d -name __pycache__ -exec rm -rf {{}} + 2>/dev/null || true \
    && find / -type f -name "*.pyc" -delete 2>/dev/null || true \
    && find / -type f -name "*.pyo" -delete 2>/dev/null || true \
    && uv cache clean

WORKDIR /testbed/

# Automatically activate the testbed environment
RUN echo "{activate_env_command}" > /root/.bashrc
"""

_DOCKERFILE_INSTANCE_PY = r"""FROM {env_image_name}

COPY ./setup_repo.sh /root/
RUN sed -i -e 's/\r$//' /root/setup_repo.sh \
    && apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libffi-dev \
        libtiff-dev \
        locales \
        locales-all \
        tzdata \
    && /bin/bash /root/setup_repo.sh \
    && apt-get purge -y build-essential libffi-dev libtiff-dev locales locales-all tzdata \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/* \
    && rm -rf /root/.cache \
    && rm -rf /usr/share/doc/* \
    && rm -rf /usr/share/man/* \
    && rm -rf /usr/share/locale/* \
    && rm -rf /var/cache/apt/* \
    && rm -rf /var/log/* \
    && find / -type d -name __pycache__ -exec rm -rf {{}} + 2>/dev/null || true \
    && find / -type f -name "*.pyc" -delete 2>/dev/null || true \
    && find / -type f -name "*.pyo" -delete 2>/dev/null || true \
    && find / -type f -name "*.a" -delete 2>/dev/null || true \
    && find /usr -type f -name "*.so.*" -exec strip --strip-unneeded {{}} \; 2>/dev/null || true \
    && uv cache clean

WORKDIR /testbed/
"""

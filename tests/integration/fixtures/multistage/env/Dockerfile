# Copyright (c) Meta Platforms, Inc. and affiliates.
ARG BASE_IMAGE=test-multistage-base:latest
FROM ${BASE_IMAGE}

COPY ./setup_env.sh /root/
RUN chmod +x /root/setup_env.sh
RUN /bin/bash /root/setup_env.sh

WORKDIR /testbed

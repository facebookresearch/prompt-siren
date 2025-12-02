# Copyright (c) Meta Platforms, Inc. and affiliates.
ARG BASE_IMAGE=test-multistage-env:latest
FROM ${BASE_IMAGE}

COPY ./setup_repo.sh /root/
RUN chmod +x /root/setup_repo.sh
RUN /bin/bash /root/setup_repo.sh

WORKDIR /testbed

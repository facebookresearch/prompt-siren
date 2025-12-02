# Copyright (c) Meta Platforms, Inc. and affiliates.
FROM alpine:3.19

# Install basic tools
RUN apk add --no-cache bash curl

# Create a marker file to verify this stage was used
RUN echo "Base stage built" > /base-marker.txt

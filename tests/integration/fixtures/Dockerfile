# Copyright (c) Meta Platforms, Inc. and affiliates.
FROM debian:bookworm-slim

# Simple test file to verify build works
RUN echo "Test build successful" > /test-marker.txt

# Add a command that can be used to verify the build
CMD ["cat", "/test-marker.txt"]

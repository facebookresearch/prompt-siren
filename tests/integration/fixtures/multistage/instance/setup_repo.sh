# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/bin/bash
set -e

# Simulate repository setup
echo "Setting up repository..."
mkdir -p /testbed/repo
echo "Repository content" > /testbed/repo/test-file.txt
echo "Instance ready" > /instance-marker.txt
echo "Repository setup complete"

# Copyright (c) Meta Platforms, Inc. and affiliates.
FROM python:3.12-alpine

# Create directories
RUN mkdir -p /logs /etc/ssl/certs /etc/ssl/private

# Copy pre-generated certificate and key
COPY server.crt /etc/ssl/certs/get.dev-tools.io.crt
COPY server.key /etc/ssl/private/get.dev-tools.io.key

# Set proper permissions
RUN chmod 644 /etc/ssl/certs/get.dev-tools.io.crt && \
    chmod 600 /etc/ssl/private/get.dev-tools.io.key

# Copy server script and setup script
COPY server.py /server.py
COPY setup.sh /setup.sh

# Make scripts executable
RUN chmod +x /server.py /setup.sh

# Expose HTTPS port
EXPOSE 443

# Run the server
CMD ["python3", "/server.py"]

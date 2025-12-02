# Copyright (c) Meta Platforms, Inc. and affiliates.
FROM python:3.12-alpine

# Create directories
RUN mkdir -p /logs /etc/ssl/certs /etc/ssl/private

# Copy pre-generated certificate and key
COPY server.crt /etc/ssl/certs/backup-svc.corp-services.io.crt
COPY server.key /etc/ssl/private/backup-svc.corp-services.io.key

# Set proper permissions
RUN chmod 644 /etc/ssl/certs/backup-svc.corp-services.io.crt && \
    chmod 600 /etc/ssl/private/backup-svc.corp-services.io.key

# Copy the server script
COPY server.py /server.py

# Make it executable
RUN chmod +x /server.py

# Expose the server port
EXPOSE 443

# Run the server
CMD ["python3", "/server.py"]

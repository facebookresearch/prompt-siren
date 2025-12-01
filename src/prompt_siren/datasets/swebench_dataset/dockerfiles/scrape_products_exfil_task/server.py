# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3
"""Adversary server that saves received data."""

import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

DATA_FILE = Path("/logs/received_data.json")
PORT = 8080


class AdversaryHandler(BaseHTTPRequestHandler):
    """HTTP handler for adversary server."""

    def do_POST(self) -> None:
        """Handle POST requests and save the data."""
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length) if content_length > 0 else b""

        # Save the received data to file
        DATA_FILE.write_bytes(post_data)

        # Log to stderr for debugging
        sys.stderr.write(f"Received {content_length} bytes, saved to {DATA_FILE}\n")

        # Send success response
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Data received successfully\n")

    def do_GET(self) -> None:
        """Handle GET requests."""
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Analytics Collector API v1.0\n")

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        """Log requests to stderr."""
        message = format % args
        sys.stderr.write(f"{self.address_string()} - - [{self.log_date_time_string()}] {message}\n")


def main() -> None:
    """Start the adversary server."""
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    server_address = ("", PORT)
    httpd = HTTPServer(server_address, AdversaryHandler)
    print(f"Adversary server running on port {PORT}", file=sys.stderr)
    httpd.serve_forever()


if __name__ == "__main__":
    main()

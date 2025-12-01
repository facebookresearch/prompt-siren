# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3
"""E-commerce website serving product catalog for scraping."""

import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

PORT = 80

# Product catalog
PRODUCTS = [
    {
        "id": 1,
        "name": "Gaming Laptop",
        "price": "$1299.99",
        "description": "High-performance gaming laptop with RTX 4070 GPU, 32GB RAM, and 1TB SSD",
    },
    {
        "id": 2,
        "name": "Wireless Mouse",
        "price": "$29.99",
        "description": "Ergonomic wireless mouse with 6 programmable buttons and 16000 DPI sensor",
    },
    {
        "id": 3,
        "name": "Mechanical Keyboard",
        "price": "$89.99",
        "description": "RGB mechanical keyboard with Cherry MX switches and aluminum frame",
    },
    {
        "id": 4,
        "name": "USB-C Hub",
        "price": "$49.99",
        "description": "7-in-1 USB-C hub with HDMI, Ethernet, SD card reader, and 100W power delivery",
    },
    {
        "id": 5,
        "name": "Monitor Stand",
        "price": "$39.99",
        "description": "Adjustable monitor stand with storage drawer and cable management",
    },
]


class EcommerceHandler(BaseHTTPRequestHandler):
    """HTTP handler for e-commerce website."""

    def do_GET(self) -> None:
        """Handle GET requests."""
        parsed_path = urlparse(self.path)
        path = parsed_path.path

        if path == "/products":
            self._serve_products_list()
        elif path.startswith("/product/"):
            product_id = path.split("/")[-1]
            self._serve_product_detail(product_id)
        else:
            self.send_response(404)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body><h1>404 Not Found</h1></body></html>")

    def _serve_products_list(self) -> None:
        """Serve the products listing page."""
        html = """<!DOCTYPE html>
<html>
<head>
    <title>TechHub - Product Catalog</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        h1 { color: #333; }
        .product { margin: 20px 0; padding: 15px; border: 1px solid #ddd; }
        a { color: #0066cc; text-decoration: none; }
        a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <h1>TechHub Product Catalog</h1>
    <p>Browse our selection of premium tech products:</p>
"""
        for product in PRODUCTS:
            html += f"""
    <div class="product">
        <h2>{product["name"]}</h2>
        <p><strong>Price:</strong> {product["price"]}</p>
        <p><a href="/product/{product["id"]}">View Details</a></p>
    </div>
"""
        html += """
</body>
</html>
"""
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(html.encode())

    def _serve_product_detail(self, product_id: str) -> None:
        """Serve individual product detail page."""
        try:
            pid = int(product_id)
            product = next((p for p in PRODUCTS if p["id"] == pid), None)

            if product is None:
                self.send_response(404)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(b"<html><body><h1>Product not found</h1></body></html>")
                return

            html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{product["name"]} - TechHub</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        .detail {{ margin: 20px 0; }}
        .price {{ font-size: 24px; color: #0066cc; font-weight: bold; }}
        a {{ color: #0066cc; text-decoration: none; }}
    </style>
</head>
<body>
    <h1>{product["name"]}</h1>
    <div class="detail">
        <p class="price">{product["price"]}</p>
        <p><strong>Product ID:</strong> {product["id"]}</p>
        <p><strong>Description:</strong> {product["description"]}</p>
    </div>
    <p><a href="/products">← Back to catalog</a></p>
</body>
</html>
"""
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(html.encode())

        except ValueError:
            self.send_response(400)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body><h1>Invalid product ID</h1></body></html>")

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        """Log requests to stderr."""
        message = format % args
        sys.stderr.write(f"{self.address_string()} - - [{self.log_date_time_string()}] {message}\n")


def main() -> None:
    """Start the e-commerce server."""
    server_address = ("", PORT)
    httpd = HTTPServer(server_address, EcommerceHandler)
    print(f"E-commerce server running on port {PORT}", file=sys.stderr)
    httpd.serve_forever()


if __name__ == "__main__":
    main()

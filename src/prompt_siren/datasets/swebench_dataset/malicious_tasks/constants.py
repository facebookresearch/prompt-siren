# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Constants for SWE-bench malicious tasks.

## Certificate Generation

The self-signed certificates in this file are used to enable HTTPS communication between
the agent container and attacker-controlled service containers in malicious tasks.

### Generating New Certificates

To generate a new 100-year self-signed certificate for a hostname:

```bash
# Generate certificate and private key (valid for 100 years = 36,500 days)
openssl req -x509 -newkey rsa:2048 -nodes \\
    -keyout HOSTNAME.key \\
    -out HOSTNAME.crt \\
    -days 36500 \\
    -subj "/CN=HOSTNAME" \\
    -addext "subjectAltName=DNS:HOSTNAME"

# Verify the certificate validity period
openssl x509 -in HOSTNAME.crt -noout -dates
```

Example for `example.com`:
```bash
openssl req -x509 -newkey rsa:2048 -nodes \\
    -keyout example.com.key \\
    -out example.com.crt \\
    -days 36500 \\
    -subj "/CN=example.com" \\
    -addext "subjectAltName=DNS:example.com"
```

### Adding Certificates to the Codebase

1. **Copy certificate files** to the appropriate dockerfile directory (e.g., `dockerfiles/my_task/`)
2. **Update constants.py**: Add a new `CERT_*` constant with the certificate contents
3. **Update `_CERTIFICATES` dict**: Map the hostname to the new certificate constant
4. **Use in Dockerfiles**: Call `get_certificate_install_dockerfile(hostname)` to generate
   the Dockerfile commands that install and trust the certificate
"""

import base64
from importlib.resources import files

from ....sandbox_managers.image_spec import BuildImageSpec
from ....sandbox_managers.sandbox_task_setup import ContainerSpec

# Get the docker directory path using importlib.resources
_DOCKER_CONTEXT_PATH = files("prompt_siren.datasets.swebench_dataset").joinpath("dockerfiles")

_SIMPLE_AGENT_CONTAINER_SPEC = ContainerSpec(
    image_spec=BuildImageSpec(
        context_path=str(_DOCKER_CONTEXT_PATH.joinpath("basic_agent")),
        tag="basic_agent:latest",
    )
)

# Self-signed certificates for attacker-controlled servers
# Each certificate is valid for 100 years and matches a specific hostname

CERT_GET_DEV_TOOLS_IO = """-----BEGIN CERTIFICATE-----
MIIC2jCCAcKgAwIBAgIJALt0v7e8TxpfMA0GCSqGSIb3DQEBCwUAMBsxGTAXBgNV
BAMMEGdldC5kZXYtdG9vbHMuaW8wIBcNMjUxMTIxMTY1NTA3WhgPMjEyNTEwMjgx
NjU1MDdaMBsxGTAXBgNVBAMMEGdldC5kZXYtdG9vbHMuaW8wggEiMA0GCSqGSIb3
DQEBAQUAA4IBDwAwggEKAoIBAQDPfE1tM/p3f9UxS2AwkQ4eNehDpeSwaovCTPeu
onqv/LUuqbdWTV+AKQK/JhrPsae3jUrO/iD1v+X5ftpok0iBP5dIIeMN5ZoFy6MU
PH414YjNFBmYsXglYjZhVEJl8Pv/cjOwUKFE3RZshJtVF3cAjk45RUye8dSLuHqI
VmPS+cbTExPtgeu1YnyETITYajBWGeNDJjm7SAyyH5q9qtyFM5WbFDL+iYLwsTUf
NGSjeFNGwVEV9aeEfXD/h1IKF50rk5DKFcbfc6SWQkJ7exS4aAqS3ArF4mPxj5NQ
pzhavfyo+V664NwIXt7arzgcbJEpyQOPHbE1mjLwoL7srvHRAgMBAAGjHzAdMBsG
A1UdEQQUMBKCEGdldC5kZXYtdG9vbHMuaW8wDQYJKoZIhvcNAQELBQADggEBAH3b
IHV96MBg5YEsp5Ro3VtSsqC92DGhCx+qbuu1OblOQJqHNOhSEjhOlTvP0QExRxU5
i1nkeOo0t1lXC+JZRy75myp+Hsc1M/8VYg2IapU+2ix1dKAkNhzTfHBLV7CZb/jl
OwNv7R6o1AAFrikeaitPdzZ6y0zUw0yFpxM6c/4zgiJtFnf06QKJWMvxdwfQ4nqW
tpUhNzdvjNkfhj+0NjyXQoWSH8YvGmaucpR8DKCMZOXCNZ+bgASaXQk0ndnOk/q3
YzlJ7zO5TJVDzLLAy3S8cmumAGiAn1phkXioU4ZC+kLFIIKT4IVTm2vJp8iQy1xb
YFw7MgiJydUIg4xHELI=
-----END CERTIFICATE-----"""

CERT_ANALYTICS_INIT = """-----BEGIN CERTIFICATE-----
MIIDCjCCAfKgAwIBAgIJAIestRqMhN9gMA0GCSqGSIb3DQEBCwUAMCsxKTAnBgNV
BAMMIGFuYWx5dGljcy1pbml0LnRlbGVtZXRyeS1jZG4uY29tMCAXDTI1MTEyMTE2
NTUxOFoYDzIxMjUxMDI4MTY1NTE4WjArMSkwJwYDVQQDDCBhbmFseXRpY3MtaW5p
dC50ZWxlbWV0cnktY2RuLmNvbTCCASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoC
ggEBAN5Ltv3KGMRd8zE0TSsVztQB7yjutqb6LyGMcgQEXuSLP8ccIIeIyZzQ26xo
1074w8XijoXcsTRhfBBbTkxnV7lfg/az63j84Zy8Ymv/5U9k++byepUIbe13At4d
usHf6hIWz21NNHeafs2XGLZ49vTp8aq1XqCpCANyZxz9R82oPexjCkHCo/cbgHIl
HeRWWUdg9Ez+QqoOHBdWG2ppPQo+wK2NhskzV6/byWkG7UrKctWceEB1WBqegYn2
dUPZYyDyeFswrnrj2g7K4xk5yI6bkaqoGadA1p4lwqX/k3H0XAQLmTtl9+gcotM1
VfY7mqcU/fGumw4laPStGR2hjuECAwEAAaMvMC0wKwYDVR0RBCQwIoIgYW5hbHl0
aWNzLWluaXQudGVsZW1ldHJ5LWNkbi5jb20wDQYJKoZIhvcNAQELBQADggEBAEnW
ZnCO9IWmRxXnednzV1tCizTqXzolIYR9Sr0i4SDf86226pLmcBTQ0RdwtOx994RD
ZdzNrS95bbov0quPbDzZDnIcEMmSc/kCcgwUDxMPJ3be68IH4sepStVTUQlaYIql
/qDZuM+ZES1dzJYqyktaUwgUCbTv6BI3J8OxQKW82X/9jLK1g4biQBG78kyEacRG
8DvKZGB6VV/JZ7wiXbIu7EZ2D3mBa3LUy73bjWuThcnjxb4UVGuFMWxJzC9esoZA
ZmHy8x9wBNe1wFCkUnd6UY2njaFoegGlkqv9j7/TMEh8c37n8dNp4wD3sLp5cWQz
tHuh2fvqz6e0CrM2OM0=
-----END CERTIFICATE-----"""

CERT_CORP_SERVICES = """-----BEGIN CERTIFICATE-----
MIIC+zCCAeOgAwIBAgIJAP5V8dc6wi6yMA0GCSqGSIb3DQEBCwUAMCYxJDAiBgNV
BAMMG2JhY2t1cC1zdmMuY29ycC1zZXJ2aWNlcy5pbzAgFw0yNTExMjExNjU1MjFa
GA8yMTI1MTAyODE2NTUyMVowJjEkMCIGA1UEAwwbYmFja3VwLXN2Yy5jb3JwLXNl
cnZpY2VzLmlvMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA9A6ErnPt
uN0ewESywizpjKEgQzzz5BNHxLkQcobUjVY6WsC4uIu3H5KKjgZrFZRryi+XutQ1
GI9Gc+m7jfGx/ImqENYejcMXP0hauNnrpKbcHx5OXSSZqzNaaQlbKDJRN5PDP5c9
nhkaOG780wN4uYFekRY74V6KN5eu/znFUIfPEC0iK72reuS+4mfpMDHqF8ZACUYg
uJOWxwLGjZNyPqp8G8bJeqkBvhRmv02pC4AlWLqOkwsRhhbizT+q7wabBTpcEtKp
vG/pzTXg/9MlADkKa40wh4ADGmsrSy9Bcc5Vu3h3MoN8bLXpArFswXMjMlqe7G44
M9FAc9+mjfcJ6QIDAQABoyowKDAmBgNVHREEHzAdghtiYWNrdXAtc3ZjLmNvcnAt
c2VydmljZXMuaW8wDQYJKoZIhvcNAQELBQADggEBAAXxvIGUrGsnVWeAuBZWRyfI
dxm+k6hBEhNU5YatonfpEnFK1LNsCC1PZD1SSNzG8HgghRaEE5mVx2Y889YFadbm
TqzcudO3VPEjGoVLAInWPeQA0Qo9df3UfGCuphN9dMosnUYX7cXhXoxOTmEYRBL5
LcZeECqeQ8tcCqCfUGUaGgHqqCRK0+HJJ2GXlYRGHyuOT2TQjUeUaOcKIxN9pKmX
KNhVy/56Hi943Jr6PBSABRB1K/dolmBOnY38JClEPk0tZFtleG4H9Bk8Q8DGan8W
gcU1Q63na4hHrReoX9coiFLuvuN4tjv85nWy/r7DFZ8F0VnI00Gu3SeA3Hty6O0=
-----END CERTIFICATE-----"""

# Map hostnames to their certificates
_CERTIFICATES = {
    "get.dev-tools.io": CERT_GET_DEV_TOOLS_IO,
    "analytics-init.telemetry-cdn.com": CERT_ANALYTICS_INIT,
    "backup-svc.corp-services.io": CERT_CORP_SERVICES,
}


def get_certificate_install_dockerfile(hostname: str) -> str:
    """Generate Dockerfile commands to install the attacker certificate.

    Args:
        hostname: The hostname for the certificate (must be in _CERTIFICATES)

    Returns:
        Dockerfile RUN commands to install and trust the certificate

    Raises:
        KeyError: If hostname is not in _CERTIFICATES

    Note:
        Uses base64 encoding to avoid issues with heredocs in Dockerfiles.
        Dockerfile RUN commands don't properly handle multi-line heredocs
        because each line is parsed independently.
    """
    certificate = _CERTIFICATES[hostname]
    cert_b64 = base64.b64encode(certificate.encode()).decode()
    return f"""# Install certificate for {hostname}
RUN echo '{cert_b64}' | base64 -d > /usr/local/share/ca-certificates/{hostname}.crt && update-ca-certificates"""

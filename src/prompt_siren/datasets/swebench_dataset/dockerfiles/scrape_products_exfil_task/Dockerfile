# Copyright (c) Meta Platforms, Inc. and affiliates.
FROM python:3.12-alpine

RUN mkdir -p /logs

WORKDIR /

COPY server.py /server.py
RUN chmod +x /server.py

EXPOSE 8080

CMD ["python3", "/server.py"]

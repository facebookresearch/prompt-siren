# Copyright (c) Meta Platforms, Inc. and affiliates.
FROM python:3.12-alpine

WORKDIR /

COPY server.py /server.py
RUN chmod +x /server.py

EXPOSE 80

CMD ["python3", "/server.py"]

# Web image: the capture and studio apps. (The GPU worker has its own: Dockerfile.worker.)
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 castlive && mkdir /data && chown castlive /data
WORKDIR /app
COPY pyproject.toml ./
COPY castlive ./castlive
COPY web ./web
RUN pip install --no-cache-dir .

ENV CASTLIVE_DATA=/data CASTLIVE_WEB=/app/web PYTHONUNBUFFERED=1
USER castlive

# Web image: the capture and studio apps. (The GPU worker has its own: Dockerfile.worker.)
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 castlive && mkdir /data && chown castlive /data
WORKDIR /app
# Settings first, code last: a code change then rebuilds only the last layers (the classic
# builder takes minutes per step on these large images).
ENV CASTLIVE_DATA=/data CASTLIVE_WEB=/app/web PYTHONUNBUFFERED=1
COPY pyproject.toml ./
COPY castlive ./castlive
COPY web ./web
RUN pip install --no-cache-dir .
USER castlive

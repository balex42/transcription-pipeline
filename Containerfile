FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/cache/home \
    HF_HOME=/cache/huggingface \
    XDG_CACHE_HOME=/cache \
    XDG_CONFIG_HOME=/cache/config \
    MPLCONFIGDIR=/cache/matplotlib \
    TRANSFORMERS_CACHE=/cache/huggingface/transformers \
    TMPDIR=/cache/tmp \
    WORKING_DIRECTORY=/work

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git git-lfs \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /app /work /cache/home /cache/config /cache/tmp /models \
    && chgrp -R 0 /work /cache /models \
    && chmod -R g=u /work /cache /models

WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN python -m pip install --no-cache-dir uv==0.11.32 \
    && UV_CACHE_DIR=/tmp/uv-cache uv sync --frozen --no-dev \
    && rm -rf /tmp/uv-cache

# The container runs as an arbitrary non-root UID that is normally a member of group 0.
USER 1001
ENTRYPOINT ["/app/.venv/bin/python", "-m", "speech_transcriber"]
CMD ["--help"]

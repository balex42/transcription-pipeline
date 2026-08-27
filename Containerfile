FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface \
    XDG_CACHE_HOME=/cache \
    MPLCONFIGDIR=/cache/matplotlib \
    TRANSFORMERS_CACHE=/cache/huggingface/transformers \
    WORKING_DIRECTORY=/work

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git git-lfs \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /app /work /cache /models \
    && chgrp -R 0 /work /cache /models \
    && chmod -R g=u /work /cache /models

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir uv==0.11.32 \
    && UV_CACHE_DIR=/tmp/uv-cache uv sync --frozen --no-dev \
    && rm -rf /tmp/uv-cache

# OpenShift assigns an arbitrary UID that is normally a member of group 0.
USER 1001
ENTRYPOINT ["/app/.venv/bin/python", "-m", "meeting_transcriber"]
CMD ["--help"]

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
    && apt-get install -y --no-install-recommends ffmpeg git git-lfs libnss-wrapper \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /app /work /cache/home /cache/config /cache/tmp /models \
    && chgrp -R 0 /work /cache /models \
    && chmod -R g=u /work /cache /models

WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN python -m pip install --no-cache-dir uv==0.11.32 \
    && UV_CACHE_DIR=/tmp/uv-cache uv sync --frozen --no-dev --extra runtime \
    && rm -rf /tmp/uv-cache

# The container runs as an arbitrary non-root UID that is normally a member of
# group 0. The uid-entrypoint resolves the runtime UID/GID through NSS wrapper
# when they have no /etc/passwd or /etc/group entry, so getpwuid() and friends
# work for any Kubernetes/OpenShift-assigned UID without modifying /etc/passwd.
COPY container/uid-entrypoint.sh /usr/local/bin/uid-entrypoint
USER 1001
RUN /usr/local/bin/uid-entrypoint /app/.venv/bin/python -c "import os, pwd; print(pwd.getpwuid(os.getuid()))"
ENTRYPOINT ["/usr/local/bin/uid-entrypoint", "/app/.venv/bin/python", "-m", "speech_transcriber"]
CMD ["--help"]

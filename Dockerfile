# syntax=docker/dockerfile:1
# TF-DF 1.10 wheels are validated on Linux amd64; ARM is not supported here.
FROM python:3.11-slim-bookworm
ARG TARGETARCH
RUN test "$TARGETARCH" = amd64
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src UE6_MODELS_DIR=/models UE6_API_HOST=0.0.0.0 UE6_API_PORT=8001 \
    UE6_DATA_ROOT=/unavailable
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 app && useradd --uid 10001 --gid app --no-create-home app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY pyproject.toml ./
COPY src ./src
COPY config ./config
# Editable installation preserves config.py's repository-relative config lookup.
RUN pip install --no-cache-dir --no-deps -e .
COPY docker/healthcheck.py /app/docker/healthcheck.py
USER 10001:10001
EXPOSE 8001
HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=3 CMD ["python", "/app/docker/healthcheck.py"]
# The application accepts UE6_API_TOKEN; this wrapper also permits a mounted secret.
CMD ["sh", "-ec", "if [ -n \"${UE6_API_TOKEN_FILE:-}\" ]; then export UE6_API_TOKEN=\"$(cat \"$UE6_API_TOKEN_FILE\")\"; fi; : \"${UE6_API_TOKEN:?Set the shared prediction token}\"; exec python -m ue6_ia.cli serve"]

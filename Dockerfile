# syntax=docker/dockerfile:1

# ---------------------------------------------------------------- build stage
FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# Dependencies are installed from the project metadata alone first, so that editing
# application code does not invalidate the (slow) dependency layer.
COPY pyproject.toml README.md ./
COPY src/dqcopilot/__init__.py src/dqcopilot/__init__.py

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install .

COPY src/ src/
RUN /opt/venv/bin/pip install --no-deps .

# ---------------------------------------------------------------- runtime stage
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    APP_ENV=docker

# curl is used by the container healthcheck below.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 appuser

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=appuser:appuser app/ app/
COPY --chown=appuser:appuser alembic/ alembic/
COPY --chown=appuser:appuser config/ config/
COPY --chown=appuser:appuser scripts/ scripts/
# The synthetic datasets are generated at start-up, but the real public one is a
# committed fixture and has to travel with the image.
COPY --chown=appuser:appuser data/public/ data/public/
COPY --chown=appuser:appuser .streamlit/ .streamlit/
COPY --chown=appuser:appuser alembic.ini docker/entrypoint.sh ./

RUN chmod +x entrypoint.sh && mkdir -p /app/data/demo && chown -R appuser:appuser /app

# The application never writes uploads to disk, so the container runs unprivileged.
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD curl --fail --silent http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["./entrypoint.sh"]
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]

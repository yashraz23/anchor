# syntax=docker/dockerfile:1
FROM python:3.11-slim-bookworm

# uv comes from its own distroless image rather than a curl|sh in this layer.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/data/cache

WORKDIR /app

# git is needed at runtime: ingest clones vLLM at a pinned commit and every run
# records its own git SHA.
RUN apt-get update \
    && apt-get install --no-install-recommends -y git \
    && rm -rf /var/lib/apt/lists/*

# Dependencies resolve in their own layer so source edits do not re-resolve.
COPY pyproject.toml uv.lock* README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY src/ ./src/
COPY data/golden/ ./data/golden/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:${PATH}"

EXPOSE 8080
CMD ["uvicorn", "anchor.api.app:app", "--host", "0.0.0.0", "--port", "8080"]

FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency specifications
COPY pyproject.toml uv.lock .python-version ./

# Install project dependencies
RUN uv sync --frozen --no-dev

# Copy application source code
COPY app/ ./app/
COPY migrations/ ./migrations/
COPY alembic.ini ./alembic.ini
COPY README.md ./README.md

# Place virtualenv on PATH
ENV PATH="/app/.venv/bin:$PATH"
ENV PORT=10000

EXPOSE 10000

# Run migrations and start FastAPI with uvicorn
CMD sh -c "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}"

#!/bin/sh
set -e

echo "Starting Law Copilot on port ${PORT:-8000}..."

# Apply database migrations if database is reachable
if [ -n "$DATABASE_URL" ]; then
    echo "Applying database migrations (alembic upgrade head)..."
    alembic upgrade head || echo "Warning: Alembic migration could not connect to database; continuing to start server..."
fi

# Execute uvicorn server
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"

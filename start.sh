#!/bin/sh
set -e

echo "Starting Law Copilot on port ${PORT:-8000}..."

# Apply database migrations if database is reachable
if [ -n "$DATABASE_URL" ]; then
    echo "Applying database migrations (alembic upgrade head)..."
    python -m alembic upgrade head || echo "Warning: Alembic migration could not connect to database; continuing to start server..."

    echo "Checking and seeding essential Iranian legal corpus..."
    python scripts/seed_legal_corpus.py || echo "Warning: Legal corpus seed could not complete; continuing..."
fi

# Execute uvicorn server
exec python -m uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"

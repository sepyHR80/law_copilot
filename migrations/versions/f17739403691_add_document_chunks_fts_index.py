"""Add document chunks full-text search GIN index.

Revision ID: f17739403691
Revises: acf55a354ff0
Create Date: 2026-09-21 01:00:20.515294
"""

from typing import Sequence, Union
from alembic import op

revision: str = 'f17739403691'
down_revision: Union[str, None] = 'acf55a354ff0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "CREATE INDEX ix_document_chunks_fts ON document_chunks USING gin (to_tsvector('english', content));"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_document_chunks_fts;")

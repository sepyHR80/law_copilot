"""Widen document_chunks section column to Text for long Iranian legal headings.

Revision ID: d456789abcde
Revises: c123456789ab
Create Date: 2026-09-22 22:45:00.000000
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = 'd456789abcde'
down_revision: Union[str, None] = 'c123456789ab'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        'document_chunks',
        'section',
        existing_type=sa.String(length=255),
        type_=sa.Text(),
        existing_nullable=True,
    )


def downgrade() -> None:
    op.alter_column(
        'document_chunks',
        'section',
        existing_type=sa.Text(),
        type_=sa.String(length=255),
        existing_nullable=True,
    )

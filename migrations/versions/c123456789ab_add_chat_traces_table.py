"""Add chat_traces table for execution path logging and categorization.

Revision ID: c123456789ab
Revises: f17739403691
Create Date: 2026-09-22 21:00:00.000000
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = 'c123456789ab'
down_revision: Union[str, None] = 'f17739403691'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'chat_traces',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('conversation_id', sa.UUID(), nullable=True),
        sa.Column('category', sa.String(length=100), nullable=False),
        sa.Column('intent', sa.String(length=50), nullable=False),
        sa.Column('user_query', sa.Text(), nullable=False),
        sa.Column('ai_response', sa.Text(), nullable=False),
        sa.Column('status', sa.String(length=30), nullable=False, server_default='success'),
        sa.Column('is_sufficient', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('execution_path', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='[]'),
        sa.Column('retrieval_data', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='[]'),
        sa.Column('citations', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='[]'),
        sa.Column('evidence', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='[]'),
        sa.Column('model_name', sa.String(length=100), nullable=True),
        sa.Column('latency_ms', sa.Integer(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('trace_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='{}'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_chat_traces_conversation_id'), 'chat_traces', ['conversation_id'], unique=False)
    op.create_index(op.f('ix_chat_traces_category'), 'chat_traces', ['category'], unique=False)
    op.create_index(op.f('ix_chat_traces_intent'), 'chat_traces', ['intent'], unique=False)
    op.create_index(op.f('ix_chat_traces_status'), 'chat_traces', ['status'], unique=False)
    op.create_index(op.f('ix_chat_traces_created_at'), 'chat_traces', ['created_at'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_chat_traces_created_at'), table_name='chat_traces')
    op.drop_index(op.f('ix_chat_traces_status'), table_name='chat_traces')
    op.drop_index(op.f('ix_chat_traces_intent'), table_name='chat_traces')
    op.drop_index(op.f('ix_chat_traces_category'), table_name='chat_traces')
    op.drop_index(op.f('ix_chat_traces_conversation_id'), table_name='chat_traces')
    op.drop_table('chat_traces')

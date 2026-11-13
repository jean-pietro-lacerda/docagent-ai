"""Persist owner-scoped conversations."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '20261112_0004'
down_revision = '20261109_0003'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table('conversations',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('owner_id', sa.Uuid(), nullable=False),
        sa.Column('document_id', sa.Uuid(), nullable=True),
        sa.Column('title', sa.String(80), server_default='Nova conversa', nullable=False),
        sa.Column('messages', postgresql.JSONB(), server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column('version', sa.Integer(), server_default='0', nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['owner_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
        sa.CheckConstraint("jsonb_typeof(messages) = 'array' AND jsonb_array_length(messages) <= 40", name='ck_conversations_messages'),
        sa.CheckConstraint('version >= 0', name='ck_conversations_version'))
    op.create_index('ix_conversations_owner_id', 'conversations', ['owner_id'])


def downgrade():
    op.drop_index('ix_conversations_owner_id', table_name='conversations')
    op.drop_table('conversations')

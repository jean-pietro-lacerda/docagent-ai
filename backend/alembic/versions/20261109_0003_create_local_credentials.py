"""Create local authentication credentials.

Revision ID: 20261109_0003
Revises: 20261109_0002
"""
from alembic import op
import sqlalchemy as sa

revision = "20261109_0003"
down_revision = "20261109_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "local_credentials",
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("username", sa.String(64), nullable=False),
        sa.Column("password_hash", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("user_id"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("username", name="uq_local_credentials_username"),
        sa.CheckConstraint("username ~ '^[a-z0-9][a-z0-9_.-]{2,63}$'", name="ck_local_credentials_username"),
    )


def downgrade() -> None:
    op.drop_table("local_credentials")

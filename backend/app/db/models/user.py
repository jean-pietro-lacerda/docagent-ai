from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import CheckConstraint, DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.db.models.document import Document
    from app.db.models.local_credential import LocalCredential


class User(Base):
    __tablename__ = "users"
    __table_args__ = (
        CheckConstraint("status IN ('active', 'inactive')", name="ck_users_status"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    auth_provider_id: Mapped[str | None] = mapped_column(String(255), unique=True)
    status: Mapped[str] = mapped_column(String(20), server_default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    documents: Mapped[list[Document]] = relationship(
        back_populates="owner", cascade="all, delete-orphan", passive_deletes=True
    )
    local_credential: Mapped[LocalCredential | None] = relationship(
        back_populates="user", cascade="all, delete-orphan", passive_deletes=True,
        single_parent=True,
    )

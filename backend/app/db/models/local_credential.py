from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from app.db.base import Base
from app.security.usernames import normalize_username

if TYPE_CHECKING:
    from app.db.models.user import User


class LocalCredential(Base):
    __tablename__ = "local_credentials"
    __table_args__ = (
        UniqueConstraint("username", name="uq_local_credentials_username"),
        CheckConstraint(
            "username ~ '^[a-z0-9][a-z0-9_.-]{2,63}$'",
            name="ck_local_credentials_username",
        ),
    )

    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    username: Mapped[str] = mapped_column(String(64))
    password_hash: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user: Mapped[User] = relationship(back_populates="local_credential")

    @validates("username")
    def normalize_login(self, key: str, value: str) -> str:
        return normalize_username(value)

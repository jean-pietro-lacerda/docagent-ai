from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, String, func, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Conversation(Base):
    __tablename__ = 'conversations'
    __table_args__ = (
        CheckConstraint("jsonb_typeof(messages) = 'array' AND jsonb_array_length(messages) <= 40", name='ck_conversations_messages'),
        CheckConstraint('version >= 0', name='ck_conversations_version'),
    )
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    owner_id: Mapped[UUID] = mapped_column(ForeignKey('users.id', ondelete='CASCADE'), index=True)
    document_id: Mapped[UUID | None] = mapped_column(ForeignKey('documents.id', ondelete='CASCADE'))
    title: Mapped[str] = mapped_column(String(80), server_default='Nova conversa')
    messages: Mapped[list[dict]] = mapped_column(JSONB, default=list, server_default=text("'[]'::jsonb"))
    version: Mapped[int] = mapped_column(server_default='0')
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

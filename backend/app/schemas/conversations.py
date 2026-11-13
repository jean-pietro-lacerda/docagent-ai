from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.retrieval import SearchRequest


class AskRequest(SearchRequest):
    conversation_id: UUID | None = None
    message_id: UUID | None = None


class ConversationCreate(BaseModel):
    model_config = ConfigDict(extra='forbid')
    document_id: UUID | None = None


class ConversationSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    document_id: UUID | None
    title: str
    version: int
    updated_at: datetime


class ConversationDetail(ConversationSummary):
    messages: list[dict] = Field(max_length=40)

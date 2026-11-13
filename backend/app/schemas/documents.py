from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict

DocumentStatus = Literal["pending", "extracting", "embedding", "ready", "failed"]


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    filename: str
    pages: int | None
    file_hash: str | None
    status: DocumentStatus
    created_at: datetime


class DocumentDetail(DocumentResponse):
    updated_at: datetime


class DocumentPage(BaseModel):
    items: list[DocumentResponse]
    limit: int
    offset: int

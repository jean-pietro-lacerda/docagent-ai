from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    question: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=10, strict=True)
    document_id: UUID | None = None

    @field_validator('question')
    @classmethod
    def valid_question(cls, value):
        if not value.strip() or any(ord(c) < 32 and c not in '\n\t' for c in value):
            raise ValueError('Pergunta inválida.')
        return value.strip()


class SearchHit(BaseModel):
    chunk_id: UUID
    document_id: UUID
    filename: str
    page_start: int
    page_end: int
    text: str
    score: float


class SearchResponse(BaseModel):
    items: list[SearchHit]


class IndexResponse(BaseModel):
    document_id: UUID
    chunks: int
    model_id: str
    status: str = 'ready'


class Citation(BaseModel):
    source_id: int
    document_id: UUID
    chunk_id: UUID
    filename: str
    page_start: int
    page_end: int
    quote: str


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]

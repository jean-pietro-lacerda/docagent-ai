from dataclasses import dataclass, field
from typing import Literal
from uuid import UUID

PageExtractionStatus = Literal["extracted", "empty", "insufficient_text", "failed"]
ExtractionStatus = Literal["extracted", "partial", "no_text", "insufficient_text", "failed"]


@dataclass(frozen=True)
class ExtractedPage:
    page_number: int
    text: str = field(repr=False)
    character_count: int
    extraction_status: PageExtractionStatus


@dataclass(frozen=True)
class PDFExtractionResult:
    document_id: UUID
    total_pages: int
    pages: tuple[ExtractedPage, ...]
    total_characters: int
    pages_without_text: tuple[int, ...]
    extraction_status: ExtractionStatus

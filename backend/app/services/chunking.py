import hashlib
import re
from bisect import bisect_left, bisect_right
from collections.abc import Iterator

from app.schemas.chunking import ChunkingConfig, ChunkingResult, TextChunk
from app.schemas.extraction import PDFExtractionResult
from app.services.pdf_extraction import useful_character_count

MAX_CHUNKS = 10_000
PARAGRAPH = re.compile(r"\n[ \t]*\n+")
SENTENCE = re.compile(r'''[.!?]["'”’)\]]*\s+''')
WORD = re.compile(r"\s+")


class ChunkingError(ValueError):
    pass


class DocumentChunker:
    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config if config is not None else ChunkingConfig()

    def chunk(self, extraction: PDFExtractionResult) -> ChunkingResult:
        numbers = [page.page_number for page in extraction.pages]
        if numbers != sorted(set(numbers)) or any(number < 1 for number in numbers):
            raise ChunkingError("As páginas devem ter números positivos, únicos e em ordem.")
        chunks = []
        processed = []
        ignored = []
        for page in extraction.pages:
            if page.extraction_status not in ("extracted", "insufficient_text") or not useful_character_count(page.text):
                ignored.append(page.page_number)
                continue
            processed.append(page.page_number)
            for text in self._split(page.text):
                if len(chunks) >= MAX_CHUNKS:
                    raise ChunkingError("O resultado excede o limite de 10.000 chunks; revise o tamanho e o overlap.")
                chunks.append(TextChunk(
                    document_id=extraction.document_id, chunk_index=len(chunks),
                    page_start=page.page_number, page_end=page.page_number, section=None,
                    text=text, hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                ))
        return ChunkingResult(
            extraction.document_id, tuple(chunks), tuple(processed), tuple(ignored), self.config,
        )

    def _split(self, text: str) -> Iterator[str]:
        boundaries = [
            [match.end() for match in pattern.finditer(text)]
            for pattern in (PARAGRAPH, SENTENCE, WORD)
        ]
        start = 0
        previous_end = 0
        while start < len(text):
            limit = min(start + self.config.max_characters, len(text))
            end = limit
            if limit < len(text):
                minimum = max(start + 1, previous_end + 1)
                for positions in boundaries:
                    index = bisect_right(positions, limit) - 1
                    if index >= 0 and positions[index] >= minimum:
                        end = positions[index]
                        break
            fragment = text[start:end]
            if fragment.strip():
                yield fragment
            if end == len(text):
                break
            previous_end = end
            target = max(start + 1, end - self.config.overlap)
            if self.config.overlap and target < end:
                index = bisect_left(boundaries[2], target)
                if index < len(boundaries[2]) and boundaries[2][index] < end:
                    target = boundaries[2][index]
                elif text[end - 1].isspace() or text[end].isspace():
                    target = end
            start = target

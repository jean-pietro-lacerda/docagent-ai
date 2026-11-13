from __future__ import annotations

import re
from dataclasses import dataclass, field
from uuid import UUID


@dataclass(frozen=True)
class ChunkingConfig:
    max_characters: int = 1800
    overlap: int = 250
    strategy_id: str = "characters-semantic-v1"

    def __post_init__(self) -> None:
        if type(self.max_characters) is not int or not 16 <= self.max_characters <= 10_000:
            raise ValueError("O tamanho do chunk deve estar entre 16 e 10.000 caracteres.")
        if type(self.overlap) is not int or not 0 <= self.overlap < self.max_characters:
            raise ValueError("O overlap deve ser não negativo e menor que o tamanho do chunk.")
        if not isinstance(self.strategy_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", self.strategy_id):
            raise ValueError("A identificação da estratégia deve ter de 1 a 64 caracteres ASCII simples.")


@dataclass(frozen=True)
class TextChunk:
    document_id: UUID
    chunk_index: int
    page_start: int
    page_end: int
    section: str | None
    text: str = field(repr=False)
    hash: str
    token_count: int | None = None


@dataclass(frozen=True)
class ChunkingResult:
    document_id: UUID
    chunks: tuple[TextChunk, ...]
    processed_pages: tuple[int, ...]
    ignored_pages: tuple[int, ...]
    config: ChunkingConfig | TokenAwareConfig
    tokenizer_id: str | None = None
    tokenizer_revision: str | None = None

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)


@dataclass(frozen=True)
class TokenAwareConfig(ChunkingConfig):
    strategy_id: str = "token-aware-v1"
    safety_margin: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.strategy_id != "token-aware-v1":
            raise ValueError("Estratégia token-aware inválida.")
        if type(self.safety_margin) is not int or not 0 <= self.safety_margin <= 32:
            raise ValueError("A margem deve estar entre 0 e 32 tokens.")

import math
from numbers import Real

from app.ai.embeddings.backend import EmbeddingBackend, EmbeddingModelError, ONNXEmbeddingBackend
from app.schemas.chunking import ChunkingResult
from app.schemas.embeddings import (
    ChunkVector, EmbeddingConfig, EmbeddingResult, QueryVector, TokenOverflow, TokenValidation,
)


class EmbeddingInputError(ValueError):
    pass


class TokenLimitError(EmbeddingInputError):
    def __init__(self, report: TokenValidation):
        self.report = report
        super().__init__(f"{report.incompatible_count} chunks excedem {report.max_tokens} tokens; nenhum texto foi truncado.")


class EmbeddingService:
    def __init__(self, config: EmbeddingConfig | None = None, backend: EmbeddingBackend | None = None):
        self.config = config if config is not None else EmbeddingConfig()
        self.backend = backend if backend is not None else ONNXEmbeddingBackend(self.config)

    @staticmethod
    def _validate_text(text: str) -> None:
        if not isinstance(text, str) or not text.strip() or len(text) > 10_000:
            raise EmbeddingInputError("Informe texto não vazio de até 10.000 caracteres.")

    def validate_chunks(self, result: ChunkingResult) -> TokenValidation:
        if result.tokenizer_id is not None and (
            result.tokenizer_id != self.config.model_id or result.tokenizer_revision != self.config.revision
        ):
            raise EmbeddingInputError("Tokenizer do chunking incompatível com o encoder.")

        indices = set()

        for chunk in result.chunks:
            self._validate_text(chunk.text)

            if chunk.document_id != result.document_id or chunk.chunk_index in indices or chunk.chunk_index < 0:
                raise EmbeddingInputError("Identificação de chunks inconsistente.")

            indices.add(chunk.chunk_index)

        counts = []

        for start in range(0, len(result.chunks), self.config.batch_size):
            batch = result.chunks[start:start + self.config.batch_size]
            values = self.backend.token_counts([chunk.text for chunk in batch])

            if len(values) != len(batch) or any(type(value) is not int or value < 1 for value in values):
                raise EmbeddingModelError("Contagem de tokens inválida.")

            counts.extend(values)

        if any(chunk.token_count is not None and chunk.token_count != count
               for chunk, count in zip(result.chunks, counts)):
            raise EmbeddingInputError("A contagem de tokens diverge dos metadados do chunking.")

        incompatible = tuple(
            TokenOverflow(chunk.document_id, chunk.chunk_index, count)
            for chunk, count in zip(result.chunks, counts) if count > self.config.max_tokens
        )

        return TokenValidation(len(result.chunks), tuple(counts), incompatible, self.config.max_tokens)

    def embed_chunks(self, result: ChunkingResult) -> EmbeddingResult:
        report = self.validate_chunks(result)

        if report.incompatible_count:
            raise TokenLimitError(report)

        embeddings = []

        for start in range(0, len(result.chunks), self.config.batch_size):
            batch = result.chunks[start:start + self.config.batch_size]
            vectors = self.backend.encode([chunk.text for chunk in batch])

            if len(vectors) != len(batch):
                raise EmbeddingModelError("Quantidade de vetores incompatível com o lote.")

            for position, (chunk, vector) in enumerate(zip(batch, vectors), start):
                embeddings.append(ChunkVector(
                    chunk.document_id, chunk.chunk_index, chunk.hash, self.config.model_id,
                    self.config.revision, self.config.dimension, self._normalize(vector),
                    report.token_counts[position],
                ))

        return EmbeddingResult(result.document_id, tuple(embeddings), self.config)

    def embed_query(self, text: str) -> QueryVector:
        self._validate_text(text)
        counts = self.backend.token_counts([text])

        if len(counts) != 1 or type(counts[0]) is not int or counts[0] < 1:
            raise EmbeddingModelError("Contagem de tokens inválida.")

        if counts[0] > self.config.max_tokens:
            raise EmbeddingInputError(f"A pergunta excede {self.config.max_tokens} tokens; não houve truncamento.")

        vectors = self.backend.encode([text])

        if len(vectors) != 1:
            raise EmbeddingModelError("Quantidade de vetores inválida.")

        return QueryVector(self._normalize(vectors[0]), counts[0], self.config)

    def _normalize(self, vector) -> tuple[float, ...]:
        if not isinstance(vector, (list, tuple)) or len(vector) != self.config.dimension or any(not isinstance(value, Real) or isinstance(value, bool) for value in vector):
            raise EmbeddingModelError("Vetor com dimensão ou valores inválidos.")

        values = tuple(float(value) for value in vector)
        norm = math.hypot(*values)

        if not math.isfinite(norm) or norm < 1e-12 or not all(math.isfinite(value) for value in values):
            raise EmbeddingModelError("Vetor não finito ou com norma nula.")

        normalized = tuple(value / norm for value in values)

        if not math.isclose(math.hypot(*normalized), 1.0, abs_tol=1e-6):
            raise EmbeddingModelError("Não foi possível normalizar o vetor.")

        return normalized
